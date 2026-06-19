"""Construcao do manifesto do dataset e split agrupado por ticket (sem vazamento).

Por que um modulo dedicado:
- Varias imagens de `with_errors` compartilham o mesmo ticket IKSWW (ex.: 3 screenshots
  do mesmo bug). Se duas delas caissem em splits diferentes (uma em train, outra em test),
  o modelo "veria" o caso de teste durante o treino -> vazamento e metrica inflada.
  Por isso o split e feito por GRUPO (ticket), nunca por imagem.
- Os nomes de arquivo carregam metadados (form factor, orientacao, foto/screenshot,
  competitor, boundBox) que sao CONFOUNDS conhecidos. Extrai-los aqui permite auditar
  depois se o modelo aprende o erro ou o confound.

Saida: CSVs em data/splits/{train,val,test}.csv com colunas:
  path,label,group,split,form_factor,orientation,kind,is_competitor,has_boundbox,source
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Rotulos binarios (mantidos p/ o gate limpo-vs-erro e p/ a esteira binaria legada)
LABEL_NO_ERROR = 0
LABEL_ERROR = 1

# --- Taxonomia multi-cluster -------------------------------------------------
# As 6 categorias de erro espelham as subpastas reais de data/input/errors_dataset/
# (nomes com ESPACO -> slug estavel). 'clean' e a classe sem-erro (no_erros/).
# A ordem de CATEGORIES define o id numerico usado no treino/decisao multi-classe:
#   0 = clean, 1..6 = categorias de erro (em ordem alfabetica do slug, deterministico).
ERROR_DIR_TO_SLUG = {
    "black bars": "black_bars",
    "disordered layout": "disordered_layout",
    "distortion": "distortion",
    "empty space": "empty_space",
    "orientation": "orientation",
    "overlay": "overlay",
}
CLEAN_CATEGORY = "clean"
CATEGORIES = [CLEAN_CATEGORY] + sorted(ERROR_DIR_TO_SLUG.values())
CATEGORY_TO_ID = {c: i for i, c in enumerate(CATEGORIES)}
ID_TO_CATEGORY = {i: c for c, i in CATEGORY_TO_ID.items()}
CLEAN_ID = CATEGORY_TO_ID[CLEAN_CATEGORY]   # 0
ABSTAIN_ID = -1                              # sentinela de abstencao (categoria fora do escopo)


def category_id(cat: str, *, strict: bool = True) -> int:
    """Mapa CANONICO categoria->id (Fase 6, 'tratamento de escopo'). 'clean'/'' -> 0; slug de
    erro conhecido -> seu id. Categoria de ERRO DESCONHECIDA: strict=True (padrao) levanta
    KeyError; strict=False devolve ABSTAIN_ID (-1). NUNCA rotula desconhecido como 'clean'
    silenciosamente (era o bug de `CATEGORY_TO_ID.get(c, 0)`)."""
    c = (cat or "").strip().lower()
    if c in ("", CLEAN_CATEGORY):
        return CLEAN_ID
    if c in CATEGORY_TO_ID:
        return CATEGORY_TO_ID[c]
    if strict:
        raise KeyError(
            f"categoria fora do escopo: {cat!r} (conhecidas: {list(CATEGORY_TO_ID)}). "
            "Mapeie-a, funda-a, ou trate como abstencao — nunca rotule como 'clean'.")
    return ABSTAIN_ID

# Regex de metadados (case-insensitive)
_RE_TICKET = re.compile(r"(IKSWW[-_]\d+)", re.IGNORECASE)
_RE_FORMFACTOR = re.compile(r"(unfold|fold|laptop|tent|desktop|unidentif\w*)", re.IGNORECASE)
_RE_ORIENT = re.compile(r"(portrait|portrair|landscape)", re.IGNORECASE)  # 'portrair' = typo no dataset
_RE_KIND = re.compile(r"(screenshot|screeshot|photo)", re.IGNORECASE)     # 'screeshot' = typo no dataset
# Timestamp de captura embutido no nome (Screenshot_YYYYMMDD_HHMMSS.png) -> identifica SESSAO.
_RE_SHOT_TS = re.compile(r"(\d{8})[_-](\d{6})")


@dataclass
class Sample:
    path: str
    label: int
    group: str
    form_factor: str
    orientation: str
    kind: str            # 'screenshot' | 'photo'
    is_competitor: bool
    has_boundbox: bool
    source: str          # 'no_erros' | 'errors_dataset' | 'with_errors'
    category: str = ""   # 'clean' | slug da categoria de erro | '' (legado with_errors)
    split: str = ""      # preenchido depois


def _parse_meta(filename: str, source: str) -> dict:
    name = filename.lower()
    ff = _RE_FORMFACTOR.search(name)
    form_factor = ff.group(1).lower() if ff else "unknown"
    if form_factor.startswith("unidentif"):
        form_factor = "unidentified"

    orient = _RE_ORIENT.search(name)
    orientation = orient.group(1).lower().replace("portrair", "portrait") if orient else "unknown"

    kind_m = _RE_KIND.search(name)
    if kind_m:
        k = kind_m.group(1).lower()
        kind = "photo" if k == "photo" else "screenshot"
    else:
        # no_erros sao todos screenshots de tela; erros sem marcador assumimos screenshot
        kind = "screenshot"

    return {
        "form_factor": form_factor,
        "orientation": orientation,
        "kind": kind,
        "is_competitor": "competitor" in name,
        "has_boundbox": "boundbox" in name or "bound_box" in name,
    }


def _group_key(path: Path, source: str) -> str:
    """Chave de agrupamento para o split sem vazamento.

    - with_errors: ticket IKSWW se existir; senao o stem do arquivo (grupo unitario).
    - no_erros: cada imagem e independente (sem identidade compartilhada) -> stem unico.
    """
    m = _RE_TICKET.search(path.name)
    if m:
        # normaliza IKSWW_123 e IKSWW-123 para a mesma chave
        return m.group(1).upper().replace("_", "-")
    return f"{source}:{path.stem}"


def _shot_time(name: str) -> datetime | None:
    """Extrai o datetime de captura de um Screenshot_YYYYMMDD_HHMMSS; None se nao houver."""
    m = _RE_SHOT_TS.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def clean_session_components(
    names: list[str],
    *,
    gap_seconds: int = 300,
    phash_of=None,
    phash_max_dist: int = 6,
) -> dict[str, str]:
    """Agrupa nomes de telas LIMPAS em COMPONENTES CONEXOS de captura, via union-find sobre
    duas relacoes (problema #3 da auditoria — telas limpas sao quase-duplicatas da mesma
    sessao/dispositivo e NAO podem cruzar splits):

      (1) SESSAO temporal: capturas com timestamps consecutivos a <= `gap_seconds` caem na
          mesma sessao (transitivo). Sequencias `Screenshot_YYYYMMDD_HHMMSS` tiradas segundos
          a segundos sao o nucleo do vazamento.
      (2) NEAR-DUPLICATE perceptual (opcional): se `phash_of(name)->int` for dado, qualquer
          par com distancia de Hamming <= `phash_max_dist` e' unido — pega duplicatas mesmo
          em sessoes/dias diferentes.

    Retorna {name: group_id}, com ids estaveis `no_erros:sessNNN` ordenados pelo menor nome
    do componente (deterministico, independente da ordem de entrada)."""
    parent = {n: n for n in names}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:        # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # (1) sessao temporal: ordena por tempo e une vizinhos dentro do gap
    timed = sorted(((t, n) for n in names if (t := _shot_time(n)) is not None),
                   key=lambda x: (x[0], x[1]))
    for (ta, na), (tb, nb) in zip(timed, timed[1:]):
        if (tb - ta).total_seconds() <= gap_seconds:
            union(na, nb)

    # (2) near-duplicate perceptual (O(n^2); n~100s de limpas -> trivial)
    if phash_of is not None:
        hh = [(n, phash_of(n)) for n in names]
        for i in range(len(hh)):
            ni, hi = hh[i]
            if hi is None:
                continue
            for j in range(i + 1, len(hh)):
                nj, hj = hh[j]
                if hj is not None and _hamming(hi, hj) <= phash_max_dist:
                    union(ni, nj)

    comps: dict[str, list[str]] = {}
    for n in names:
        comps.setdefault(find(n), []).append(n)
    ordered = sorted(comps.values(), key=lambda g: min(g))
    out: dict[str, str] = {}
    for k, members in enumerate(ordered):
        gid = f"no_erros:sess{k:03d}"
        for n in members:
            out[n] = gid
    return out


def assign_clean_session_groups(
    samples: list["Sample"],
    *,
    gap_seconds: int = 300,
    phash_of=None,
    phash_max_dist: int = 6,
) -> list["Sample"]:
    """Reescreve `group` das amostras LIMPAS (source 'no_erros') para o componente de sessao
    (ver clean_session_components). Idempotente; nao toca amostras de erro (agrupadas por
    ticket). Chamado por build_splits ANTES do split para impedir vazamento de quase-duplicatas."""
    clean = [s for s in samples if s.source == "no_erros"]
    if not clean:
        return samples
    by_name = {}
    for s in clean:
        by_name.setdefault(Path(s.path).name, []).append(s)
    groups = clean_session_components(
        list(by_name.keys()), gap_seconds=gap_seconds,
        phash_of=phash_of, phash_max_dist=phash_max_dist)
    for name, members in by_name.items():
        for s in members:
            s.group = groups[name]
    return samples


_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _scan_dir(d: Path, *, label: int, source: str, category: str) -> list[Sample]:
    samples: list[Sample] = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in _EXTS:
            continue
        samples.append(Sample(
            path=str(p.resolve()),
            label=label,
            group=_group_key(p, source),
            source=source,
            category=category,
            **_parse_meta(p.name, source),
        ))
    return samples


def scan_dataset(input_dir: Path, source: str = "errors_dataset") -> list[Sample]:
    """Le data/input e devolve a lista de Samples (sem split).

    A classe LIMPA vem sempre de `no_erros/` (label 0, category='clean').

    source="errors_dataset" (PADRAO): os erros vem de `errors_dataset/<categoria>/`,
        preservando a CATEGORIA do erro (multi-cluster). Cada subpasta vira label 1 +
        category=<slug> (ver ERROR_DIR_TO_SLUG).
    source="with_errors" (LEGADO): os erros vem da pasta flat `with_errors/` (binario),
        com category='' . Mantido para reproduzir os resultados binarios antigos.

    As fontes sao EXCLUSIVAS (nunca as duas juntas): >=40 nomes coincidem entre
    with_errors/ e errors_dataset/, e misturar criaria duplicatas/vazamento.
    """
    if source not in ("errors_dataset", "with_errors"):
        raise ValueError(f"source invalido: {source!r} (use 'errors_dataset' ou 'with_errors')")

    clean_dir = input_dir / "no_erros"
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Pasta esperada nao encontrada: {clean_dir}")
    samples = _scan_dir(clean_dir, label=LABEL_NO_ERROR, source="no_erros", category=CLEAN_CATEGORY)

    if source == "errors_dataset":
        base = input_dir / "errors_dataset"
        if not base.is_dir():
            raise FileNotFoundError(f"Pasta esperada nao encontrada: {base}")
        for folder, slug in ERROR_DIR_TO_SLUG.items():
            d = base / folder
            if not d.is_dir():
                continue  # categoria ausente -> ignora (sem quebrar)
            samples += _scan_dir(d, label=LABEL_ERROR, source="errors_dataset", category=slug)
    else:  # with_errors (legado binario)
        d = input_dir / "with_errors"
        if not d.is_dir():
            raise FileNotFoundError(f"Pasta esperada nao encontrada: {d}")
        samples += _scan_dir(d, label=LABEL_ERROR, source="with_errors", category="")

    return samples


def grouped_stratified_split(
    samples: list[Sample],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    stratify: str = "category",
) -> list[Sample]:
    """Split deterministico, AGRUPADO por `group` (ticket) e ESTRATIFICADO.

    - Agrupamento: todas as imagens de um mesmo ticket vao para o MESMO split
      (grupo atomico) -> sem vazamento. Verificado em build_splits.py.
    - Estratificacao (`stratify`):
        "category" (PADRAO, multi-cluster): estratifica pela CATEGORIA do grupo, para
            que cada categoria (inclusive raras como orientation/distortion) apareca em
            train/val/test. A chave do grupo e a categoria MAJORITARIA entre suas
            imagens (resolve os poucos tickets que cruzam categorias sem vazar).
        "label" (legado binario): estratifica por rotulo 0/1 (comportamento antigo).
      No modo legado `with_errors` a categoria e '' para todo erro, entao "category"
      degenera para 2 estratos {clean, ''} == binario.

    Para cada estrato: embaralha os grupos deterministicamente e os aloca a test/val/
    train por contagem cumulativa de imagens ate atingir as fracoes-alvo.
    """
    import random
    from collections import Counter

    rng = random.Random(seed)

    # 1. agrupa por ticket (grupo atomico)
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.group, []).append(s)

    # 2. chave de estrato por grupo
    def strat_key(members: list[Sample]):
        if stratify == "category":
            return Counter(m.category for m in members).most_common(1)[0][0]
        return members[0].label

    by_strat: dict[object, dict[str, list[Sample]]] = {}
    for gid, members in groups.items():
        by_strat.setdefault(strat_key(members), {})[gid] = members

    # 3. aloca cada estrato a train/val/test de forma BALANCEADA (deficit-fill). Aloca os
    #    grupos do MAIOR para o menor, cada um ao split mais DEFICITARIO (target - ja_alocado).
    #    Isso mantem as fracoes mesmo com POUCOS grupos grandes (ex.: 15 sessoes limpas) —
    #    o greedy antigo "enche test, depois val, depois train" deixava o train sem limpas.
    train_frac = max(0.0, 1.0 - val_frac - test_frac)
    for _strat in sorted(by_strat, key=str):
        gmap = by_strat[_strat]
        group_ids = sorted(gmap.keys())  # estabilidade
        rng.shuffle(group_ids)           # desempate determinístico (seed)
        total = sum(len(gmap[g]) for g in group_ids)
        targets = {"train": total * train_frac, "val": total * val_frac, "test": total * test_frac}
        filled = {"train": 0.0, "val": 0.0, "test": 0.0}
        # ordem: maior grupo primeiro; estavel sobre a ordem ja embaralhada (desempata size)
        for g in sorted(group_ids, key=lambda g: -len(gmap[g])):
            sz = len(gmap[g])
            split = max(("train", "val", "test"), key=lambda s: targets[s] - filled[s])
            filled[split] += sz
            for s in gmap[g]:
                s.split = split
    return samples


FIELDS = [
    "path", "label", "category", "group", "split", "form_factor",
    "orientation", "kind", "is_competitor", "has_boundbox", "source",
]


def write_manifests(samples: list[Sample], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    # manifesto completo
    with (out_dir / "all.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for s in samples:
            row = {k: asdict(s)[k] for k in FIELDS}
            w.writerow(row)
    # por split
    for split in ("train", "val", "test"):
        rows = [s for s in samples if s.split == split]
        counts[split] = len(rows)
        with (out_dir / f"{split}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for s in rows:
                w.writerow({k: asdict(s)[k] for k in FIELDS})
    return counts
