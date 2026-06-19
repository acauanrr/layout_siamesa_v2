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

# Regex de metadados (case-insensitive)
_RE_TICKET = re.compile(r"(IKSWW[-_]\d+)", re.IGNORECASE)
_RE_FORMFACTOR = re.compile(r"(unfold|fold|laptop|tent|desktop|unidentif\w*)", re.IGNORECASE)
_RE_ORIENT = re.compile(r"(portrait|portrair|landscape)", re.IGNORECASE)  # 'portrair' = typo no dataset
_RE_KIND = re.compile(r"(screenshot|screeshot|photo)", re.IGNORECASE)     # 'screeshot' = typo no dataset


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

    # 3. aloca cada estrato a test/val/train
    for _strat in sorted(by_strat, key=str):
        gmap = by_strat[_strat]
        group_ids = sorted(gmap.keys())  # estabilidade
        rng.shuffle(group_ids)
        total = sum(len(gmap[g]) for g in group_ids)
        n_test_target = round(total * test_frac)
        n_val_target = round(total * val_frac)

        n_test = n_val = 0
        for g in group_ids:
            sz = len(gmap[g])
            if n_test < n_test_target:
                split = "test"
                n_test += sz
            elif n_val < n_val_target:
                split = "val"
                n_val += sz
            else:
                split = "train"
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
