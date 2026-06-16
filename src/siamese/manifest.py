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

# Rotulos
LABEL_NO_ERROR = 0
LABEL_ERROR = 1

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
    source: str          # 'no_erros' | 'with_errors'
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


def scan_dataset(input_dir: Path) -> list[Sample]:
    """Le data/input/{no_erros,with_errors} e devolve a lista de Samples (sem split)."""
    samples: list[Sample] = []
    specs = [("no_erros", LABEL_NO_ERROR), ("with_errors", LABEL_ERROR)]
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for sub, label in specs:
        d = input_dir / sub
        if not d.is_dir():
            raise FileNotFoundError(f"Pasta esperada nao encontrada: {d}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            meta = _parse_meta(p.name, sub)
            samples.append(Sample(
                path=str(p.resolve()),
                label=label,
                group=_group_key(p, sub),
                source=sub,
                **meta,
            ))
    return samples


def grouped_stratified_split(
    samples: list[Sample],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> list[Sample]:
    """Split deterministico, ESTRATIFICADO por classe e AGRUPADO por `group`.

    Estrategia: para cada classe separadamente (estratificacao), embaralha os grupos
    de forma deterministica e os aloca a test/val/train por contagem cumulativa de
    imagens ate atingir as fracoes-alvo. Como cada grupo inteiro vai para um unico
    split, nao ha vazamento de ticket entre splits.
    """
    import random

    rng = random.Random(seed)
    by_label: dict[int, dict[str, list[Sample]]] = {}
    for s in samples:
        by_label.setdefault(s.label, {}).setdefault(s.group, []).append(s)

    for label, groups in by_label.items():
        group_ids = list(groups.keys())
        # ordena por nome para estabilidade e depois embaralha com seed
        group_ids.sort()
        rng.shuffle(group_ids)
        total = sum(len(groups[g]) for g in group_ids)
        n_test_target = round(total * test_frac)
        n_val_target = round(total * val_frac)

        n_test = n_val = 0
        for g in group_ids:
            sz = len(groups[g])
            if n_test < n_test_target:
                split = "test"
                n_test += sz
            elif n_val < n_val_target:
                split = "val"
                n_val += sz
            else:
                split = "train"
            for s in groups[g]:
                s.split = split
    return samples


FIELDS = [
    "path", "label", "group", "split", "form_factor",
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
