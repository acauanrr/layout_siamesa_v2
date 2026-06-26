#!/usr/bin/env python
"""RECONSTROI data/processed_v3 (padrao PLANO: <split>/<source>/, sem subpasta de categoria)
a partir do seu proprio conteudo, consertando os problemas da auditoria:

  1. Remove o lixo `*:Zone.Identifier`.
  2. CONSOLIDA todas as imagens reais dos 3 splits e DEDUPLICA por conteudo (sha256).
  3. RECUPERA o rotulo (categoria) de cada real via data/splits/all.csv (nome -> ticket);
     DESCARTA os reais sem rotulo (decisao do usuario).
  4. NORMALIZA nomes (dupla extensao .png.png, sufixo " (1)").
  5. RE-SPLIT agrupado por TICKET (erros) e por SESSAO de captura (limpas) — ZERO vazamento —
     estratificado por categoria. Reusa siamese.manifest (nucleo validado do projeto).
  6. REGENERA os sinteticos de TREINO a partir das limpas que cairam no TRAIN (sem vazamento
     de tela-mae), na convencao {parent}__{tipo}__v{n}.png (siamese.synthetic.inject).
  7. Escreve data/processed_v3/labels.csv (rotulos, ja que as pastas sao planas).

Constroi em STAGING (data/processed_v3.rebuild/), AUTO-VERIFICA (0 vazamento, 0 dup, rotulo
completo) e so entao troca pela pasta final.

Uso:
    python scripts/rebuild_processed_v3.py --dry-run     # so o plano/contagens
    python scripts/rebuild_processed_v3.py --apply       # constroi + verifica + troca
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# garante import de siamese/ independentemente de instalacao editable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image  # noqa: E402

from siamese.manifest import (  # noqa: E402
    Sample, grouped_stratified_split, assign_clean_session_groups,
    _parse_meta, _group_key, category_id, CATEGORIES,
)
from siamese.synthetic import inject, SYNTH_TO_CATEGORY, MULTICLASS_SYNTH_TYPES  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")
RE_TICKET = re.compile(r"(IKSWW[-_]\d+)", re.IGNORECASE)

LABELS_COLS = ["path", "split", "source", "category", "category_id", "label",
               "group", "form_factor", "orientation", "kind", "is_competitor", "has_boundbox"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ticket_of(name: str) -> str | None:
    m = RE_TICKET.search(name)
    return m.group(1).upper().replace("_", "-") if m else None


def normalize_name(name: str) -> str:
    """Conserta dupla extensao e sufixo de copia ' (1)'. Preserva o resto (inclusive boundBox)."""
    n = re.sub(r"\.(png|jpe?g)(\.(png|jpe?g))+$", r".\1", name, flags=re.I)  # .png.png -> .png
    p = Path(n)
    stem = re.sub(r"\s*\(\d+\)\s*$", "", p.stem).strip()                    # "x (1)" -> "x"
    return stem + p.suffix.lower()


def clean_score(name: str) -> tuple:
    """Menor = melhor representante de um grupo de duplicatas exatas."""
    return (bool(re.search(r"\.(png|jpe?g)\.(png|jpe?g)$", name, re.I)),  # evita dupla ext
            bool(re.search(r"\(\d+\)", name)),                            # evita " (1)"
            len(name), name)


def load_labels(csv_path: Path):
    by_name, tick_cats = {}, defaultdict(Counter)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            nm = Path(r["path"]).name
            cat = (r.get("category") or "").strip()
            if cat:
                by_name[nm] = cat
                t = ticket_of(nm)
                if t:
                    tick_cats[t][cat] += 1
    by_ticket = {t: c.most_common(1)[0][0] for t, c in tick_cats.items()}
    return by_name, by_ticket


def collect_reals(root: Path):
    out = []
    for sp in SPLITS:
        base = root / sp / "real"
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and "Zone.Identifier" not in p.name and p.suffix.lower() in IMG_EXTS:
                out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/processed_v3"))
    ap.add_argument("--labels", type=Path, default=Path("data/splits/all.csv"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-variants", type=int, default=4)
    ap.add_argument("--synth-seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error("use --dry-run ou --apply")

    by_name, by_ticket = load_labels(args.labels)
    reals = collect_reals(args.root)
    print(f"Imagens reais brutas (3 splits): {len(reals)}")

    # --- dedup por conteudo (sha256), global ---
    by_sha = defaultdict(list)
    for p in reals:
        by_sha[sha256(p)].append(p)
    uniq = []
    n_dropped_dup = 0
    for h, paths in by_sha.items():
        rep = min(paths, key=lambda p: clean_score(p.name))
        uniq.append(rep)
        n_dropped_dup += len(paths) - 1
    print(f"Unicas por conteudo: {len(uniq)}  (duplicatas removidas: {n_dropped_dup})")

    # --- rotula + descarta sem rotulo ---
    samples, dropped_unlabeled = [], []
    for p in sorted(uniq, key=lambda x: x.name):
        nm = p.name
        cat = by_name.get(nm) or by_ticket.get(ticket_of(nm) or "")
        if not cat:
            dropped_unlabeled.append(p)
            continue
        is_clean = (cat == "clean")
        src = "no_erros" if is_clean else "errors_dataset"
        s = Sample(
            path=str(p.resolve()),
            label=0 if is_clean else 1,
            group=_group_key(p, src),
            source=src,
            category=cat,
            **_parse_meta(nm, src),
        )
        s._dest = normalize_name(nm)  # nome de destino (plano)
        samples.append(s)
    print(f"Rotuladas: {len(samples)}  | descartadas SEM rotulo: {len(dropped_unlabeled)} "
          f"({len({ticket_of(p.name) for p in dropped_unlabeled})} tickets)")

    # --- agrupa limpas por sessao (anti-vazamento) e faz o split agrupado/estratificado ---
    samples = assign_clean_session_groups(samples)
    samples = grouped_stratified_split(
        samples, val_frac=args.val_frac, test_frac=args.test_frac,
        seed=args.seed, stratify="category")

    # --- verificacao de vazamento (grupo nunca cruza split) ---
    g2s = defaultdict(set)
    for s in samples:
        g2s[s.group].add(s.split)
    leaks = {g: sorted(v) for g, v in g2s.items() if len(v) > 1}
    assert not leaks, f"VAZAMENTO de grupo apos split: {dict(list(leaks.items())[:5])}"

    # --- relatorio de distribuicao ---
    def grid(src_filter):
        g = defaultdict(lambda: defaultdict(int))
        for s in samples:
            if src_filter(s):
                g[s.category][s.split] += 1
        return g

    print("\n--- REAIS por categoria x split ---")
    g = grid(lambda s: True)
    print(f"  {'categoria':20s}" + "".join(f"{sp:>8s}" for sp in SPLITS) + f"{'total':>8s}")
    for c in [c for c in CATEGORIES if c in g]:
        row = [g[c][sp] for sp in SPLITS]
        miss = [sp for sp in ("val", "test") if g[c][sp] == 0 and c != "clean"]
        print(f"  {c:20s}" + "".join(f"{v:8d}" for v in row) + f"{sum(row):8d}"
              + ("   <- AUSENTE em " + ",".join(miss) if miss else ""))
    train_clean = [s for s in samples if s.split == "train" and s.category == "clean"]
    n_synth = len(train_clean) * args.n_variants
    print(f"\n  train/clean = {len(train_clean)}  -> sinteticos a gerar = {n_synth} "
          f"({args.n_variants}/limpa, tipos: {MULTICLASS_SYNTH_TYPES})")

    if args.dry_run:
        print("\n[dry-run] nada foi escrito.")
        return

    # ------------- MATERIALIZACAO em STAGING -------------
    staging = args.root.parent / (args.root.name + ".rebuild")
    if staging.exists():
        shutil.rmtree(staging)
    for sp in SPLITS:
        (staging / sp / "real").mkdir(parents=True, exist_ok=True)
    (staging / "train" / "synthetic").mkdir(parents=True, exist_ok=True)

    labels_rows = []

    def add_label(relpath, split, source, cat, label, group, meta):
        labels_rows.append({
            "path": relpath, "split": split, "source": source, "category": cat,
            "category_id": category_id(cat), "label": label, "group": group,
            "form_factor": meta.get("form_factor", ""), "orientation": meta.get("orientation", ""),
            "kind": meta.get("kind", ""), "is_competitor": meta.get("is_competitor", ""),
            "has_boundbox": meta.get("has_boundbox", ""),
        })

    # copia reais (bytes verbatim; nome normalizado, unico no destino)
    used = defaultdict(set)
    for s in samples:
        ddir = staging / s.split / "real"
        name = s._dest
        if name in used[s.split]:
            stem, suf = Path(name).stem, Path(name).suffix
            i = 1
            while f"{stem}_{i}{suf}" in used[s.split]:
                i += 1
            name = f"{stem}_{i}{suf}"
        used[s.split].add(name)
        shutil.copy2(s.path, ddir / name)
        s._final = name
        meta = {"form_factor": s.form_factor, "orientation": s.orientation, "kind": s.kind,
                "is_competitor": s.is_competitor, "has_boundbox": s.has_boundbox}
        add_label(f"{s.split}/real/{name}", s.split, "real", s.category, s.label, s.group, meta)

    # regenera sinteticos de treino a partir das limpas do TRAIN (convencao do projeto)
    rng = random.Random(args.synth_seed)
    n_made = 0
    for s in sorted(train_clean, key=lambda s: s._final):
        img = Image.open(s.path).convert("RGB")
        stem = Path(s._final).stem
        for v in range(args.n_variants):
            corr, types = inject(img, rng, n_errors=1, types=MULTICLASS_SYNTH_TYPES)
            tstr = "+".join(types)
            cat = SYNTH_TO_CATEGORY.get(types[0]) if types else None
            cat = cat or "overlay"
            name = f"{stem}__{tstr}__v{v}.png"
            corr.save(staging / "train" / "synthetic" / name)
            add_label(f"train/synthetic/{name}", "train", "synthetic", cat, 1,
                      f"synthparent:{stem}",
                      {"form_factor": s.form_factor, "orientation": s.orientation,
                       "kind": "synthetic", "is_competitor": False, "has_boundbox": False})
            n_made += 1
    print(f"\nSinteticos gerados: {n_made}")

    # labels.csv
    with (staging / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABELS_COLS)
        w.writeheader()
        w.writerows(labels_rows)

    # ------------- AUTO-VERIFICACAO do staging -------------
    print("\n--- auto-verificacao (staging) ---")
    staged = [p for p in staging.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    zone = [p for p in staging.rglob("*") if "Zone.Identifier" in p.name]
    shas = defaultdict(list)
    for p in staged:
        shas[sha256(p)].append(p)
    cross = 0
    for h, ps in shas.items():
        if len({pp.relative_to(staging).parts[0] for pp in ps}) > 1:
            cross += 1
    # vazamento por grupo no labels.csv
    gg = defaultdict(set)
    for r in labels_rows:
        if r["source"] == "real":
            gg[r["group"]].add(r["split"])
    leak2 = {g: sorted(v) for g, v in gg.items() if len(v) > 1}
    ok = (not zone) and (cross == 0) and (not leak2)
    print(f"  imagens: {len(staged)} | zone.identifier: {len(zone)} | "
          f"dup exatas cruzando splits: {cross} | grupos cruzando splits: {len(leak2)}")
    if not ok:
        print("  FALHA na verificacao — NAO vou trocar. staging em:", staging)
        sys.exit(1)
    print("  OK: 0 lixo, 0 duplicata cruzando splits, 0 vazamento de grupo.")

    # ------------- SWAP -------------
    backup = args.root.parent / (args.root.name + ".old")
    if backup.exists():
        shutil.rmtree(backup)
    args.root.rename(backup)
    staging.rename(args.root)
    shutil.rmtree(backup)
    print(f"\nOK: {args.root} reconstruido. labels.csv: {args.root}/labels.csv")
    print("  (backup temporario removido apos troca bem-sucedida)")


if __name__ == "__main__":
    main()
