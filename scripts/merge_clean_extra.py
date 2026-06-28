#!/usr/bin/env python
"""Fase 2: funde as telas LIMPAS baixadas (data/clean_extra/, via fetch_clean_extra.py) ao
dataset de producao, gerando um NOVO dataset NAO-DESTRUTIVO data/processed_v3_plus/ (o
processed_v3 original fica intacto, p/ comparacao baseline-vs-plus).

O que faz:
  - copia TODOS os reais do processed_v3 (mantem os splits existentes -> comparabilidade);
  - particiona as limpas novas em train/val/test (grupo atomico = 1 imagem; fracs configuraveis).
    train -> expande o manifold limpo (generalizacao); val/test -> limpas em OUTRAS resolucoes
    (de-confound: o held-out passa a ter limpo E erro compartilhando faixa de resolucao);
  - REGENERA train/synthetic INTEIRO a partir de TODAS as limpas de treino (orig + novas), p/ os
    erros sinteticos herdarem as novas resolucoes (anti-confound pelo lado do erro tambem);
  - escreve processed_v3_plus/labels.csv e VERIFICA 0 vazamento de grupo entre splits.

Depois (com o dataset plus):
  python scripts/run_experiment.py --processed data/processed_v3_plus   # re-extrai, sonda, avalia
  (use um emb_dir/reports_dir separados — ex.: configs/default_plus.yaml — p/ nao misturar.)

Uso:
    python scripts/merge_clean_extra.py --dry-run
    python scripts/merge_clean_extra.py --apply
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image  # noqa: E402

from siamese.manifest import category_id, CATEGORIES  # noqa: E402
from siamese.synthetic import inject, SYNTH_TO_CATEGORY, MULTICLASS_SYNTH_TYPES  # noqa: E402

SPLITS = ("train", "val", "test")
LABELS_COLS = ["path", "split", "source", "category", "category_id", "label",
               "group", "form_factor", "orientation", "kind", "is_competitor", "has_boundbox"]


def _read(p: Path):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/processed_v3"))
    ap.add_argument("--extra", type=Path, default=Path("data/clean_extra"))
    ap.add_argument("--dest", type=Path, default=Path("data/processed_v3_plus"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.24)
    ap.add_argument("--n-variants", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synth-seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error("use --dry-run ou --apply")

    src_rows = _read(args.src / "labels.csv")
    real_rows = [r for r in src_rows if r["source"] == "real"]
    extra = _read(args.extra / "labels_extra.csv")
    print(f"reais (processed_v3): {len(real_rows)} | limpas novas (pool): {len(extra)}")

    # --- split agrupado das limpas novas (cada imagem = 1 grupo) ---
    rng = random.Random(args.seed)
    rng.shuffle(extra)
    tr_frac = max(0.0, 1.0 - args.val_frac - args.test_frac)
    targets = {"train": tr_frac, "val": args.val_frac, "test": args.test_frac}
    filled = {s: 0 for s in SPLITS}
    n = max(1, len(extra))
    for r in extra:
        s = max(SPLITS, key=lambda k: targets[k] * n - filled[k])
        r["_split"] = s
        filled[s] += 1
    print("limpas novas por split:", filled)

    # --- distribuicao alvo (reais por categoria x split) p/ o relatorio ---
    grid = defaultdict(lambda: defaultdict(int))
    for r in real_rows:
        grid[r["category"]][r["split"]] += 1
    extra_tr = sum(1 for r in extra if r["_split"] == "train")
    n_clean_train = grid["clean"]["train"] + extra_tr
    print(f"\nlimpas de TREINO: {grid['clean']['train']} (orig) + {extra_tr} (novas) = {n_clean_train}")
    print(f"sinteticos a gerar: {n_clean_train * args.n_variants} ({args.n_variants}/limpa)")
    print("clean por split (orig -> +novas):")
    for s in SPLITS:
        print(f"  {s:5s}: {grid['clean'][s]:4d} -> {grid['clean'][s] + filled[s]}")

    if args.dry_run:
        print("\n[dry-run] nada foi escrito.")
        return

    # ------------- materializacao (NAO-DESTRUTIVA: dest novo) -------------
    if args.dest.exists():
        shutil.rmtree(args.dest)
    for s in SPLITS:
        (args.dest / s / "real").mkdir(parents=True, exist_ok=True)
    (args.dest / "train" / "synthetic").mkdir(parents=True, exist_ok=True)

    out_rows: list[dict] = []

    def add(relpath, split, source, cat, label, group, meta):
        out_rows.append({
            "path": relpath, "split": split, "source": source, "category": cat,
            "category_id": category_id(cat), "label": label, "group": group,
            "form_factor": meta.get("form_factor", ""), "orientation": meta.get("orientation", ""),
            "kind": meta.get("kind", ""), "is_competitor": meta.get("is_competitor", ""),
            "has_boundbox": meta.get("has_boundbox", ""),
        })

    # 1) copia reais existentes (bytes verbatim; mesmos nomes/splits)
    for r in real_rows:
        dst = args.dest / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.src / r["path"], dst)
        add(r["path"], r["split"], "real", r["category"], int(r["label"]), r["group"],
            {k: r.get(k, "") for k in ("form_factor", "orientation", "kind", "is_competitor", "has_boundbox")})

    # 2) copia limpas novas -> dest/<split>/real/extra_<fn>
    train_clean_files = [(args.dest / r["path"]) for r in real_rows
                         if r["split"] == "train" and r["category"] == "clean"]
    for r in extra:
        sp = r["_split"]
        w, h = int(r["w"]), int(r["h"])
        name = "extra_" + Path(r["path"]).name
        src_file = args.extra.parent / r["path"]      # data/ + clean_extra/<source>/<fn>
        dst = args.dest / sp / "real" / name
        shutil.copy2(src_file, dst)
        meta = {"form_factor": "external", "orientation": "portrait" if h >= w else "landscape",
                "kind": "screenshot", "is_competitor": "", "has_boundbox": ""}
        add(f"{sp}/real/{name}", sp, "real", "clean", 0, r["group"], meta)
        if sp == "train":
            train_clean_files.append(dst)

    # 3) regenera train/synthetic a partir de TODAS as limpas de treino (orig + novas)
    srng = random.Random(args.synth_seed)
    n_made = 0
    for f in sorted(train_clean_files):
        img = Image.open(f).convert("RGB")
        stem = f.stem
        for v in range(args.n_variants):
            corr, types = inject(img, srng, n_errors=1, types=MULTICLASS_SYNTH_TYPES)
            cat = (SYNTH_TO_CATEGORY.get(types[0]) if types else None) or "overlay"
            name = f"{stem}__{'+'.join(types)}__v{v}.png"
            corr.save(args.dest / "train" / "synthetic" / name)
            add(f"train/synthetic/{name}", "train", "synthetic", cat, 1, f"synthparent:{stem}",
                {"form_factor": "", "orientation": "", "kind": "synthetic"})
            n_made += 1
    print(f"\nsinteticos gerados: {n_made}")

    # 4) labels.csv
    with (args.dest / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABELS_COLS)
        w.writeheader()
        w.writerows(out_rows)

    # 5) verificacao de vazamento de grupo (real)
    g2s = defaultdict(set)
    for r in out_rows:
        if r["source"] == "real":
            g2s[r["group"]].add(r["split"])
    leaks = {g: sorted(v) for g, v in g2s.items() if len(v) > 1}
    assert not leaks, f"VAZAMENTO de grupo: {dict(list(leaks.items())[:5])}"
    print(f"OK: {args.dest} criado | {len(out_rows)} linhas | 0 vazamento de grupo")
    print("clean total por split:", {s: sum(1 for r in out_rows
          if r["split"] == s and r["category"] == "clean" and r["source"] == "real") for s in SPLITS})


if __name__ == "__main__":
    main()
