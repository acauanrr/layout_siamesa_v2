#!/usr/bin/env python
"""Constroi os manifestos de split (train/val/test) e imprime uma AUDITORIA DE CONFOUNDS.

Uso:
    python scripts/build_splits.py --input data/input --out data/splits --seed 42

A auditoria mostra explicitamente o quanto a label esta correlacionada com confounds
(resolucao, foto vs screenshot, form factor), para que a interpretacao das metricas
do modelo seja honesta.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from siamese.manifest import (
    scan_dataset,
    grouped_stratified_split,
    write_manifests,
)


def _resolution(path: str) -> tuple[int, int]:
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return (-1, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/input"))
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--source", choices=["errors_dataset", "with_errors"],
                    default="errors_dataset",
                    help="fonte dos erros: errors_dataset (categorizado, padrao) | with_errors (legado binario)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.24)   # teste >= 40 limpas
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = scan_dataset(args.input, source=args.source)
    print(f"Fonte de erros: {args.source}")
    samples = grouped_stratified_split(
        samples, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    counts = write_manifests(samples, args.out)

    n = len(samples)
    n_err = sum(s.label == 1 for s in samples)
    print(f"\n=== DATASET: {n} imagens | erro={n_err} ({n_err/n:.0%}) | sem-erro={n-n_err} ===")
    print("Grupos (tickets) unicos:", len({s.group for s in samples}))

    # Distribuicao por split x classe
    print("\n--- Split x classe (imagens) ---")
    grid = defaultdict(lambda: [0, 0])
    for s in samples:
        grid[s.split][s.label] += 1
    for split in ("train", "val", "test"):
        a, b = grid[split]
        print(f"  {split:5s}: total={counts[split]:3d}  sem-erro={a:3d}  erro={b:3d}")

    # Distribuicao por split x CATEGORIA (multi-cluster) -- alerta p/ classes raras
    cats = sorted({s.category for s in samples})
    if len(cats) > 2:  # so faz sentido no modo categorizado
        print("\n--- Split x categoria (imagens) ---")
        cgrid = defaultdict(lambda: defaultdict(int))
        for s in samples:
            cgrid[s.category][s.split] += 1
        header = "  " + f"{'categoria':18s}" + "".join(f"{sp:>7s}" for sp in ("train", "val", "test"))
        print(header)
        for c in cats:
            row = "  " + f"{c:18s}" + "".join(f"{cgrid[c][sp]:7d}" for sp in ("train", "val", "test"))
            faltam = [sp for sp in ("val", "test") if cgrid[c][sp] == 0 and c != "clean"]
            print(row + ("   <- AUSENTE em " + ",".join(faltam) if faltam else ""))

    # Verificacao de vazamento: nenhum grupo em mais de um split
    g2splits = defaultdict(set)
    for s in samples:
        g2splits[s.group].add(s.split)
    leaks = {g: sp for g, sp in g2splits.items() if len(sp) > 1}
    print(f"\n--- Vazamento de grupo entre splits: {len(leaks)} (esperado 0) ---")
    if leaks:
        for g, sp in list(leaks.items())[:10]:
            print("  VAZAMENTO:", g, sp)

    # AUDITORIA DE CONFOUNDS
    print("\n=== AUDITORIA DE CONFOUNDS (correlacao label x atributo) ===")

    def confound(attr_fn, title):
        print(f"\n--- {title} ---")
        tbl = defaultdict(lambda: [0, 0])
        for s in samples:
            tbl[attr_fn(s)][s.label] += 1
        print(f"  {'valor':18s} sem-erro  erro")
        for k in sorted(tbl, key=str):
            a, b = tbl[k]
            print(f"  {str(k):18s} {a:7d} {b:6d}")

    confound(lambda s: s.kind, "Foto vs Screenshot")
    confound(lambda s: s.form_factor, "Form factor")
    confound(lambda s: s.orientation, "Orientacao")
    confound(lambda s: s.category, "Categoria de erro")

    # Resolucao (amostrando todas; pode demorar ~1s)
    print("\n--- Resolucao (WxH) por classe ---")
    res_by_label = defaultdict(Counter)
    ar_by_label = defaultdict(list)
    for s in samples:
        w, h = _resolution(s.path)
        res_by_label[s.label][(w, h)] += 1
        if h > 0:
            ar_by_label[s.label].append(round(w / h, 2))
    for label, name in [(0, "sem-erro"), (1, "erro")]:
        res = res_by_label[label]
        top = res.most_common(5)
        ars = sorted(set(ar_by_label[label]))
        print(f"  {name}: {len(res)} resolucoes distintas | top: {top}")
        print(f"         aspect ratios distintos: {len(ars)} (ex: {ars[:8]})")

    print(f"\nManifestos escritos em: {args.out}/")
    print("ALERTA: se um atributo separa quase perfeitamente as classes, o modelo pode")
    print("aprende-lo em vez do erro real. As mitigacoes estao no pre-processamento e na")
    print("avaliacao controlada (ver docs/DESIGN.md).")


if __name__ == "__main__":
    main()
