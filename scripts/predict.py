#!/usr/bin/env python
"""Inferencia em imagens novas.

Uso:
    python scripts/predict.py --models artifacts/models img1.png img2.png ...
    python scripts/predict.py --models artifacts/models --dir data/input/with_errors
"""
from __future__ import annotations

import argparse
from pathlib import Path

from siamese.infer import Predictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=str)
    ap.add_argument("--models", type=Path, default=Path("artifacts/models"))
    ap.add_argument("--dir", type=Path, default=None, help="processa todas as imagens da pasta")
    args = ap.parse_args()

    paths = list(args.images)
    if args.dir:
        paths += [str(p) for p in sorted(args.dir.iterdir())
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not paths:
        ap.error("forneca imagens ou --dir")

    pred = Predictor(args.models)
    print(f"limiar (precisao-alvo {pred.target_precision:.2f}) = {pred.threshold:.3f}")
    if pred.multiclass:
        print(f"modo multi-cluster | categorias: {', '.join(pred.categories)}\n")
        print(f"{'arquivo':48s} {'p(erro)':>8s}  {'decisao':8s}  categoria")
    else:
        print()
        print(f"{'arquivo':52s} {'p(erro)':>8s}  decisao")
    results = [pred.predict(p) for p in paths]
    for r in sorted(results, key=lambda d: -d["p_erro"]):
        if pred.multiclass:
            cat = r["categoria"] or "-"
            print(f"{r['file'][:48]:48s} {r['p_erro']:8.3f}  {r['decisao']:8s}  {cat}")
        else:
            print(f"{r['file'][:52]:52s} {r['p_erro']:8.3f}  {r['decisao']}")


if __name__ == "__main__":
    main()
