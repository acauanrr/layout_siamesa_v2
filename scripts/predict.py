#!/usr/bin/env python
"""Inferencia em imagens novas (roteamento por dominio: near-square -> gate foldable).

Uso:
    python scripts/predict.py --models artifacts/bb_L_reg4/models img1.png img2.png ...
    python scripts/predict.py --models artifacts/bb_L_reg4/models --dir data/input/with_errors
"""
from __future__ import annotations

import argparse
from pathlib import Path

from siamese.infer import Predictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=str)
    ap.add_argument("--models", type=Path, default=Path("artifacts/bb_L_reg4/models"))
    ap.add_argument("--dir", type=Path, default=None, help="processa todas as imagens da pasta")
    ap.add_argument("--no-route-foldable", action="store_true",
                    help="desliga o roteamento por dominio (usa o gate global em tudo)")
    args = ap.parse_args()

    paths = list(args.images)
    if args.dir:
        paths += [str(p) for p in sorted(args.dir.iterdir())
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    if not paths:
        ap.error("forneca imagens ou --dir")

    pred = Predictor(args.models, route_foldable=not args.no_route_foldable)
    fold_thr = getattr(pred, "foldable_threshold", float("nan"))
    print(f"limiar global {pred.threshold:.3f} | limiar foldable {fold_thr:.3f} "
          f"| roteamento={'ON' if pred.route_foldable else 'OFF'} (near-square -> gate protótipo)")
    if pred.multiclass:
        print(f"modo multi-cluster | categorias: {', '.join(pred.categories)}\n")
        print(f"{'arquivo':44s} {'p(erro)':>8s}  {'decisao':8s} {'gate':8s}  categoria")
    else:
        print()
        print(f"{'arquivo':48s} {'p(erro)':>8s}  {'decisao':8s} {'gate':8s}")
    results = [pred.predict(p) for p in paths]
    for r in sorted(results, key=lambda d: -d["p_erro"]):
        g = "foldable" if r["near_square"] else "global"
        if pred.multiclass:
            print(f"{r['file'][:44]:44s} {r['p_erro']:8.3f}  {r['decisao']:8s} {g:8s}  {r['categoria'] or '-'}")
        else:
            print(f"{r['file'][:48]:48s} {r['p_erro']:8.3f}  {r['decisao']:8s} {g:8s}")


if __name__ == "__main__":
    main()
