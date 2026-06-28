#!/usr/bin/env python
"""Gera mapas de calor de erro (ONDE esta o erro) para imagens.

Padrao: localizador SUPERVISIONADO por erros sinteticos (aprende a assinatura de faixa
preta / vazio / overlay / crop). Alternativa: --patchcore (novidade vs telas limpas).

Uso:
    python scripts/localize.py --config configs/default.yaml --dir data/input/with_errors --n 12
    python scripts/localize.py --config configs/default.yaml img1.png img2.png
    python scripts/localize.py --config configs/default.yaml --patchcore img1.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from siamese.config import Config
from siamese.backbone import DinoV2Backbone, BackboneConfig
from siamese.localize import SyntheticPatchLocalizer, PatchCoreLocalizer
from siamese.region_detector import GeometricDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=str)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--geometric", action="store_true",
                    help="detector geometrico de black-region/empty-space (localiza barras "
                         "pretas; NAO classifica: AUROC~0.5 pois letterbox de video tambem "
                         "tem barras pretas). Ferramenta de EVIDENCIA, nao de decisao.")
    ap.add_argument("--supervised", action="store_true",
                    help="localizador por-patch supervisionado (EXPERIMENTAL: ruidoso, "
                         "overlay/disorder confundem o classificador)")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.geometric:
        det = GeometricDetector()
        out_dir = Path(cfg.paths.reports_dir) / "heatmaps"
        paths = list(args.images)
        if args.dir:
            paths += [str(p) for p in sorted(args.dir.iterdir())
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg"}][:args.n]
        print(f"{'arquivo':52s} {'score':>7s}  regioes")
        for p in paths:
            res = det.save_overlay(p, out_dir / (Path(p).stem + "_geo.png"))
            regs = "; ".join(f"{r.tipo}[{r.bordas or 'bloco'},{int(r.area_frac*100)}%]" for r in res.regions) or "-"
            print(f"{Path(p).name[:52]:52s} {res.score:7.3f}  {regs}")
        print(f"\nMapas em: {out_dir}/  (NB: localiza, nao classifica — AUROC~0.5)")
        return

    bb = DinoV2Backbone(BackboneConfig(size=cfg.backbone.size, use_patch_stats=cfg.backbone.use_patch_stats,
                                       preprocess=cfg.backbone.preprocess, device=device))
    models_dir = Path(cfg.paths.models_dir)
    train_csv = Path(cfg.paths.splits_dir) / "train.csv"

    if args.supervised:
        clf_path = models_dir / "patch_localizer.npz"
        if args.retrain or not clf_path.exists():
            print("Treinando localizador por-patch (supervisionado por sinteticos)...")
            loc = SyntheticPatchLocalizer.train(bb, train_csv,
                                                n_variants=cfg.synthetic.n_variants,
                                                max_errors=cfg.synthetic.max_errors_per_image)
            loc.save(models_dir)
        else:
            loc = SyntheticPatchLocalizer.load(bb, models_dir)
    else:
        print("Construindo memory bank (PatchCore, novidade vs telas limpas)...")
        bank = PatchCoreLocalizer.build_bank(bb, train_csv)
        loc = PatchCoreLocalizer(bb, bank)

    paths = list(args.images)
    if args.dir:
        paths += [str(p) for p in sorted(args.dir.iterdir())
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"}][:args.n]

    out_dir = Path(cfg.paths.reports_dir) / "heatmaps"
    print(f"\n{'arquivo':52s} {'score':>8s}")
    for p in paths:
        out = out_dir / (Path(p).stem + "_heatmap.png")
        score = loc.save_overlay(p, out)
        print(f"{Path(p).name[:52]:52s} {score:8.3f}")
    print(f"\nMapas salvos em: {out_dir}/")


if __name__ == "__main__":
    main()
