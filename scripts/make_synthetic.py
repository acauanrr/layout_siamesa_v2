#!/usr/bin/env python
"""Gera e cacheia embeddings de erros SINTETICOS a partir das imagens limpas de treino.

Uso:
    python scripts/make_synthetic.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from siamese.config import Config
from siamese.backbone import DinoV2Backbone, BackboneConfig
from siamese.synth_features import extract_synthetic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = ap.parse_args()
    cfg = Config.load(args.config)

    if not cfg.synthetic.enabled:
        print("synthetic.enabled=false -> nada a fazer.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bcfg = BackboneConfig(model_name=cfg.backbone.model_name, size=cfg.backbone.size,
                          use_patch_stats=cfg.backbone.use_patch_stats,
                          preprocess=cfg.backbone.preprocess, device=device)
    backbone = DinoV2Backbone(bcfg)

    # train_synth -> treino anti-confound;  test_synth/val_synth -> avaliacao LIVRE de confound
    # (erros injetados nas proprias imagens limpas held-out, mesma resolucao/device).
    for split, seed in [("train", cfg.synthetic.seed),
                        ("val", cfg.synthetic.seed + 100),
                        ("test", cfg.synthetic.seed + 200)]:
        csv = Path(cfg.paths.splits_dir) / f"{split}.csv"
        if not csv.exists():
            continue
        info = extract_synthetic(
            csv,
            Path(cfg.paths.emb_dir) / f"{split}_synth.npz",
            backbone,
            n_variants=cfg.synthetic.n_variants,
            max_errors_per_image=cfg.synthetic.max_errors_per_image,
            seed=seed,
            batch_size=cfg.backbone.batch_size,
        )
        print(f"  {split}: erros sinteticos {info['n']} -> {info['out']}")


if __name__ == "__main__":
    main()
