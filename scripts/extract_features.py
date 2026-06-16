#!/usr/bin/env python
"""Extrai e cacheia embeddings DINOv2 para cada split.

Uso:
    python scripts/extract_features.py --splits data/splits --out artifacts/embeddings \
        [--use-patch-stats] [--size 518] [--batch-size 16]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from siamese.backbone import DinoV2Backbone, BackboneConfig
from siamese.features import extract_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("data/splits"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/embeddings"))
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--use-patch-stats", action="store_true",
                    help="concatena media+std dos tokens de patch (saida 3*384=1152)")
    ap.add_argument("--preprocess", choices=["resize", "pad"], default="resize")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = BackboneConfig(size=args.size, use_patch_stats=args.use_patch_stats,
                         preprocess=args.preprocess, device=args.device)
    print(f"Carregando backbone DINOv2 ({cfg.model_name}) em {args.device}...")
    backbone = DinoV2Backbone(cfg)
    print(f"out_dim = {backbone.out_dim}")

    for split in ("train", "val", "test"):
        csv_path = args.splits / f"{split}.csv"
        if not csv_path.exists():
            print(f"  (pulando {split}: {csv_path} nao existe)")
            continue
        t0 = time.time()
        info = extract_split(
            csv_path, args.out / f"{split}.npz", backbone,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        print(f"  {split}: {info['n']} imagens -> {info['out']} ({time.time()-t0:.1f}s)")

    print("Embeddings cacheados. Treino da cabeca siamesa agora e quase instantaneo.")


if __name__ == "__main__":
    main()
