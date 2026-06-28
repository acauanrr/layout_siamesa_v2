#!/usr/bin/env python
"""Extrai e cacheia embeddings DINOv2 a partir do DATASET CANONICO em data/processed_v3/.

FONTE DA VERDADE = data/processed_v3/ (NAO data/input/). O que estiver materializado em
processed/<split>/<fonte>/<categoria>/ e' exatamente o que o modelo usa — incluindo
correcoes/ajustes manuais. Varre a arvore, deriva split/fonte/categoria/label do caminho e
os metadados do nome do arquivo, e cacheia os embeddings por (split, fonte).

Saida: artifacts/embeddings/{train,val,test}.npz (reais) + train_synth.npz (sinteticos de
treino, materializados em processed/train/synthetic/). val/test_synth (sonda livre de
confound) sao gerados por make_synthetic.py a partir de processed/{val,test}/real/clean.

Uso:
    python scripts/extract_features.py --processed data/processed_v3 --out artifacts/embeddings \
        [--use-patch-stats] [--preprocess pad] [--size 518] [--batch-size 16]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from siamese.backbone import DinoV2Backbone, BackboneConfig
from siamese.features import extract_processed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed_v3"),
                    help="dataset canonico (fonte da verdade): plano com labels.csv "
                         "(processed_v3) ou arvore legada categorizada")
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
    print(f"Backbone DINOv2 ({cfg.model_name}) em {args.device} | FONTE: {args.processed}/")
    backbone = DinoV2Backbone(cfg)
    print(f"out_dim = {backbone.out_dim}")

    t0 = time.time()
    summary = extract_processed(args.processed, args.out, backbone,
                                batch_size=args.batch_size, num_workers=args.num_workers)
    for k, n in summary.items():
        print(f"  {k:18s} {n:4d} imagens")
    print(f"Embeddings cacheados a partir de {args.processed}/ em {time.time()-t0:.1f}s "
          f"(fonte da verdade — honra correcoes manuais).")


if __name__ == "__main__":
    main()
