#!/usr/bin/env python
"""Treina a cabeca siamesa sobre embeddings cacheados.

Uso:
    python scripts/train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from siamese.config import Config
from siamese.train import train_head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = ap.parse_args()
    cfg = Config.load(args.config)
    train_head(cfg)


if __name__ == "__main__":
    main()
