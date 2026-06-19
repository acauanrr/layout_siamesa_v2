#!/usr/bin/env python
"""Gera a SONDA SINTETICA LIVRE DE CONFOUND (val/test) a partir das telas LIMPAS de
data/processed/ — usada pela metrica honesta de deteccao em evaluate.py.

FONTE DA VERDADE = data/processed/. As limpas vem de processed/{val,test}/real/clean/ (as
mesmas que entram em val.npz/test.npz). Injeta erros nelas (mesma resolucao/device) -> mede
deteccao de CONTEUDO de erro sem o confound de resolucao.

NB: o sintetico de TREINO (train_synth.npz) NAO e gerado aqui — ele e' materializado em
processed/train/synthetic/ por export_processed.py e embedado por extract_features.py
(fonte da verdade unica). Este script cobre apenas val/test (sonda de avaliacao).

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

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _clean_rows(processed: Path, split: str) -> list[dict]:
    """Limpas reais de processed/<split>/real/clean/ (mesmas que entram em <split>.npz)."""
    d = processed / split / "real" / "clean"
    if not d.is_dir():
        return []
    return [{"path": str(p.resolve())} for p in sorted(d.iterdir()) if p.suffix.lower() in _EXTS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
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

    # sonda livre de confound: erros injetados nas limpas held-out de val/test (mesma
    # resolucao/device). Seeds distintos por split p/ reprodutibilidade.
    for split, seed in [("val", cfg.synthetic.seed + 100), ("test", cfg.synthetic.seed + 200)]:
        rows = _clean_rows(args.processed, split)
        if not rows:
            print(f"  (pulando {split}: sem limpas em {args.processed}/{split}/real/clean)")
            continue
        info = extract_synthetic(
            None, Path(cfg.paths.emb_dir) / f"{split}_synth.npz", backbone,
            n_variants=cfg.synthetic.n_variants,
            max_errors_per_image=cfg.synthetic.max_errors_per_image,
            seed=seed, batch_size=cfg.backbone.batch_size,
            multiclass=cfg.train.multiclass, clean_rows=rows,
        )
        print(f"  {split}: sonda sintetica {info['n']} -> {info['out']} (de processed/{split}/real/clean)")


if __name__ == "__main__":
    main()
