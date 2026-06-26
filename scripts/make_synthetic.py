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
from siamese.features import clean_rows
from siamese.synth_features import extract_synthetic, extract_reflow_clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed_v3"))
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

    emb_dir = Path(cfg.paths.emb_dir)

    # sonda livre de confound (ERROS): erros injetados nas limpas held-out de val/test (mesma
    # resolucao/device). Seeds distintos por split p/ reprodutibilidade.
    for split, seed in [("val", cfg.synthetic.seed + 100), ("test", cfg.synthetic.seed + 200)]:
        rows = clean_rows(args.processed, split)
        if not rows:
            print(f"  (pulando {split}: sem limpas em {args.processed}/{split}/real/clean)")
            continue
        info = extract_synthetic(
            None, emb_dir / f"{split}_synth.npz", backbone,
            n_variants=cfg.synthetic.n_variants,
            max_errors_per_image=cfg.synthetic.max_errors_per_image,
            seed=seed, batch_size=cfg.backbone.batch_size,
            multiclass=cfg.train.multiclass, clean_rows=rows,
        )
        print(f"  {split}: sonda sintetica {info['n']} -> {info['out']} (de {args.processed} {split}/clean)")

    # REFLOW-CLEAN (variantes LIMPAS de layout legitimo). train_reflow -> entra no TREINO como
    # negativos; val_reflow/test_reflow -> sondas de FALSO-POSITIVO (o gate NAO deve acender em
    # reflow). Anti-confound pelo lado limpo (ar_relayout tira a limpa de 2076x2152). Ver reflow.py.
    if cfg.synthetic.reflow_clean:
        for split, seed in [("train", cfg.synthetic.seed + 300),
                            ("val", cfg.synthetic.seed + 400),
                            ("test", cfg.synthetic.seed + 500)]:
            rows = clean_rows(args.processed, split)
            if not rows:
                print(f"  (pulando reflow {split}: sem limpas em {args.processed}/{split}/real/clean)")
                continue
            info = extract_reflow_clean(
                emb_dir / f"{split}_reflow.npz", backbone, rows,
                n_variants=cfg.synthetic.n_reflow_variants,
                reflow_ops=cfg.synthetic.reflow_ops,
                max_reflow_ops=cfg.synthetic.max_reflow_ops,
                benign=cfg.synthetic.benign_augment,
                seed=seed, batch_size=cfg.backbone.batch_size,
            )
            print(f"  {split}: reflow-clean {info['n']} -> {info['out']} (de {args.processed} {split}/clean)")
    else:
        print("  (reflow_clean=false -> nao gera *_reflow.npz)")


if __name__ == "__main__":
    main()
