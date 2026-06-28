#!/usr/bin/env python
"""Fase 3: compara backbones DINOv2 CONGELADOS (ViT-S vs base/large + registers) no dataset
de-confoundado processed_v3_plus. SELECIONA na VAL (val_synth_gate, livre de confound) — NAO
toca o teste (train_head so le train/val). Re-extrai embeddings por backbone (cache em
artifacts/bb_<tag>/) + sondas, treina a cabeca, reporta val_synth_gate + val_cat_f1.

Backbone congelado e' o TETO de separabilidade das features -> trocar e' barato (so re-extrai).
Os reg4 (registers) costumam melhorar os TOKENS DE PATCH (usados em mean/std). out_dim escala:
S 1152, base 2304, large 3072 -> a cabeca adapta (in_dim = X.shape[1]).

Uso: python scripts/compare_backbones.py --config configs/default_plus.yaml --processed data/processed_v3_plus
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from siamese.config import Config
from siamese.backbone import DinoV2Backbone, BackboneConfig
from siamese.features import extract_processed, clean_rows
from siamese.synth_features import extract_synthetic, extract_reflow_clean
from siamese.train import train_head

BACKBONES = {
    "S":      "vit_small_patch14_dinov2.lvd142m",
    "B_reg4": "vit_base_patch14_reg4_dinov2",
    "L_reg4": "vit_large_patch14_reg4_dinov2",
}


def prep(cfg: Config, processed: Path, emb: Path) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = DinoV2Backbone(BackboneConfig(
        model_name=cfg.backbone.model_name, size=cfg.backbone.size,
        use_patch_stats=cfg.backbone.use_patch_stats, preprocess=cfg.backbone.preprocess, device=dev))
    if not (emb / "train.npz").exists():
        extract_processed(processed, emb, bb, batch_size=cfg.backbone.batch_size)
    for split, seed in [("val", cfg.synthetic.seed + 100), ("test", cfg.synthetic.seed + 200)]:
        if not (emb / f"{split}_synth.npz").exists():
            rows = clean_rows(processed, split)
            if rows:
                extract_synthetic(None, emb / f"{split}_synth.npz", bb,
                                  n_variants=cfg.synthetic.n_variants,
                                  max_errors_per_image=cfg.synthetic.max_errors_per_image,
                                  seed=seed, batch_size=cfg.backbone.batch_size,
                                  multiclass=True, clean_rows=rows)
    if cfg.synthetic.reflow_clean:
        for split, seed in [("train", cfg.synthetic.seed + 300), ("val", cfg.synthetic.seed + 400),
                            ("test", cfg.synthetic.seed + 500)]:
            if not (emb / f"{split}_reflow.npz").exists():
                rows = clean_rows(processed, split)
                if rows:
                    extract_reflow_clean(emb / f"{split}_reflow.npz", bb, rows,
                                         n_variants=cfg.synthetic.n_reflow_variants,
                                         reflow_ops=cfg.synthetic.reflow_ops,
                                         max_reflow_ops=cfg.synthetic.max_reflow_ops,
                                         benign=cfg.synthetic.benign_augment,
                                         seed=seed, batch_size=cfg.backbone.batch_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default_plus.yaml"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed_v3_plus"))
    ap.add_argument("--backbones", nargs="*", default=list(BACKBONES))
    args = ap.parse_args()
    base = Config.load(args.config)

    rows = []
    for tag in args.backbones:
        model = BACKBONES[tag]
        cfg = copy.deepcopy(base)
        cfg.backbone.model_name = model
        emb = Path(f"artifacts/bb_{tag}")
        cfg.paths.emb_dir = str(emb)
        cfg.paths.models_dir = str(emb / "models")
        print(f"\n########## backbone {tag} ({model}) ##########", flush=True)
        prep(cfg, args.processed, emb)
        res = train_head(cfg)
        m = res["best_metrics"]
        rows.append((tag, model, float(res["best_sel"]),
                     float(m.get("val_cat_f1", float("nan"))), float(m.get("val_ap", float("nan")))))

    print("\n================ COMPARACAO (VAL, livre de confound — selecao honesta) ================")
    print(f"{'backbone':8s} {'val_synth_gate':>15s} {'val_cat_f1':>11s} {'val_ap':>8s}   modelo")
    for tag, model, sel, catf1, ap in sorted(rows, key=lambda r: -r[2]):
        print(f"{tag:8s} {sel:15.3f} {catf1:11.3f} {ap:8.3f}   {model}")
    if rows:
        win = max(rows, key=lambda r: r[2])
        print(f"\nVencedor (maior val_synth_gate): {win[0]} ({win[1]}). "
              f"Rodar o final-test SO nele (run_experiment com este backbone).")


if __name__ == "__main__":
    main()
