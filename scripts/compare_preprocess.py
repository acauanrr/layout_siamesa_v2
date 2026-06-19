#!/usr/bin/env python
"""Compara empiricamente RESIZE (anamorfico) vs PAD (padding cinza) no pipeline completo.

Responde a pergunta: o padding cinza (preservando aspecto) ajuda a detectar o erro de
CONTEUDO sem reintroduzir confound? Mede, para cada modo:
  - synt_AUROC/AP : deteccao livre de confound (erros sinteticos em imagens limpas held-out)
  - pad_frac base : AUROC de prever erro SO pela fracao de cinza (confound reintroduzido?)
  - ->resolucao   : o modelo rastreia resolucao? (quanto menor, mais honesto)
  - global/controlado

Uso: python scripts/compare_preprocess.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from siamese.config import Config
from siamese.backbone import DinoV2Backbone, BackboneConfig
from siamese.features import extract_processed
from siamese.synth_features import extract_synthetic
from siamese.train import train_head
from siamese.evaluate import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def run_mode(base: Config, mode: str) -> dict:
    cfg = copy.deepcopy(base)
    cfg.backbone.preprocess = mode
    cfg.backbone.use_patch_stats = True
    root = Path(f"artifacts/cmp_{mode}")
    cfg.paths.emb_dir = str(root / "emb")
    cfg.paths.models_dir = str(root / "models")
    cfg.paths.reports_dir = str(root / "reports")

    bb = DinoV2Backbone(BackboneConfig(size=cfg.backbone.size, use_patch_stats=True,
                                       preprocess=mode, device=DEVICE))
    emb = Path(cfg.paths.emb_dir)
    processed = Path("data/processed")
    # reais + train_synth da FONTE DA VERDADE (data/processed/)
    extract_processed(processed, emb, bb, batch_size=cfg.backbone.batch_size)
    # sonda livre de confound SO da VAL (anti-snooping: nunca toca test_synth aqui)
    d = processed / "val" / "real" / "clean"
    rows = [{"path": str(p.resolve())} for p in sorted(d.iterdir()) if p.suffix.lower() in _EXTS]
    if rows:
        extract_synthetic(None, emb / "val_synth.npz", bb,
                          n_variants=cfg.synthetic.n_variants,
                          max_errors_per_image=cfg.synthetic.max_errors_per_image,
                          seed=100, batch_size=cfg.backbone.batch_size,
                          multiclass=cfg.train.multiclass, clean_rows=rows)
    train_head(cfg, device=DEVICE)
    return evaluate(cfg, device=DEVICE)   # modo DEV (val); teste blindado


def main() -> None:
    base = Config.load("configs/default.yaml")
    out = {}
    for mode in ("resize", "pad"):
        print(f"\n################## modo = {mode} ##################")
        out[mode] = run_mode(base, mode)

    print("\n\n=========== RESIZE vs PAD (test) ===========")
    print(f"{'metrica':32s} {'resize':>10s} {'pad':>10s}")
    def row(label, fn):
        print(f"{label:32s} {fn(out['resize']):10.3f} {fn(out['pad']):10.3f}")
    row("synt_AUROC (livre confound)", lambda r: r["sintetico_livre_de_confound"]["modelo_fusao"]["auroc"])
    row("synt_AP    (livre confound)", lambda r: r["sintetico_livre_de_confound"]["modelo_fusao"]["ap"])
    row("global_AUROC (confundido)", lambda r: r["global_vs_baselines"]["modelo_fusao"]["auroc"])
    row("controlado_AUROC", lambda r: r.get("primaria_subconjunto_controlado", {}).get("modelo_fusao", {}).get("auroc", float("nan")))
    row("base fracao_padding_cinza", lambda r: r["global_vs_baselines"].get("baseline_fracao_padding_cinza", {}).get("auroc", float("nan")))
    row("->modelo prediz RESOLUCAO", lambda r: r["falseabilidade"].get("auroc_modelo_predizendo_resolucao", float("nan")))
    row("->modelo prediz ERRO", lambda r: r["falseabilidade"].get("auroc_modelo_predizendo_erro", float("nan")))
    p95 = lambda r: r["limiar_por_precisao"]["precisao_alvo_0.95"]
    row("precisao@alvo0.95 (test)", lambda r: p95(r)["test_precision"])
    row("recall@alvo0.95 (test)", lambda r: p95(r)["test_recall"])
    print("\nLeitura: melhor pad se synt_AUROC sobe E pad_frac/->resolucao NAO sobem.")


if __name__ == "__main__":
    main()
