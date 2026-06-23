#!/usr/bin/env python
"""SONDA DE NAO-COLISAO do reflow (rode ANTES de confiar no treino com reflow-clean).

O reflow so e' um bom negativo se NAO "parecer um bug" (principio de nao-colisao, reflow.py).
Esta sonda mede, nos embeddings DINOv2 CRUS (sem a cabeca treinada — diagnostico independente):

  1. AUROC(limpo-real vs limpo-REFLOW)  -> deve ser ~0.5  (reflow indistinguivel de limpo = bom)
  2. AUROC(limpo-real vs erro-SINTETICO) -> deve ser ALTO  (erro E' distinguivel = sanidade)
  3. distancia cosseno media ao centroide limpo: reflow ~ limpo  <<  erro
  4. quebra POR OPERADOR de reflow (scroll_shift/two_pane/ar_relayout/band_jitter): qual e' o
     mais "parecido com erro"? (auditoria especifica de band_jitter vs distortion, two_pane vs faixa)

Se algum operador colidir (AUROC limpo-vs-reflow alto, ou dist ~ erro), reduza o peso dele em
configs/default.yaml synthetic.reflow_ops ANTES de treinar. (DEV-only: usa train/val; nunca test.)

Uso: python scripts/audit_reflow.py --config configs/default.yaml [--split train|val]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

from siamese.config import Config
from siamese.features import load_embeddings


def _centroid_dist(z: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return 1.0 - normalize(z) @ centroid.T.ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--split", choices=["train", "val"], default="train",
                    help="split de onde vem as limpas/reflow (test fica trancado)")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    emb = Path(cfg.paths.emb_dir)

    real = load_embeddings(emb / f"{args.split}.npz")
    clean = real["emb"][real["label"] == 0]
    rf_path = emb / f"{args.split}_reflow.npz"
    sy_path = emb / f"{args.split}_synth.npz" if args.split != "train" else emb / "train_synth.npz"
    if not rf_path.exists():
        raise SystemExit(f"{rf_path} ausente — rode scripts/make_synthetic.py primeiro.")
    reflow = load_embeddings(rf_path)
    synth = load_embeddings(sy_path) if sy_path.exists() else None

    centroid = normalize(normalize(clean).mean(0, keepdims=True))
    d_clean = _centroid_dist(clean, centroid)
    d_reflow = _centroid_dist(reflow["emb"], centroid)

    print(f"\n===== SONDA DE NAO-COLISAO DO REFLOW ({args.split}, DINOv2 CRU) =====")
    print(f"  n: limpo-real={len(clean)}  reflow={len(reflow['emb'])}"
          + (f"  erro-synth={len(synth['emb'])}" if synth is not None else ""))

    # (1) limpo-real vs reflow -> ~0.5 desejado
    y1 = np.r_[np.zeros(len(clean)), np.ones(len(reflow["emb"]))]
    auc_rf = roc_auc_score(y1, np.r_[d_clean, d_reflow])
    flag_rf = "OK (~0.5, nao colide)" if auc_rf <= 0.65 else "ATENCAO: reflow separavel de limpo!"
    print(f"\n  (1) AUROC(limpo vs reflow)  = {auc_rf:.3f}   [{flag_rf}]")

    # (2) limpo-real vs erro-synth -> ALTO desejado
    if synth is not None:
        d_err = _centroid_dist(synth["emb"], centroid)
        y2 = np.r_[np.zeros(len(clean)), np.ones(len(synth["emb"]))]
        auc_err = roc_auc_score(y2, np.r_[d_clean, d_err])
        print(f"  (2) AUROC(limpo vs erro)    = {auc_err:.3f}   [sanidade: deve ser ALTO]")
        # (2b) reflow vs erro -> reflow deve ficar do LADO do limpo
        y3 = np.r_[np.zeros(len(reflow["emb"])), np.ones(len(synth["emb"]))]
        auc_re = roc_auc_score(y3, np.r_[d_reflow, d_err])
        print(f"  (2b) AUROC(reflow vs erro)  = {auc_re:.3f}   [alto = reflow fica do lado LIMPO]")

    # (3) distancias medias
    print(f"\n  (3) dist. media ao centroide limpo:")
    print(f"      limpo-real = {d_clean.mean():.3f}   reflow = {d_reflow.mean():.3f}"
          + (f"   erro = {d_err.mean():.3f}" if synth is not None else ""))
    print(f"      (desejado: reflow ~ limpo  <<  erro)")

    # (4) quebra por operador
    print(f"\n  (4) por OPERADOR de reflow (dist media + 'error-likeness' = AUROC vs limpo):")
    applied = reflow.get("applied", np.array([""] * len(reflow["emb"])))
    ops = ["scroll_shift", "two_pane", "ar_relayout", "band_jitter"]
    rows = []
    for op in ops:
        m = np.array([op in str(a).split("+") for a in applied])
        if m.sum() == 0:
            continue
        d_op = d_reflow[m]
        yo = np.r_[np.zeros(len(clean)), np.ones(int(m.sum()))]
        auc_op = roc_auc_score(yo, np.r_[d_clean, d_op])
        rows.append((op, int(m.sum()), float(d_op.mean()), float(auc_op)))
    for op, n, dm, auc in sorted(rows, key=lambda r: -r[3]):
        warn = "  <-- mais error-like" if auc > 0.70 else ""
        print(f"      {op:14s} n={n:4d}  dist={dm:.3f}  AUROC_vs_limpo={auc:.3f}{warn}")

    print(f"\n  VEREDITO: {'reflow NAO colide com erro -> seguro treinar.' if auc_rf <= 0.65 else 'AJUSTE os pesos de reflow_ops antes de treinar.'}")


if __name__ == "__main__":
    main()
