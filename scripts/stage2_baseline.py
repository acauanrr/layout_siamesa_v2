#!/usr/bin/env python
"""Baseline OBRIGATORIO do Estagio 2 (plano de acao, Fase 5.5).

Classificadores SIMPLES de categoria sobre as features DINOv2 CRUAS (sem a cabeca siamesa),
para responder honestamente: o decisor por PROTOTIPO no espaco aprendido (z) supera um
baseline trivial nas mesmas features congeladas? Reporta F1-macro nas taxonomias FINA (4
classes) e GROSSA (2 super-classes), comparavel ao bloco `estagio2_categoria` do
evaluation_report (oraculo = todas as imagens de erro classificadas).

DEV-only: treina em train (erros reais), avalia em val (erros reais). NAO toca o teste
(protocolo): so le train.npz/val.npz via siamese.features.load_embeddings.

Uso:  python scripts/stage2_baseline.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

from siamese.config import Config
from siamese.features import load_embeddings
from siamese.manifest import CATEGORIES, COARSE_CATEGORIES, coarse_of


def _errors(npz_path: Path):
    """Features L2-normalizadas + slug de categoria das imagens de ERRO (label==1)."""
    d = load_embeddings(npz_path)
    m = d["label"] == 1
    X = normalize(d["emb"][m].astype(np.float64))
    y = np.array([str(c) for c in d["category"][m]])
    return X, y


def _scores(y_true, y_pred, fine_classes, coarse_classes):
    return {
        "f1_macro_fina": float(f1_score(list(y_true), list(y_pred), average="macro",
                                        labels=fine_classes, zero_division=0)),
        "f1_macro_grossa": float(f1_score([coarse_of(s) for s in y_true],
                                          [coarse_of(s) for s in y_pred], average="macro",
                                          labels=coarse_classes, zero_division=0)),
        "acc_fina": float(accuracy_score(list(y_true), list(y_pred))),
        "acc_grossa": float(accuracy_score([coarse_of(s) for s in y_true],
                                           [coarse_of(s) for s in y_pred])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = ap.parse_args()
    cfg = Config.load(args.config)
    emb = Path(cfg.paths.emb_dir)

    Xtr, ytr = _errors(emb / "train.npz")
    Xva, yva = _errors(emb / "val.npz")
    fine = [c for c in CATEGORIES if c != "clean"]
    coarse = [c for c in COARSE_CATEGORIES if c != "clean"]

    out = {
        "_modo": "DEV (treina em train erros, avalia em val erros; teste NAO tocado)",
        "n_train_err": int(len(ytr)), "n_val_err": int(len(yva)),
        "suporte_val_fina": dict(Counter(yva)),
        "suporte_val_grossa": dict(Counter(coarse_of(s) for s in yva)),
    }

    # Baseline 1: regressao logistica multinomial sobre features cruas.
    lr = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")
    lr.fit(Xtr, ytr)
    out["logreg_raw"] = _scores(yva, lr.predict(Xva), fine, coarse)

    # Baseline 2: centroide mais proximo (cosseno) sobre features cruas — analogo "sem cabeca"
    # do decisor por prototipo (mede o ganho da projecao siamesa aprendida).
    uniq = sorted(set(ytr))
    C = normalize(np.stack([Xtr[ytr == c].mean(0) for c in uniq]))
    pred_nc = np.array([uniq[i] for i in (Xva @ C.T).argmax(1)])
    out["nearest_centroid_raw"] = _scores(yva, pred_nc, fine, coarse)

    rep = Path(cfg.paths.reports_dir)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "stage2_baseline.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"Estagio 2 — BASELINE por categoria (features DINOv2 cruas)  "
          f"[train_err={len(ytr)}  val_err={len(yva)}]")
    print(f"{'metodo':>22s} | {'F1macro_fina':>12s} {'F1macro_grossa':>14s} "
          f"{'acc_fina':>9s} {'acc_grossa':>10s}")
    for name in ("logreg_raw", "nearest_centroid_raw"):
        s = out[name]
        print(f"{name:>22s} | {s['f1_macro_fina']:>12.3f} {s['f1_macro_grossa']:>14.3f} "
              f"{s['acc_fina']:>9.3f} {s['acc_grossa']:>10.3f}")
    print("Compare com evaluation_report_dev.json -> estagio2_categoria.oraculo "
          "(prototipo no z aprendido). Baseline obrigatorio do plano (Fase 5.5).")
    print(f"-> {rep / 'stage2_baseline.json'}")


if __name__ == "__main__":
    main()
