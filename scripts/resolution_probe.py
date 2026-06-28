#!/usr/bin/env python
"""Fase 2 — PORTAO DE ACEITE do anti-confound: a RESOLUCAO ainda prediz clean-vs-erro?

No processed_v3 toda limpa e' 2076x2152 e os erros sao heterogeneos -> resolucao separa quase
perfeito (regra trivial AUROC 1.0; ver evaluate.baseline_resolucao_trivial). Apos juntar limpas
diversas (processed_v3_plus), esta sonda deve CAIR rumo a 0.5 — resolucao deixa de ser atalho.

Treina LogisticRegression (5-fold CV) sobre [W, H, aspecto, escala-p/-518] -> rotulo (tem erro?)
usando SO os reais. Reporta AUROC media. Compara um ou mais datasets lado a lado.

Uso:
    python scripts/resolution_probe.py data/processed_v3 data/processed_v3_plus
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def probe(root: Path):
    feats, ys = [], []
    with open(root / "labels.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("source") != "real":
                continue
            p = root / r["path"]
            if not p.exists():
                continue
            with Image.open(p) as im:
                w, h = im.size
            feats.append([w, h, w / h, max(w, h) / 518.0])
            ys.append(0 if r["category"] == "clean" else 1)
    X, y = np.array(feats, float), np.array(ys)
    if len(set(y.tolist())) < 2:
        return None
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], lr.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs)), int((y == 0).sum()), int(y.sum())


def main():
    roots = [Path(a) for a in sys.argv[1:]] or [Path("data/processed_v3")]
    print(f"{'dataset':26s} {'AUROC_resolucao':>16s}   clean/erro")
    for root in roots:
        if not (root / "labels.csv").exists():
            print(f"{str(root):26s}   (sem labels.csv)")
            continue
        res = probe(root)
        if res is None:
            print(f"{str(root):26s}   (1 classe so)")
            continue
        m, s, nc, ne = res
        flag = " <- CONFOUND" if m > 0.85 else " <- quebrando" if m > 0.65 else " <- QUEBRADO"
        print(f"{str(root):26s}   {m:.3f} +/- {s:.3f}   {nc}/{ne}{flag}")
    print("\n~1.0 = resolucao separa tudo (confound); ~0.5 = resolucao nao prediz (quebrado).")


if __name__ == "__main__":
    main()
