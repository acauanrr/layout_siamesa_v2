"""Anti-vazamento no dataset PLANO de PRODUCAO (data/processed_v3): nenhum GRUPO real
(ticket de erro / sessao de captura limpa) pode aparecer em mais de um split, e os sinteticos
so existem no train (sao injetados nas limpas de treino). Complementa test_split_isolation.py
(que cobre a arvore legada data/processed). Pula se o dataset nao estiver materializado."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

LABELS = Path(__file__).resolve().parent.parent / "data" / "processed_v3" / "labels.csv"
pytestmark = pytest.mark.skipif(not LABELS.exists(), reason="data/processed_v3 ausente")


def _rows():
    with open(LABELS, newline="") as f:
        return list(csv.DictReader(f))


def test_no_real_group_crosses_splits():
    g2s = defaultdict(set)
    for r in _rows():
        if r["source"] == "real":
            g2s[r["group"]].add(r["split"])
    leaks = {g: sorted(v) for g, v in g2s.items() if len(v) > 1}
    assert not leaks, f"grupos reais cruzando splits (vazamento): {dict(list(leaks.items())[:5])}"


def test_synthetic_lives_only_in_train():
    splits = {r["split"] for r in _rows() if r["source"] == "synthetic"}
    assert splits <= {"train"}, f"sintetico fora do train (sonda val/test e' gerada a parte): {splits}"
