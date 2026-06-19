"""Teste de Imagem-Mae (Fase 6): a tela-mae limpa e (por construcao) os seus derivados
sinteticos nunca caem em splits diferentes. Como os sinteticos sao gerados POR SPLIT a partir
das limpas daquele split, basta garantir que cada GRUPO de sessao limpa seja atomico (1 split).
Tambem testa a unidade de agrupamento de sessao/near-dup."""
from __future__ import annotations

import collections

import pytest

from siamese.manifest import (clean_session_components, assign_clean_session_groups,
                              scan_dataset, grouped_stratified_split)
from conftest import ROOT


def test_session_grouping_unit():
    # capturas a 10s e 30s -> mesma sessao; 2h depois -> outra
    names = [
        "Screenshot_20260101_100000.png",  # t0
        "Screenshot_20260101_100010.png",  # +10s
        "Screenshot_20260101_100040.png",  # +40s
        "Screenshot_20260101_120000.png",  # +2h
    ]
    g = clean_session_components(names, gap_seconds=300)
    assert g[names[0]] == g[names[1]] == g[names[2]], "capturas proximas deviam ser 1 sessao"
    assert g[names[0]] != g[names[3]], "captura 2h depois deviam ser sessao distinta"


def test_phash_does_not_overmerge_distinct_screens():
    # dois nomes sem timestamp comum e hashes distantes -> grupos distintos (sem colapso)
    names = ["a.png", "b.png"]
    far = {"a.png": 0, "b.png": (1 << 60) | 0b1010101010}  # distancia de Hamming grande
    g = clean_session_components(names, gap_seconds=300, phash_of=lambda n: far[n], phash_max_dist=4)
    assert g["a.png"] != g["b.png"]


@pytest.mark.skipif(not (ROOT / "data" / "input" / "no_erros").is_dir(),
                    reason="data/input ausente")
def test_clean_mother_groups_are_split_atomic():
    samples = scan_dataset(ROOT / "data" / "input", source="errors_dataset")
    samples = assign_clean_session_groups(samples)        # so timestamp (sem ler pixels)
    samples = grouped_stratified_split(samples, val_frac=0.15, test_frac=0.24, seed=42)
    # cada grupo limpo (sessao = mae + suas variantes) deve estar em 1 unico split
    g2s = collections.defaultdict(set)
    for s in samples:
        if s.source == "no_erros":
            g2s[s.group].add(s.split)
    assert all(len(v) == 1 for v in g2s.values()), "uma sessao-mae limpa cruzou splits"
