"""Isolamento de folds da Grouped Nested CV (Fase 2): nenhum grupo cruza outer folds; a
imagem-mae e seus derivados (mesmo grupo) nunca se separam; cobertura OOF total e determinismo."""
from __future__ import annotations

import numpy as np

from siamese.cv import grouped_kfold, grouped_holdout


def test_grouped_kfold_no_group_crosses_and_full_coverage():
    groups = np.array([f"g{i % 13}" for i in range(100)])
    seen = np.zeros(100, dtype=bool)
    for tr, te in grouped_kfold(groups, n_splits=5, seed=0):
        assert set(groups[tr]).isdisjoint(set(groups[te])), "grupo cruzou train/test no fold"
        assert not seen[te].any(), "indice aparece em mais de um fold de teste"
        seen[te] = True
    assert seen.all(), "cobertura OOF incompleta (item sem fold de teste)"


def test_grouped_kfold_deterministic_by_seed():
    groups = np.array([f"g{i % 9}" for i in range(60)])
    a = [tuple(te.tolist()) for _, te in grouped_kfold(groups, 4, seed=1)]
    b = [tuple(te.tolist()) for _, te in grouped_kfold(groups, 4, seed=1)]
    c = [tuple(te.tolist()) for _, te in grouped_kfold(groups, 4, seed=2)]
    assert a == b, "mesmo seed deveria dar a mesma particao"
    assert a != c, "seeds diferentes deveriam variar a particao"


def test_mother_image_group_never_split():
    # itens 'sinteticos' compartilham o grupo da imagem-mae limpa -> nunca se separam num fold
    groups = np.array(["sess0", "sess0", "sess0", "sess1", "sess1", "sess2", "sess2", "sess2", "sess3"])
    for tr, te in grouped_kfold(groups, n_splits=3, seed=7):
        for g in set(groups.tolist()):
            idx = np.where(groups == g)[0]
            all_tr = np.isin(idx, tr).all()
            all_te = np.isin(idx, te).all()
            assert all_tr or all_te, f"grupo (imagem-mae) {g} foi dividido entre folds"


def test_grouped_holdout_is_group_disjoint():
    groups = np.array([f"g{i % 8}" for i in range(48)])
    idx = np.arange(48)
    in_tr, in_val = grouped_holdout(idx, groups, frac=0.25, seed=3)
    assert set(in_tr).isdisjoint(set(in_val)), "indices repetidos entre inner-train e inner-val"
    assert set(groups[in_tr]).isdisjoint(set(groups[in_val])), "grupo cruzou inner-train/inner-val"
    assert len(in_val) > 0 and len(in_tr) > 0
