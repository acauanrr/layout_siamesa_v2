"""Tratamento de escopo (Fase 6): categoria fora do escopo levanta excecao ou abstem —
NUNCA e' rotulada como 'clean' silenciosamente (era o bug de CATEGORY_TO_ID.get(c, 0))."""
from __future__ import annotations

import pytest

from siamese.manifest import category_id, CATEGORY_TO_ID, CLEAN_ID, ABSTAIN_ID


def test_clean_and_empty_map_to_clean():
    assert category_id("clean") == CLEAN_ID == 0
    assert category_id("") == CLEAN_ID
    assert category_id("CLEAN") == CLEAN_ID          # case-insensitive


def test_known_slugs_roundtrip():
    for c, i in CATEGORY_TO_ID.items():
        assert category_id(c) == i


def test_unknown_category_raises_strict():
    with pytest.raises(KeyError):
        category_id("categoria_inexistente_xyz")


def test_unknown_category_abstains_not_clean():
    cid = category_id("categoria_inexistente_xyz", strict=False)
    assert cid == ABSTAIN_ID == -1
    assert cid != CLEAN_ID, "categoria desconhecida NUNCA pode virar 'clean'"
