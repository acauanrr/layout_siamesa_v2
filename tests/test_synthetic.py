"""Testes dos geradores de anomalias sinteticas (src/siamese/synthetic.py) — o NUCLEO
anti-confound (injeta erro na MESMA resolucao da limpa, p/ o modelo aprender o conteudo do
erro e nao o device). Garante: (1) cada gerador preserva resolucao/modo e ALTERA a imagem;
(2) inject() respeita os tipos multi-classe e a resolucao; (3) determinismo por seed;
(4) o mapa tipo->categoria cobre exatamente as 4 categorias reais."""
from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from siamese.synthetic import (
    inject, SYNTH_TO_CATEGORY, MULTICLASS_SYNTH_TYPES, ERROR_TYPES, _FUNCS,
)
from siamese.manifest import CATEGORY_TO_ID


def _clean_img(w=140, h=150):
    """Imagem 'limpa' sintetica com conteudo (gradiente + bloco) p/ os geradores mexerem."""
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    a[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    a[30:70, 20:90] = 180
    return Image.fromarray(a)


@pytest.mark.parametrize("name", ERROR_TYPES)
def test_generator_preserves_geometry_and_changes(name):
    img = _clean_img()
    out = _FUNCS[name](img, random.Random(0))
    assert out.size == img.size, f"{name} alterou a resolucao (quebra o pareamento anti-confound)"
    assert out.mode == "RGB"
    assert not np.array_equal(np.asarray(out), np.asarray(img)), f"{name} nao alterou a imagem"


def test_inject_multiclass_preserves_resolution_and_maps_category():
    img = _clean_img(160, 170)
    corr, applied = inject(img, random.Random(1), n_errors=1, types=MULTICLASS_SYNTH_TYPES)
    assert corr.size == img.size
    assert len(applied) == 1 and applied[0] in MULTICLASS_SYNTH_TYPES
    cat = SYNTH_TO_CATEGORY[applied[0]]
    assert cat in CATEGORY_TO_ID and cat != "clean"


def test_inject_is_deterministic_under_seed():
    img = _clean_img()
    a_img, a_types = inject(img, random.Random(42), n_errors=2, types=MULTICLASS_SYNTH_TYPES)
    b_img, b_types = inject(img, random.Random(42), n_errors=2, types=MULTICLASS_SYNTH_TYPES)
    assert a_types == b_types
    assert np.array_equal(np.asarray(a_img), np.asarray(b_img))


def test_multiclass_types_cover_the_four_real_categories():
    cats = {SYNTH_TO_CATEGORY[t] for t in MULTICLASS_SYNTH_TYPES}
    assert cats == {"black_bars", "empty_space", "overlay", "disordered_layout"}
    assert SYNTH_TO_CATEGORY["cropped"] is None  # 'cropped' nao tem categoria real (so anti-confound)
