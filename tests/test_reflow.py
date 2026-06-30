"""Testes do modulo de REFLOW (negativos de layout legitimo).

Invariantes que sustentam o uso do reflow como NEGATIVO (label=clean):
  - operadores rodam e devolvem PIL RGB; ops sem mudanca de canvas preservam o tamanho.
  - PRINCIPIO DE NAO-COLISAO: reflow PRESERVA conteudo (nao cria regiao morta uniforme) — uma
    variante de reflow nunca pode "parecer um bug" de regiao preta/vazia.
  - ar_relayout MUDA a resolucao (e' a alavanca que quebra o confound pelo lado limpo).
  - determinismo por seed; pesos zerados -> identidade.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from siamese.reflow import (reflow_augment, REFLOW_OPS, DEFAULT_REFLOW_WEIGHTS, _FUNCS,
                            _ar_relayout)


def _textured(w=320, h=400, seed=0):
    """Imagem com textura (gradiente + ruido + blocos) — conteudo 'vivo' nao uniforme."""
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w, 3), np.uint8)
    yy = np.linspace(0, 255, h).astype(np.uint8)
    a[:] = yy[:, None, None]                         # gradiente vertical
    a[h // 4:h // 2, w // 4:w // 2] = [200, 50, 50]  # bloco colorido
    a = np.clip(a.astype(int) + rng.integers(-20, 20, a.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def test_operadores_rodam_e_tipo():
    img = _textured()
    rng = random.Random(0)
    for op in REFLOW_OPS:
        out = _FUNCS[op](img, rng)
        assert isinstance(out, Image.Image) and out.mode == "RGB"


def test_ops_sem_mudanca_de_canvas_preservam_tamanho():
    img = _textured()
    rng = random.Random(1)
    for op in ("scroll_shift", "two_pane", "band_jitter"):
        assert _FUNCS[op](img, rng).size == img.size


def test_ar_relayout_muda_resolucao():
    """A propriedade que QUEBRA o confound: variante limpa sai da resolucao canonica."""
    img = _textured(w=400, h=400)   # quadrado -> ar_relayout deve desquadrar
    changed = False
    for s in range(8):
        out = _FUNCS["ar_relayout"](img, random.Random(s))
        if out.size != img.size:
            changed = True
    assert changed, "ar_relayout deveria mudar a resolucao em pelo menos um sorteio"


def test_nao_colisao_preserva_conteudo():
    """Reflow nao pode virar regiao morta uniforme (assinatura de bug). A variancia de pixel
    da saida deve permanecer comparavel a da entrada (conteudo preservado)."""
    img = _textured()
    base_std = float(np.asarray(img).std())
    rng = random.Random(7)
    for _ in range(20):
        out, ops = reflow_augment(img, rng, max_ops=2)
        std = float(np.asarray(out).std())
        assert std > 0.4 * base_std, f"reflow {ops} achatou o conteudo (std {std:.1f} << {base_std:.1f})"


def test_pesos_zerados_identidade():
    img = _textured()
    out, ops = reflow_augment(img, random.Random(0), ops_weights={k: 0.0 for k in REFLOW_OPS})
    assert ops == [] and np.array_equal(np.asarray(out), np.asarray(img))


def test_determinismo_por_seed():
    img = _textured()
    o1, a1 = reflow_augment(img, random.Random(42))
    o2, a2 = reflow_augment(img, random.Random(42))
    assert a1 == a2 and np.array_equal(np.asarray(o1), np.asarray(o2))


def test_ar_relayout_mira_aspecto_alvo():
    """Fase 2.4: com target_aspects, ar_relayout reescala a limpa para perto do AR-alvo (w/h).
    Ex.: imagem 9:16 (phone, AR 0.5625) mirando near-square 0.96 deve sair ~near-square."""
    img = _textured(w=450, h=800)                 # AR = 0.5625 (phone retrato)
    ars = []
    for s in range(40):
        out = _ar_relayout(img, random.Random(s), target_aspects=[0.96])
        w, h = out.size
        ars.append(w / h)
    med = float(np.median(ars))
    assert 0.86 <= med <= 1.07, f"AR mediano {med:.3f} longe do alvo 0.96 (jitter ~6%)"
    # e claramente diferente do AR original (0.56) -> de fato remapeou
    assert med > 0.75, "nao remapeou para o alvo near-square"


def test_ar_relayout_alvo_multiplo_cobre_distribuicao():
    """Com varios alvos (distrib. de erro), as saidas espalham perto dos alvos pedidos."""
    img = _textured(w=400, h=400)
    targets = [0.45, 0.96, 1.78]                  # phone-retrato, near-square, desktop-paisagem
    near_sq = 0
    for s in range(60):
        out = _ar_relayout(img, random.Random(s), target_aspects=targets)
        w, h = out.size
        if 0.85 <= w / h <= 1.18:
            near_sq += 1
    assert near_sq > 0, "nenhuma saida near-square apesar de 0.96 estar nos alvos"


def test_ar_relayout_sem_alvo_mantem_comportamento_legado():
    """Sem target_aspects, segue o aspecto aleatorio U(0.5,2.0) (retrocompat)."""
    img = _textured(w=400, h=400)
    sizes = {_ar_relayout(img, random.Random(s)).size for s in range(8)}
    assert len(sizes) > 1, "deveria variar a resolucao aleatoriamente sem alvos"


def test_ar_relayout_vai_por_ultimo():
    """Quando ar_relayout e' sorteado junto com outro op, deve ser aplicado por ULTIMO
    (muda o canvas) — verifica a ordenacao em reflow_augment."""
    img = _textured()
    for s in range(30):
        _, ops = reflow_augment(img, random.Random(s), max_ops=2)
        if "ar_relayout" in ops and len(ops) > 1:
            assert ops[-1] == "ar_relayout"
