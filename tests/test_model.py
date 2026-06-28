"""Smoke test da cabeca siamesa (src/siamese/model.py): formas e L2-normalizacao de z,
e o gate binario (num_classes==1) que devolve um logit por amostra (squeeze)."""
from __future__ import annotations

import torch

from siamese.model import SiameseNet, ProjectionHead


def test_projection_head_is_l2_normalized():
    h = ProjectionHead(64, hidden=32, proj_dim=16)
    z = h(torch.randn(8, 64))
    assert z.shape == (8, 16)
    assert torch.allclose(z.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_siamesenet_multiclass_shapes():
    net = SiameseNet(64, hidden=32, proj_dim=16, num_classes=5)
    z, aux = net(torch.randn(4, 64))
    assert z.shape == (4, 16) and aux.shape == (4, 5)


def test_siamesenet_binary_squeezes_logit():
    net = SiameseNet(64, hidden=32, proj_dim=16, num_classes=1)
    z, aux = net(torch.randn(3, 64))
    assert z.shape == (3, 16) and aux.shape == (3,)
