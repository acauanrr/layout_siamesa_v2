"""Cabeca siamesa sobre embeddings DINOv2 congelados.

A "rede siamesa" aqui = a MESMA funcao g (pesos compartilhados) aplicada a qualquer
imagem, mapeando o embedding DINOv2 (D-dim) para um espaco metrico z (L2-normalizado)
onde "tela limpa" forma um cluster compacto e "tela com erro" cai fora. Comparar duas
entradas = comparar z1 e z2 (distancia/cosine). Isto satisfaz a definicao de siamesa
(ramos de pesos compartilhados + comparacao no espaco latente) sem cair na armadilha de
"alvo vs uma unica referencia boa" (telas limpas de apps diferentes sao legitimamente
distintas; por isso comparamos contra PROTOTIPOS do cluster limpo, nao contra 1 imagem).

Duas cabecas:
  ProjectionHead  -> embedding metrico z (para SupCon/contrastiva + decisao por prototipo)
  SiamesePairHead -> cabeca de fusao [z1,z2,|z1-z2|,z1*z2] + sigmoid (formulacao pareada
                     classica que o usuario descreveu), opcional.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """g(x): D -> proj_dim, L2-normalizado. Encoder siames de pesos compartilhados."""

    def __init__(self, in_dim: int, hidden: int = 256, proj_dim: int = 128,
                 p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, proj_dim),
        )
        self.proj_dim = proj_dim

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1) if normalize else z


class SiamesePairHead(nn.Module):
    """Cabeca de classificacao pareada: recebe dois embeddings projetados e devolve
    o logit de P(par dissimilar / erro). Vetor de fusao [z1, z2, |z1-z2|, z1*z2]."""

    def __init__(self, proj_dim: int = 128, hidden: int = 128, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * proj_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        fusion = torch.cat([z1, z2, (z1 - z2).abs(), z1 * z2], dim=-1)
        return self.net(fusion).squeeze(-1)


class SiameseNet(nn.Module):
    """Modelo de producao: encoder de projecao COMPARTILHADO (g) + cabecalho auxiliar
    de classificacao binaria direta sobre z.

    - z = g(x)         -> espaco metrico L2-normalizado (treinado por SupCon/contrastiva)
    - aux_logit = w.z  -> detector binario direto (nao depende de banco de referencias)

    A decisao final funde score_de_prototipo(z) com aux_logit (ver decision.py / evaluate).
    """

    def __init__(self, in_dim: int, hidden: int = 256, proj_dim: int = 128, p_drop: float = 0.3):
        super().__init__()
        self.proj = ProjectionHead(in_dim, hidden, proj_dim, p_drop)
        self.aux = nn.Linear(proj_dim, 1)
        self.proj_dim = proj_dim

    def forward(self, x: torch.Tensor):
        z = self.proj(x, normalize=True)
        aux_logit = self.aux(z).squeeze(-1)
        return z, aux_logit
