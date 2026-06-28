"""Funcoes de perda para a cabeca siamesa.

- supcon_loss: Supervised Contrastive (Khosla et al. 2020). Aproxima embeddings da MESMA
  classe e afasta de classes diferentes. Escolhida em vez da contrastiva-por-par classica
  porque usa todos os positivos/negativos do batch (mais estavel com poucos dados) e nao
  exige uma referencia unica (a classe "limpa" e diversa).
- contrastive_loss: contrastiva de pares (Hadsell 2006), incluida para a formulacao
  pareada classica que o usuario descreveu.
- BCE pareada: usar nn.BCEWithLogitsLoss diretamente sobre SiamesePairHead.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss.

    z: [B, d] L2-normalizado. labels: [B] inteiros.
    Para cada ancora i, positivos = mesmas labels (exceto i); minimiza -log da razao
    softmax sobre os positivos. Ancoras sem positivo no batch sao ignoradas.
    """
    device = z.device
    B = z.shape[0]
    sim = z @ z.t() / temperature                      # [B, B]
    # estabilidade numerica
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & ~self_mask     # mesmos rotulos, sem a diagonal

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    if valid.sum() == 0:
        return z.sum() * 0.0
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_counts[valid]
    return -mean_log_prob_pos.mean()


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, y: torch.Tensor,
                     margin: float = 1.0) -> torch.Tensor:
    """Contrastiva de pares. y=0 par similar (mesma classe), y=1 dissimilar.
    Usa distancia euclidiana entre embeddings L2-normalizados."""
    d = F.pairwise_distance(z1, z2)
    loss_sim = (1 - y) * 0.5 * d.pow(2)
    loss_dis = y * 0.5 * F.relu(margin - d).pow(2)
    return (loss_sim + loss_dis).mean()


def triplet_loss(z: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Triplet loss com ONLINE BATCH-HARD mining (Hermans et al. 2017), vetorizada.

    z: [B, d] L2-normalizado. labels: [B] inteiros (categoria). Para cada ancora i, escolhe o
    POSITIVO mais distante (mesma classe) e o NEGATIVO mais proximo (classe diferente) DENTRO do
    batch — os trios mais dificeis — e minimiza relu(d(a,p_hard) - d(a,n_hard) + margin). Distancia
    euclidiana sobre z normalizado (= sqrt(2-2cos)). Ancoras sem positivo OU sem negativo no batch
    sao ignoradas (o sampler balanceado garante >=2/classe, entao positivos existem).

    Alternativa configuravel ao SupCon (train.loss='triplet'). SupCon usa TODOS os positivos/
    negativos por batch (sem mining); o batch-hard foca no trio mais dificil — comparativo empirico
    em scripts/compare_methods.py.
    """
    B = z.shape[0]
    d = torch.cdist(z, z)                                  # [B,B] distancia euclidiana
    same = labels.view(-1, 1) == labels.view(1, -1)        # [B,B] mesma classe
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    pos_mask = same & ~eye                                 # positivos (sem a diagonal)
    neg_mask = ~same                                       # negativos
    # hardest positive = MAIOR distancia entre mesma classe; hardest negative = MENOR entre classes
    d_pos = d.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    d_neg = d.masked_fill(~neg_mask, float("inf")).min(dim=1).values
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if valid.sum() == 0:
        return z.sum() * 0.0
    return F.relu(d_pos[valid] - d_neg[valid] + margin).mean()
