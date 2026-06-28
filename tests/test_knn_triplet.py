"""Testes de unidade do código novo: Triplet loss (batch-hard) + decisores k-NN."""
import numpy as np
import torch
import torch.nn.functional as F

from siamese.losses import triplet_loss
from siamese.decision import KNNDecider, KNNCategoryClassifier, PrototypeDecider, fit_prototypes


# ---------------------------------------------------------------- Triplet loss
def _norm(x):
    return F.normalize(x, dim=1)


def test_triplet_zero_quando_separado_perfeitamente():
    # cada classe num eixo ortogonal -> hardest_pos=0, hardest_neg=sqrt(2) -> relu(0-1.41+0.5)=0
    y = torch.tensor([0, 1, 2, 3] * 3)
    z = _norm(torch.eye(4)[y].float())
    assert triplet_loss(z, y, margin=0.5).item() == 0.0


def test_triplet_positivo_quando_misturado():
    torch.manual_seed(0)
    y = torch.tensor([0, 1, 2, 3, 4] * 4)
    z = _norm(torch.randn(20, 8))
    loss = triplet_loss(z, y, margin=0.5)
    assert torch.isfinite(loss) and loss.item() > 0


def test_triplet_batch_monoclasse_nao_quebra_backward():
    # sem negativos -> nenhuma âncora válida -> retorna 0 mas mantém o grafo (backward ok)
    z = _norm(torch.randn(6, 8, requires_grad=True))
    y = torch.zeros(6, dtype=torch.long)
    loss = triplet_loss(z, y, margin=0.5)
    loss.backward()                      # não deve levantar
    assert loss.item() == 0.0


def test_triplet_bate_com_forca_bruta():
    torch.manual_seed(1)
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    z = _norm(torch.randn(6, 4))
    d = torch.cdist(z, z)
    margin = 0.5
    ref = []
    for i in range(6):
        pos = [j for j in range(6) if y[j] == y[i] and j != i]
        neg = [j for j in range(6) if y[j] != y[i]]
        hp = max(d[i, j].item() for j in pos)
        hn = min(d[i, j].item() for j in neg)
        ref.append(max(0.0, hp - hn + margin))
    assert abs(triplet_loss(z, y, margin).item() - float(np.mean(ref))) < 1e-5


# ---------------------------------------------------------------- KNNDecider (gate)
def test_knn_gate_direcao_anomalia():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 0.1, (40, 8)) + np.eye(8)[0]
    q_clean = rng.normal(0, 0.1, (5, 8)) + np.eye(8)[0]
    q_err = rng.normal(0, 0.1, (5, 8)) + np.eye(8)[1]
    knn = KNNDecider(clean, k=5)
    assert knn.scores(q_err).mean() > knn.scores(q_clean).mean()   # erro mais anômalo
    assert (knn.scores(q_clean) >= 0).all()                        # 1 - cos em [0, 2]
    # mesma DIREÇÃO que o protótipo (não inverte a fusão/limiar)
    proto = PrototypeDecider(fit_prototypes(clean, k=3), 0.0, 0.95)
    assert proto.scores(q_err).mean() > proto.scores(q_clean).mean()


def test_knn_clamp_k_maior_que_refs():
    z = np.random.default_rng(0).normal(size=(3, 8))
    assert KNNDecider(z, k=100)._k == 3            # k limitado ao nº de referências
    assert KNNDecider(z, k=100).scores(z).shape == (3,)


# ---------------------------------------------------------------- KNNCategoryClassifier (E2)
def test_knn_categoria_acerta_classes_separaveis():
    rng = np.random.default_rng(0)
    z_err = np.vstack([rng.normal(0, 0.1, (20, 8)) + np.eye(8)[i] for i in range(1, 5)])
    cats = np.repeat([1, 2, 3, 4], 20)
    clf = KNNCategoryClassifier(z_err, cats, k=5)
    q = np.vstack([rng.normal(0, 0.1, (3, 8)) + np.eye(8)[i] for i in range(1, 5)])
    assert (clf.predict(q) == np.repeat([1, 2, 3, 4], 3)).all()
    sc, classes = clf.class_scores(q)
    assert sc.shape == (12, 4) and classes.tolist() == [1, 2, 3, 4]


def test_knn_categoria_classe_rara_de_um_exemplo():
    # classe 3 tem só 1 ref -> kk=min(k, n_c)=1, sem erro de índice
    z_err = np.vstack([np.eye(8)[1], np.eye(8)[1], np.eye(8)[2], np.eye(8)[3]])
    cats = np.array([1, 1, 2, 3])
    clf = KNNCategoryClassifier(z_err, cats, k=5)
    assert int(clf.predict(np.eye(8)[3:4])[0]) == 3
