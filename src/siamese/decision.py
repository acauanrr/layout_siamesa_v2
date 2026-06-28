"""Regra de decisao por PROTOTIPO/CLUSTER (a ideia do usuario, no espaco aprendido).

Depois que a cabeca siamesa reformata o espaco, as telas LIMPAS formam cluster(es)
compacto(s). Resumimos o cluster limpo em k prototipos (k-means; k=1 ja basta quando
limpo e unimodal, k>1 da robustez). O score de anomalia de uma imagem = distancia cosseno
ao prototipo limpo mais proximo. Decisao: erro se score > limiar.

O limiar e escolhido na VALIDACAO REAL para atingir uma PRECISAO-ALVO (ex.: 0.95),
porque o gerente pediu ALTA PRECISAO. Reportamos o recall obtido nesse ponto.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


@dataclass
class PrototypeDecider:
    prototypes: np.ndarray   # [k, d] L2-normalizados
    threshold: float
    target_precision: float

    def scores(self, z: np.ndarray) -> np.ndarray:
        """Distancia cosseno ao prototipo limpo mais proximo (maior = mais anomalo)."""
        zc = normalize(z)
        sims = zc @ self.prototypes.T          # [N, k]
        return 1.0 - sims.max(axis=1)          # 1 - cos do prototipo mais proximo

    def predict(self, z: np.ndarray) -> np.ndarray:
        return (self.scores(z) > self.threshold).astype(int)


def fit_prototypes(z_clean: np.ndarray, k: int = 1, seed: int = 0) -> np.ndarray:
    """Calcula k prototipos do cluster limpo (k-means sobre embeddings L2-normalizados)."""
    zc = normalize(z_clean)
    if k <= 1 or len(zc) <= k:
        proto = zc.mean(axis=0, keepdims=True)
        return normalize(proto)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(zc)
    return normalize(km.cluster_centers_)


def fit_category_prototypes(z_err: np.ndarray, cat_ids_err: np.ndarray, *,
                            k: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """ESTAGIO 2: protótipos POR categoria de erro (k-means dentro de cada categoria).

    z_err: embeddings projetados de imagens de ERRO; cat_ids_err: id de categoria (>=1)
    de cada uma. Retorna (protos [M, d] L2-normalizados, proto_cat [M] = id de cada proto).
    Categorias raras com poucas amostras caem para k=1 automaticamente (media).
    """
    cat_ids_err = np.asarray(cat_ids_err)
    protos, proto_cat = [], []
    for c in sorted({int(x) for x in cat_ids_err}):
        zc = z_err[cat_ids_err == c]
        if len(zc) == 0:
            continue
        p = fit_prototypes(zc, k=min(k, len(zc)), seed=seed)
        protos.append(p)
        proto_cat.extend([c] * len(p))
    return np.concatenate(protos, axis=0), np.array(proto_cat, dtype=int)


def assign_category(z: np.ndarray, protos: np.ndarray, proto_cat: np.ndarray) -> np.ndarray:
    """Atribui cada z à categoria do protótipo de erro mais próximo (cosseno). Retorna [N] ids."""
    zc = normalize(z)
    sims = zc @ protos.T
    return proto_cat[sims.argmax(axis=1)]


# ---------------------------------------------------------------------------------------------
# DECISORES k-NN (não-paramétricos) — alternativa aos protótipos de k-means para manifolds
# multimodais. A classe "limpa" é multimodal (onboarding/home/settings de apps distintos): um
# centroide de k-means pode cair no "vazio" entre modos e gerar falso-positivo; o k-NN mede a
# proximidade às limpas REAIS de treino, adaptando-se a qualquer geometria. Ativado por
# decision.gate_method='knn' / stage2_method='knn'. Custo desprezível (~poucas centenas de refs).
# REFERÊNCIA = só embeddings de TREINO (anti-vazamento, idêntico ao protótipo).
# RESULTADO EMPÍRICO (docs/COMPARACAO_KNN_TRIPLET.md): no gate NÃO supera o protótipo (até reduz a
# especificidade — o cluster limpo de teste é mesmo-device); no Estágio 2 dá ganho modesto na
# categoria coarse (0.626->0.641). Default permanece 'prototype'.
# ---------------------------------------------------------------------------------------------
@dataclass
class KNNDecider:
    """Gate one-class por k-NN: score de anomalia = 1 − média da similaridade-cosseno aos k
    vizinhos LIMPOS de treino mais próximos. Mesma semântica/interface de PrototypeDecider.scores
    (maior = mais anômalo), então entra direto na fusão/limiar do evaluate.py."""
    z_clean: np.ndarray            # [M, d] embeddings LIMPOS de treino (referência)
    k: int = 5
    threshold: float = 0.0

    def __post_init__(self):
        self._ref = normalize(self.z_clean)
        self._k = int(min(self.k, len(self._ref)))

    def scores(self, z: np.ndarray) -> np.ndarray:
        sims = normalize(z) @ self._ref.T                  # [N, M] cos-sim
        topk = np.partition(sims, -self._k, axis=1)[:, -self._k:]   # k MAIORES sims por linha
        return 1.0 - topk.mean(axis=1)                     # 1 − média top-k = distância média

    def predict(self, z: np.ndarray) -> np.ndarray:
        return (self.scores(z) > self.threshold).astype(int)


@dataclass
class KNNCategoryClassifier:
    """ESTÁGIO 2 por k-NN: para cada categoria c, score_c(z) = média das top-k similaridades aos
    embeddings de ERRO de treino da classe c. Robusto a desbalanceamento de classe (cada classe é
    pontuada no seu próprio espaço, ao contrário da votação majoritária global por bincount, que
    favoreceria classes mais frequentes). predict = argmax_c score_c."""
    z_err: np.ndarray              # [M, d] embeddings de ERRO de treino
    cat_ids: np.ndarray            # [M] id de categoria (>=1) de cada um
    k: int = 5

    def __post_init__(self):
        self._ref = normalize(self.z_err)
        self.classes = np.array(sorted({int(c) for c in self.cat_ids}), dtype=int)

    def class_scores(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(scores [N, C], classes [C]) — média das top-k sims aos refs de cada classe."""
        zc = normalize(z)
        out = np.zeros((len(z), len(self.classes)), dtype=float)
        for j, c in enumerate(self.classes):
            sims_c = zc @ self._ref[self.cat_ids == c].T   # [N, n_c]
            kk = min(self.k, sims_c.shape[1])
            out[:, j] = np.partition(sims_c, -kk, axis=1)[:, -kk:].mean(axis=1)
        return out, self.classes

    def predict(self, z: np.ndarray) -> np.ndarray:
        sc, classes = self.class_scores(z)
        return classes[sc.argmax(axis=1)]


def select_threshold_max_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    """Limiar que MAXIMIZA o F1 (ponto de operacao BALANCEADO, justo para comparacao).

    Varre os scores como pontos de corte e devolve o que maximiza F1 = 2PR/(P+R).
    """
    order = np.argsort(-scores)
    s_sorted = scores[order]
    y_sorted = labels[order]
    P = int(labels.sum())
    tp = fp = 0
    best = (-1.0, float(scores.max()) + 1e-9, 0.0, 0.0)  # (f1, thr, prec, rec)
    for i in range(len(s_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        if i + 1 < len(s_sorted) and s_sorted[i + 1] == s_sorted[i]:
            continue
        precision = tp / (tp + fp)
        recall = tp / P if P > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        if f1 > best[0]:
            best = (f1, s_sorted[i], precision, recall)
    f1, thr, prec, rec = best
    return float(thr - 1e-9), {"val_f1": float(f1), "val_precision": float(prec),
                               "val_recall": float(rec)}


def select_threshold_for_specificity(
    scores: np.ndarray, labels: np.ndarray, target_specificity: float = 0.80,
) -> tuple[float, dict]:
    """MAIOR recall cuja ESPECIFICIDADE (TN/(TN+FP)) no conjunto de calibracao >= alvo.

    O gerente quer POUCOS falso-alarmes: fixamos especificidade-alvo e maximizamos o recall
    sob essa restricao. Varre cada score como ponto de corte (preve erro se score > thr).
    Se nenhum atinge o alvo, devolve o de maior especificidade. (Complementa o best-F1 e o
    precision-target; util quando os negativos sao confiaveis — ex.: calibracao livre de
    confound com muitas limpas/reflow.)"""
    order = np.argsort(-scores)
    s_sorted = scores[order]
    y_sorted = labels[order]
    P = int(labels.sum())
    N = int((labels == 0).sum())
    best = None       # (recall, thr, specificity)
    best_spec = None  # fallback: (specificity, recall, thr)
    tp = fp = 0
    for i in range(len(s_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        if i + 1 < len(s_sorted) and s_sorted[i + 1] == s_sorted[i]:
            continue
        recall = tp / P if P else 0.0
        spec = (N - fp) / N if N else 1.0
        if best_spec is None or spec > best_spec[0]:
            best_spec = (spec, recall, s_sorted[i])
        if spec >= target_specificity and (best is None or recall > best[0]):
            best = (recall, s_sorted[i], spec)
    if best is None:
        spec, recall, thr = best_spec
        return float(thr - 1e-9), {"val_recall": float(recall), "val_specificity": float(spec),
                                   "target_specificity": float(target_specificity)}
    recall, thr, spec = best
    return float(thr - 1e-9), {"val_recall": float(recall), "val_specificity": float(spec),
                               "target_specificity": float(target_specificity)}


def select_threshold_for_precision(
    scores: np.ndarray, labels: np.ndarray, target_precision: float = 0.95,
) -> tuple[float, dict]:
    """Menor limiar (=maior recall) cuja precisao em (scores,labels) >= alvo.

    Se nenhum limiar atinge o alvo, devolve o de maior precisao. Devolve tambem
    metricas no ponto escolhido.
    """
    order = np.argsort(-scores)  # do mais anomalo ao menos
    s_sorted = scores[order]
    y_sorted = labels[order]
    P = int(labels.sum())

    best = None  # (recall, threshold, precision)
    tp = fp = 0
    # candidatos = cada valor de score como ponto de corte (prediz erro se score >= s)
    for i in range(len(s_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        # so avalia no fim de empates de score
        if i + 1 < len(s_sorted) and s_sorted[i + 1] == s_sorted[i]:
            continue
        precision = tp / (tp + fp)
        recall = tp / P if P > 0 else 0.0
        thr = s_sorted[i]
        if precision >= target_precision:
            if best is None or recall > best[0]:
                best = (recall, thr, precision)

    if best is None:
        # fallback: ponto de precisao maxima
        tp = fp = 0
        best_prec = -1.0
        for i in range(len(s_sorted)):
            if y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            if i + 1 < len(s_sorted) and s_sorted[i + 1] == s_sorted[i]:
                continue
            precision = tp / (tp + fp)
            if precision > best_prec:
                best_prec = precision
                best = (tp / P if P > 0 else 0.0, s_sorted[i], precision)

    recall, thr, precision = best
    # limiar estritamente '>' : usa um epsilon abaixo do score de corte
    eps = 1e-9
    return float(thr - eps), {
        "val_precision": float(precision),
        "val_recall": float(recall),
        "target_precision": float(target_precision),
    }
