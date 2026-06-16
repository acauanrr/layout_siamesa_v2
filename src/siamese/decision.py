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
