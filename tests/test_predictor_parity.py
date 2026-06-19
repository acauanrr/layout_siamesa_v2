"""Paridade Predictor (producao) <-> evaluate (estatico) (Fase 6 / 'paridade de execucao').

Sem backbone/GPU: testa que a MESMA matematica de score e' usada nos dois lados —
(1) score escalar de erro da aux head; (2) formula de fusao logistica == sklearn."""
from __future__ import annotations

import numpy as np

from siamese.evaluate import _aux_err as eval_aux_err
from siamese.infer import Predictor, _sigmoid


class _Stub:
    """Expoe so o atributo que Predictor._aux_err usa, sem instanciar o backbone."""
    def __init__(self, multiclass: bool):
        self.multiclass = multiclass


def test_aux_err_multiclass_parity():
    rng = np.random.default_rng(0)
    aux = rng.normal(size=(6, 7))                  # [N, C]
    ev = eval_aux_err(aux, multiclass=True)        # vetor [N]
    for i in range(aux.shape[0]):
        infer_val = Predictor._aux_err(_Stub(True), aux[i:i + 1])   # codigo REAL de infer.py
        assert abs(infer_val - ev[i]) < 1e-9


def test_aux_err_binary_parity():
    rng = np.random.default_rng(1)
    aux = rng.normal(size=(6,))                    # [N] logits
    ev = eval_aux_err(aux, multiclass=False)
    for i in range(aux.shape[0]):
        infer_val = Predictor._aux_err(_Stub(False), aux[i:i + 1])
        assert abs(infer_val - ev[i]) < 1e-9


def test_fusion_formula_matches_sklearn():
    # evaluate.py salva fus.coef_/intercept_ no bundle; infer.py reaplica
    # sigmoid(coef.[sp,ae]+intercept). Provar que isso == predict_proba (paridade do gate).
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=80) > 0).astype(int)
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    coef = lr.coef_.ravel(); intercept = float(lr.intercept_[0])
    logit = coef[0] * X[:, 0] + coef[1] * X[:, 1] + intercept
    p_infer = _sigmoid(logit)
    p_sklearn = lr.predict_proba(X)[:, 1]
    assert np.allclose(p_infer, p_sklearn, atol=1e-9)
