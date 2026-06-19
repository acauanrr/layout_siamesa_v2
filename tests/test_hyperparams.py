"""Garantia de hiperparametros (Fase 6): early_stop_metric e synthetic.enabled REALMENTE
respeitados em todos os branches (antes eram ignorados). Usa os embeddings cacheados; pula
se ausentes (precisa de extract_features)."""
from __future__ import annotations

import pytest

from siamese.config import Config
from conftest import have_embeddings, have_val_synth, ROOT, EMB

pytestmark = pytest.mark.skipif(not have_embeddings(),
                                reason="embeddings ausentes — rode scripts/extract_features.py")


def _cfg(tmp_path):
    cfg = Config.load(str(ROOT / "configs" / "default.yaml"))
    cfg.train.epochs = 6
    cfg.train.patience = 6
    cfg.paths.emb_dir = str(EMB)
    cfg.paths.models_dir = str(tmp_path / "m")
    cfg.paths.reports_dir = str(tmp_path / "r")
    return cfg


def test_early_stop_metric_is_respected(tmp_path):
    from siamese.train import train_head
    cfg = _cfg(tmp_path)
    cfg.train.early_stop_metric = "val_ap_aux"
    res = train_head(cfg, device="cpu")
    assert res["sel_name"] == "val_ap_aux"
    # no melhor epoch, o criterio de selecao 'sel' coincide com a metrica pedida
    best = max(res["history"], key=lambda e: e["sel"])
    assert abs(best["sel"] - best["val_ap_aux"]) < 1e-9


@pytest.mark.skipif(not have_val_synth(), reason="val_synth.npz ausente")
def test_synthetic_enabled_gates_synthetic(tmp_path):
    from siamese.train import train_head
    # enabled=False -> sem sonda sintetica -> gate livre de confound indisponivel (NaN)
    off = _cfg(tmp_path / "off"); off.synthetic.enabled = False
    g_off = train_head(off, device="cpu")["best_metrics"]["val_synth_gate"]
    assert g_off != g_off, "synthetic.enabled=False ainda usou sintetico (gate nao deveria existir)"
    # enabled=True -> gate e' um numero real
    on = _cfg(tmp_path / "on"); on.synthetic.enabled = True
    g_on = train_head(on, device="cpu")["best_metrics"]["val_synth_gate"]
    assert g_on == g_on, "synthetic.enabled=True nao produziu o gate livre de confound"
