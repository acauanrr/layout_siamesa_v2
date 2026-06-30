"""Testes do capture_foldable.py (passo 1 da coleta foldable, Fase 2.b).

Exercita o NÚCLEO (ingest/dedup/manifesto) sem adb — a imagem entra direto como PIL. Garante o
contrato que o merge_clean_extra.py consome: schema do manifesto, convenção de path, form_factor/
orientation populados (destravam o subconjunto controlado), grupo por conteúdo (anti-vazamento),
dedup perceptual.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# carrega o script (não é pacote) como módulo
_SPEC = importlib.util.spec_from_file_location(
    "capture_foldable", Path(__file__).resolve().parents[1] / "scripts" / "capture_foldable.py")
cf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cf)


def _img(w: int, h: int, seed: int) -> Image.Image:
    """Imagem de ruído determinística — dHashes de seeds distintos ficam ~32 bits apart (>6)."""
    a = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(a)


def test_ingest_salva_png_e_linha_do_manifesto(tmp_path):
    out = tmp_path / "clean_extra_fold"
    rows, seen = [], []
    row = cf.ingest(_img(480, 500, 1), out, rows, seen, device="Pixel Fold",
                    app="Settings", screen="WiFi", form_factor="unfold", orientation="portrait")
    assert row is not None
    assert row["w"] == 480 and row["h"] == 500 and abs(row["aspect"] - 0.96) < 0.01
    assert row["form_factor"] == "unfold" and row["orientation"] == "portrait"
    assert row["device"] == "pixel-fold" and row["group"] == "fold:settings:wifi"
    # convenção de path consumida por merge_clean_extra (relativa a data/, prefixada pelo out.name)
    assert row["path"] == f"{out.name}/pixel-fold/{Path(row['path']).name}"
    assert (out / "pixel-fold" / Path(row["path"]).name).exists()
    assert set(row) == set(cf.MANIFEST_COLS)
    # round-trip do manifesto
    cf.write_manifest(out, rows)
    rows2, seen2 = cf.load_manifest(out)
    assert len(rows2) == 1 and int(rows2[0]["phash"]) == row["phash"] and seen2 == [row["phash"]]


def test_dedup_perceptual_ignora_recaptura(tmp_path):
    out = tmp_path / "clean_extra_fold"
    rows, seen = [], []
    img = _img(480, 500, 2)
    assert cf.ingest(img, out, rows, seen, device="d", app="a", screen="s", form_factor="fold")
    # MESMA imagem (mesmo phash) -> duplicata -> None, pool intacto
    assert cf.ingest(img.copy(), out, rows, seen, device="d", app="a", screen="s",
                     form_factor="unfold") is None
    assert len(rows) == 1 and len(seen) == 1


def test_postures_distintas_mesmo_grupo_de_conteudo(tmp_path):
    out = tmp_path / "clean_extra_fold"
    rows, seen = [], []
    a = cf.ingest(_img(500, 480, 3), out, rows, seen, device="d", app="Maps", screen="Home",
                  form_factor="unfold")
    b = cf.ingest(_img(480, 500, 4), out, rows, seen, device="d", app="Maps", screen="Home",
                  form_factor="fold")
    # conteúdo igual (app:screen) -> MESMO grupo -> nunca em splits diferentes (anti-vazamento)
    assert a and b and a["group"] == b["group"] == "fold:maps:home"
    assert len(rows) == 2 and a["form_factor"] != b["form_factor"]


def test_orientation_auto_por_aspecto(tmp_path):
    out = tmp_path / "clean_extra_fold"
    rows, seen = [], []
    land = cf.ingest(_img(800, 400, 5), out, rows, seen, device="d", app="a", screen="s",
                     form_factor="tent")
    port = cf.ingest(_img(400, 800, 6), out, rows, seen, device="d", app="a", screen="s2",
                     form_factor="laptop")
    assert land["orientation"] == "landscape" and port["orientation"] == "portrait"


def test_form_factor_invalido_falha_alto(tmp_path):
    out = tmp_path / "clean_extra_fold"
    with pytest.raises(ValueError):
        cf.ingest(_img(480, 500, 7), out, [], [], device="d", app="a", screen="s",
                  form_factor="tablet")   # nao esta em FORM_FACTORS -> cega o controlado


def test_manifesto_compativel_com_merge(tmp_path):
    """O schema do manifesto deve conter exatamente o que merge_clean_extra lê."""
    needed = {"path", "w", "h", "group", "form_factor", "orientation"}
    assert needed <= set(cf.MANIFEST_COLS)
