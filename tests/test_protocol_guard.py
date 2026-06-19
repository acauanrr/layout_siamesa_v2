"""Blindagem do TESTE (Fase 0 / criterio de aceite #6) + grid shield (Fase 6).

Garante que NENHUM caminho de selecao consiga ler artefatos `test*` sem a trava liberada,
e que importar o grid_search nao destrava nada nem importa evaluate (anti-snooping #2)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from siamese import protocol
from siamese.protocol import (TestSetAccessError, is_test_artifact, guard_path,
                              allow_test_access)
from siamese.protocol import test_access_allowed as _access_allowed  # alias: evita coleta pytest

ROOT = Path(__file__).resolve().parents[1]


def setup_function(_):
    allow_test_access(False)


def teardown_function(_):
    allow_test_access(False)


def test_is_test_artifact_detection():
    assert is_test_artifact("artifacts/embeddings/test.npz")
    assert is_test_artifact("a/b/test_synth.npz")
    assert is_test_artifact("data/processed/test/real/clean/x.png")
    assert is_test_artifact("data/splits/test.csv")
    # nao confundir com nomes parecidos
    assert not is_test_artifact("artifacts/embeddings/val.npz")
    assert not is_test_artifact("artifacts/embeddings/train.npz")
    assert not is_test_artifact("a/latest/model.npz")
    assert not is_test_artifact("a/pytest/x.npz")


def test_guard_blocks_test_when_locked():
    with pytest.raises(TestSetAccessError):
        guard_path("artifacts/embeddings/test.npz")
    with pytest.raises(TestSetAccessError):
        guard_path("artifacts/embeddings/test_synth.npz")


def test_guard_allows_nontest():
    assert guard_path("artifacts/embeddings/val.npz")
    assert guard_path("artifacts/embeddings/train.npz")


def test_unlock_then_allows():
    assert not _access_allowed()
    allow_test_access(True)
    assert _access_allowed()
    assert guard_path("artifacts/embeddings/test.npz")


def test_load_embeddings_is_guarded(tmp_path):
    # bloqueia ANTES de abrir (o arquivo nem precisa existir)
    from siamese.features import load_embeddings
    with pytest.raises(TestSetAccessError):
        load_embeddings(tmp_path / "test.npz")


def _import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grid_shield_import_does_not_unlock_or_import_evaluate():
    allow_test_access(False)
    gs = _import_script("grid_search")
    # importar o grid nao pode destravar o teste...
    assert _access_allowed() is False
    # ...nem trazer evaluate para o escopo de selecao (anti-snooping #2)
    assert not hasattr(gs, "evaluate"), "grid_search nao deve importar evaluate (selecao = so val)"
