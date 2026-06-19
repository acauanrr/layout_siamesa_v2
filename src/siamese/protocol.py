"""Blindagem do conjunto de TESTE (Fase 0 do protocolo anti-vazamento).

O teste e' HELD-OUT: so pode ser lido por UM unico fluxo — `scripts/evaluate.py --final-test`,
executado UMA vez, depois de congelar arquitetura/pesos/limiares. Qualquer outro codigo
(grid_search, ablation, compare_preprocess, visualize, treino) que tente carregar um artefato
de teste (`test.npz`, `test_synth.npz`, `data/processed/test/...`, `test.csv`) levanta
`TestSetAccessError`. Assim a SELECAO de hiperparametros nunca enxerga o teste (anti-snooping,
problema #2 da auditoria) e o teste so e' tocado uma vez (criterio de aceite #6).

Mecanismo: uma trava global (default TRANCADA). `allow_test_access()` destrava — e SO o faz
o caminho `--final-test`. `siamese.features.load_embeddings` chama `guard_path` no chokepoint
por onde TODO modelo consome embeddings, entao basta blindar ali.
"""
from __future__ import annotations

from pathlib import Path

__all__ = [
    "TestSetAccessError", "allow_test_access", "test_access_allowed",
    "is_test_artifact", "guard_path",
]


class TestSetAccessError(RuntimeError):
    """Acesso a um artefato de TESTE sem a trava `--final-test` liberada."""
    __test__ = False   # nao e' uma classe de teste pytest (apesar do prefixo 'Test')


_ALLOW_TEST = False


def allow_test_access(flag: bool = True) -> None:
    """Libera (ou retranca) a leitura de artefatos de teste. SO o fluxo --final-test deve
    chamar isto, e uma unica vez. Tambem util em testes para exercitar os dois estados."""
    global _ALLOW_TEST
    _ALLOW_TEST = bool(flag)


def test_access_allowed() -> bool:
    return _ALLOW_TEST


def is_test_artifact(path) -> bool:
    """True se o caminho aponta para um artefato do split de TESTE.

    Cobre `test.npz`/`test_synth.npz`/`test.csv` (nome comeca com 'test') e qualquer caminho
    com um segmento de diretorio == 'test' (ex.: data/processed/test/real/clean/...). NAO
    confunde com nomes como 'latest'/'pytest' (checa segmento exato e prefixo de nome)."""
    p = Path(path)
    if p.name.lower().startswith("test"):
        return True
    return any(seg.lower() == "test" for seg in p.parts)


def guard_path(path):
    """Levanta TestSetAccessError se `path` for de teste e a trava estiver fechada; senao
    devolve o proprio path (para encadear: `load(guard_path(p))`)."""
    if is_test_artifact(path) and not _ALLOW_TEST:
        raise TestSetAccessError(
            f"Acesso ao TESTE bloqueado: {path}\n"
            "O teste e' held-out (Fase 0 do protocolo). So pode ser lido via "
            "`scripts/evaluate.py --final-test`, UMA vez, apos congelar a config. "
            "Selecao/ajuste de hiperparametros deve usar VALIDACAO ou cross-validation."
        )
    return path
