"""Isolamento de splits (Fase 6 / criterio de aceite #2): nenhum grupo (ticket de erro ou
SESSAO limpa) cruza train/val/test, e cada categoria aparece nos tres splits."""
from __future__ import annotations

import collections
import csv
from pathlib import Path

import pytest

from conftest import ROOT, SPLITS, read_split_csv

PROC = ROOT / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (SPLITS / "all.csv").exists(), reason="data/splits ausente — rode scripts/build_splits.py")


def test_no_group_crosses_split():
    rows = read_split_csv("all.csv")
    g2s = collections.defaultdict(set)
    for r in rows:
        g2s[r["group"]].add(r["split"])
    leaks = {g: sorted(s) for g, s in g2s.items() if len(s) > 1}
    assert not leaks, f"grupos cruzando splits (vazamento): {dict(list(leaks.items())[:5])}"


def test_clean_sessions_are_grouped_not_singletons():
    rows = [r for r in read_split_csv("all.csv") if r["source"] == "no_erros"]
    assert rows, "sem telas limpas no manifesto"
    groups = {r["group"] for r in rows}
    # devem ser grupos de SESSAO (no_erros:sessNNN), nao um grupo por arquivo
    assert all(g.startswith("no_erros:sess") for g in groups), "limpas nao foram reagrupadas por sessao"
    assert len(groups) < len(rows), "cada limpa virou um grupo unitario (vazamento nao corrigido)"
    # cada sessao limpa em exatamente um split
    g2s = collections.defaultdict(set)
    for r in rows:
        g2s[r["group"]].add(r["split"])
    assert all(len(s) == 1 for s in g2s.values())


def test_every_category_present_in_every_split():
    rows = read_split_csv("all.csv")
    by = collections.defaultdict(set)
    for r in rows:
        by[r["category"]].add(r["split"])
    missing = {c: {"train", "val", "test"} - s for c, s in by.items()
               if s != {"train", "val", "test"}}
    assert not missing, f"categorias ausentes em algum split: {missing}"


# --- arvore MATERIALIZADA data/processed/ (o que e' compartilhado com outros times) ---

@pytest.mark.skipif(not (PROC / "train" / "real" / "clean").is_dir(),
                    reason="data/processed ausente — rode scripts/export_processed.py")
def test_processed_tree_clean_sessions_are_split_atomic():
    """Nenhuma SESSAO de captura limpa pode aparecer em mais de um split-dir no disco."""
    from siamese.manifest import clean_session_components
    proc_split = {}
    for sp in ("train", "val", "test"):
        d = PROC / sp / "real" / "clean"
        if d.is_dir():
            for p in d.iterdir():
                if p.suffix.lower() == ".png":
                    proc_split[p.name] = sp
    assert proc_split, "sem telas limpas materializadas em processed/"
    sess = clean_session_components(sorted(proc_split))
    g2s = collections.defaultdict(set)
    for n, sp in proc_split.items():
        g2s[sess[n]].add(sp)
    crossing = {g: sorted(s) for g, s in g2s.items() if len(s) > 1}
    assert not crossing, f"VAZAMENTO: sessoes limpas cruzando split-dirs em processed/: {crossing}"


@pytest.mark.skipif(not (PROC / "manifest.csv").exists(),
                    reason="processed/manifest.csv ausente")
def test_processed_matches_corrected_splits():
    """Cada arquivo REAL em processed/ esta no mesmo split que data/splits/*.csv manda."""
    csv_split = {Path(r["path"]).name: r["split"] for r in read_split_csv("all.csv")}
    with open(PROC / "manifest.csv", newline="") as f:
        man = list(csv.DictReader(f))
    mism = [Path(m["origem"]).name for m in man
            if m["source"] == "real" and csv_split.get(Path(m["origem"]).name, m["split"]) != m["split"]]
    assert not mism, f"{len(mism)} arquivos em processed/ com split != CSV corrigido (stale): {mism[:5]}"
