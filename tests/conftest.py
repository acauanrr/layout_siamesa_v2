"""Helpers comuns aos testes de integridade (Fase 6)."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
EMB = ROOT / "artifacts" / "embeddings"
MODELS = ROOT / "artifacts" / "models"


def read_split_csv(name: str) -> list[dict]:
    with open(SPLITS / name, newline="") as f:
        return list(csv.DictReader(f))


def have_embeddings() -> bool:
    return (EMB / "train.npz").exists() and (EMB / "val.npz").exists()


def have_val_synth() -> bool:
    return (EMB / "val_synth.npz").exists()
