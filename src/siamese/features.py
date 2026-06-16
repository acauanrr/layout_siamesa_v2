"""Extracao e cache de embeddings DINOv2.

Como o backbone e congelado, os embeddings de cada imagem sao FIXOS. Calcula-los uma
unica vez e cachear em disco torna o treino da cabeca siamesa quase instantaneo e
permite muitas execucoes/experimentos rapidos (o gargalo, o ViT, roda so 1x por imagem).

Cache: artifacts/embeddings/<split>.npz com arrays:
  emb   [N, D] float32   (embeddings do backbone, nao normalizados)
  label [N]    int64
  group [N]    <U...     (ticket, para amostragem sem vazamento)
  path  [N]    <U...
  kind, form_factor, orientation [N] <U...   (para auditoria de confound)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .backbone import DinoV2Backbone, BackboneConfig, load_image
from .geometry import preprocess_image


class _ImageManifestDataset(Dataset):
    """Nao guarda o modelo (evita enviar CUDA aos workers); so os params de preproc."""
    def __init__(self, rows: list[dict], size: int, mode: str, pad_color):
        self.rows = rows
        self.size = size
        self.mode = mode
        self.pad_color = tuple(pad_color)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        img = load_image(self.rows[i]["path"])
        x, mask = preprocess_image(img, self.size, self.mode, self.pad_color)
        return x, mask, i


def read_manifest(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def extract_split(
    manifest_csv: Path,
    out_npz: Path,
    backbone: DinoV2Backbone,
    *,
    batch_size: int = 16,
    num_workers: int = 8,
) -> dict:
    rows = read_manifest(manifest_csv)
    ds = _ImageManifestDataset(rows, backbone.cfg.size, backbone.cfg.preprocess, backbone.cfg.pad_color)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                    shuffle=False, pin_memory=True)

    embs = np.zeros((len(rows), backbone.out_dim), dtype=np.float32)
    for x, mask, idx in tqdm(dl, desc=f"extract {manifest_csv.stem}"):
        feat = backbone(x, mask).cpu().numpy()
        embs[idx.numpy()] = feat

    def col(name):
        return np.array([r[name] for r in rows])

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        emb=embs,
        label=col("label").astype(np.int64),
        group=col("group"),
        path=col("path"),
        kind=col("kind"),
        form_factor=col("form_factor"),
        orientation=col("orientation"),
    )
    return {"n": len(rows), "dim": backbone.out_dim, "out": str(out_npz)}


def load_embeddings(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    return {k: z[k] for k in z.files}
