"""Gera e cacheia embeddings de ANOMALIAS SINTETICAS a partir das imagens LIMPAS de treino.

Para cada imagem sem-erro do split de treino, cria N variantes corrompidas (mesma
resolucao/device), passa pelo DINOv2 congelado e salva os embeddings com label=1 e o
indice da imagem-mae (para auditoria/pareamento). Esses negativos sao "casados em
confound": diferem da limpa SO pelo conteudo do erro -> forcam o modelo a aprender o erro.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .backbone import DinoV2Backbone, load_image
from .synthetic import inject
from .features import read_manifest


def extract_synthetic(
    train_csv: Path,
    out_npz: Path,
    backbone: DinoV2Backbone,
    *,
    n_variants: int = 4,
    max_errors_per_image: int = 2,
    seed: int = 0,
    batch_size: int = 16,
) -> dict:
    rows = [r for r in read_manifest(train_csv) if int(r["label"]) == 0]  # so imagens limpas
    rng = random.Random(seed)

    embs, parent_idx, applied = [], [], []
    buf, mbuf, meta = [], [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(backbone.device)
        mask = torch.stack(mbuf)
        feat = backbone(x, mask).cpu().numpy()
        embs.append(feat)
        for pi, ap in meta:
            parent_idx.append(pi)
            applied.append(ap)
        buf.clear(); mbuf.clear(); meta.clear()

    for i, r in enumerate(tqdm(rows, desc="synthetic")):
        img = load_image(r["path"])
        for _ in range(n_variants):
            corr, types = inject(img, rng, n_errors=max_errors_per_image)
            x, m = backbone.preprocess(corr)   # mascara reflete o aspecto original
            buf.append(x); mbuf.append(m)
            meta.append((i, "+".join(types)))
            if len(buf) >= batch_size:
                flush()
    flush()

    emb = np.concatenate(embs, axis=0).astype(np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        emb=emb,
        label=np.ones(len(emb), dtype=np.int64),
        parent=np.array(parent_idx, dtype=np.int64),
        applied=np.array(applied),
    )
    return {"n": len(emb), "out": str(out_npz)}
