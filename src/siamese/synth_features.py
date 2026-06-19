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
from .synthetic import inject, SYNTH_TO_CATEGORY, MULTICLASS_SYNTH_TYPES
from .features import read_manifest


def extract_synthetic(
    train_csv: Path | None,
    out_npz: Path,
    backbone: DinoV2Backbone,
    *,
    n_variants: int = 4,
    max_errors_per_image: int = 2,
    seed: int = 0,
    batch_size: int = 16,
    multiclass: bool = True,
    types: list[str] | None = None,
    clean_rows: list[dict] | None = None,
) -> dict:
    """Gera embeddings sinteticos rotulados a partir de imagens LIMPAS.

    Fonte das limpas: `clean_rows` (lista de {'path': ...}, p/ ler de data/processed/) se
    fornecido; senao filtra label==0 de `train_csv` (legado, data/input/).

    multiclass=True (PADRAO): injeta UM unico tipo por imagem (n_errors=1), restrito aos
        tipos com categoria real correspondente (MULTICLASS_SYNTH_TYPES), e grava a
        CATEGORIA real (slug) por amostra -> fonte rotulada multi-classe + anti-confound.
    multiclass=False (legado binario): comportamento antigo (ate `max_errors_per_image`
        tipos, todos os ERROR_TYPES), category='' .
    """
    if clean_rows is not None:
        rows = clean_rows
    else:
        rows = [r for r in read_manifest(train_csv) if int(r["label"]) == 0]  # so imagens limpas
    rng = random.Random(seed)
    pool = types or (MULTICLASS_SYNTH_TYPES if multiclass else None)
    n_err = 1 if multiclass else max_errors_per_image

    embs, parent_idx, applied, category = [], [], [], []
    buf, mbuf, meta = [], [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(backbone.device)
        mask = torch.stack(mbuf)
        feat = backbone(x, mask).cpu().numpy()
        embs.append(feat)
        for pi, ap, cat in meta:
            parent_idx.append(pi)
            applied.append(ap)
            category.append(cat)
        buf.clear(); mbuf.clear(); meta.clear()

    for i, r in enumerate(tqdm(rows, desc="synthetic")):
        img = load_image(r["path"])
        for _ in range(n_variants):
            corr, t_applied = inject(img, rng, n_errors=n_err, types=pool)
            # categoria = tipo primario mapeado p/ slug real ('' se nao mapeia / binario)
            primary = t_applied[0] if t_applied else ""
            cat = SYNTH_TO_CATEGORY.get(primary) or ""
            x, m = backbone.preprocess(corr)   # mascara reflete o aspecto original
            buf.append(x); mbuf.append(m)
            meta.append((i, "+".join(t_applied), cat))
            if len(buf) >= batch_size:
                flush()
    flush()

    emb = np.concatenate(embs, axis=0).astype(np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        emb=emb,
        label=np.ones(len(emb), dtype=np.int64),   # sintetico = erro (gate binario)
        category=np.array(category),                # slug da categoria real ('' se nao rotulado)
        parent=np.array(parent_idx, dtype=np.int64),
        applied=np.array(applied),
    )
    return {"n": len(emb), "out": str(out_npz)}
