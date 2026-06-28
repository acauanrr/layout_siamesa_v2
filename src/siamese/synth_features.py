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

from PIL import Image

from .backbone import DinoV2Backbone, load_image
from .synthetic import inject, SYNTH_TO_CATEGORY, MULTICLASS_SYNTH_TYPES
from .reflow import reflow_augment, DEFAULT_REFLOW_WEIGHTS
from .features import read_manifest


def benign_augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Mudanca de APARENCIA inofensiva (round-trip de resolucao + jitter foto-metrico leve).

    Portado do legado (synthetic.benign_augment). Motivo no v2: 47 erros reais sao FOTOS de
    camera (mais baixa resolucao/nitidez) -> o modelo pode usar nitidez como atalho. Aplicar
    o round-trip de resolucao + brilho/contraste/blur leve nas LIMPAS ensina que variacao de
    aparencia/qualidade NAO e' erro. Preserva todo o conteudo (so reamostra/ajusta tom)."""
    import numpy as np
    out = img
    if rng.random() < 0.7:                                   # round-trip de resolucao
        w, h = out.size
        s = rng.uniform(0.45, 1.0)
        small = out.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR)
        out = small.resize((w, h), Image.BILINEAR)
    a = np.asarray(out).astype(np.float32)
    a *= rng.uniform(0.9, 1.1)                               # brilho
    a = (a - a.mean()) * rng.uniform(0.92, 1.08) + a.mean()  # contraste
    out = Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    if rng.random() < 0.3:
        from PIL import ImageFilter
        out = out.filter(ImageFilter.GaussianBlur(radius=1.0))
    return out


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
    p_reflow_pos: float = 0.0,
    reflow_ops: dict | None = None,
    max_reflow_ops: int = 2,
    benign: bool = False,
) -> dict:
    """Gera embeddings sinteticos rotulados a partir de imagens LIMPAS.

    Fonte das limpas: `clean_rows` (lista de {'path': ...}, p/ ler do dataset processado) se
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

    rw = reflow_ops or DEFAULT_REFLOW_WEIGHTS
    for i, r in enumerate(tqdm(rows, desc="synthetic")):
        img = load_image(r["path"])
        for _ in range(n_variants):
            base = img
            # p_reflow_pos (LEGADO/ablacao, default 0): injeta o erro SOBRE um layout reflowado,
            # para que "layout mudou" nao vire atalho de "limpo". reflow ANTES, erro DEPOIS (a
            # regiao morta do erro fica inequivoca sobre o canvas reflowado). No legado foi
            # resultado NEGATIVO -> mantido off por padrao.
            if p_reflow_pos > 0.0 and rng.random() < p_reflow_pos:
                base, _ = reflow_augment(base, rng, ops_weights=rw, max_ops=max_reflow_ops)
            if benign:
                base = benign_augment(base, rng)
            corr, t_applied = inject(base, rng, n_errors=n_err, types=pool)
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


def extract_reflow_clean(
    out_npz: Path,
    backbone: DinoV2Backbone,
    clean_rows: list[dict],
    *,
    n_variants: int = 4,
    reflow_ops: dict | None = None,
    max_reflow_ops: int = 2,
    benign: bool = False,
    seed: int = 0,
    batch_size: int = 16,
) -> dict:
    """Gera embeddings de variantes LIMPAS de REFLOW (label=0, category='clean').

    Para cada imagem limpa de `clean_rows`, cria N variantes com mudancas de LAYOUT LEGITIMAS
    (src/siamese/reflow.py — scroll/dual-pane/aspect/espacamento) e, opcionalmente, benign
    (round-trip de resolucao). Sao HARD NEGATIVES: ensinam o gate que "mesmo conteudo, layout
    diferente = ainda limpo" e que resolucao/aspecto nao-canonicos podem ser limpos (quebra o
    confound pelo lado limpo). Esquema do npz identico aos sinteticos, mas label=0/clean."""
    if n_variants <= 0 or not clean_rows:
        raise ValueError(f"extract_reflow_clean: n_variants={n_variants}, "
                         f"clean_rows={len(clean_rows) if clean_rows else 0} -> nada a gerar.")
    rng = random.Random(seed)
    rw = reflow_ops or DEFAULT_REFLOW_WEIGHTS
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

    for i, r in enumerate(tqdm(clean_rows, desc="reflow-clean")):
        img = load_image(r["path"])
        for _ in range(n_variants):
            out, ops = reflow_augment(img, rng, ops_weights=rw, max_ops=max_reflow_ops)
            if benign:
                out = benign_augment(out, rng)
            x, m = backbone.preprocess(out)   # mascara reflete o NOVO aspecto (ar_relayout)
            buf.append(x); mbuf.append(m)
            meta.append((i, "+".join(ops) if ops else "identity"))
            if len(buf) >= batch_size:
                flush()
    flush()

    emb = np.concatenate(embs, axis=0).astype(np.float32) if embs else np.zeros((0, backbone.out_dim), np.float32)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        emb=emb,
        label=np.zeros(len(emb), dtype=np.int64),       # reflow legitimo = LIMPO
        category=np.array(["clean"] * len(emb)),
        parent=np.array(parent_idx, dtype=np.int64),
        applied=np.array(applied),
    )
    return {"n": len(emb), "out": str(out_npz)}
