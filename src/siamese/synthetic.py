"""Injecao de ANOMALIAS SINTETICAS em imagens limpas.

POR QUE ISTO E O NUCLEO ANTI-CONFOUND
-------------------------------------
O dataset real tem um confound quase perfeito: imagens sem-erro sao todas 2076x2152
(um device), enquanto as de erro tem mil resolucoes/fotos/form-factors. Um modelo pode
"detectar erro" so olhando resolucao/aspecto (baseline trivial da ~98% de acuracia).

Para forcar o modelo a aprender o CONTEUDO do erro (e nao o device), geramos pares
casados: pegamos uma imagem LIMPA e injetamos nela um dos 5 tipos de erro, mantendo
EXATAMENTE a mesma resolucao, aspect ratio e origem. Assim o par (limpa, corrompida)
difere SO pelo erro -> todos os confounds geometricos sao constantes. Esta e a versao
"siamesa one-class" correta: comparar uma tela contra a nocao aprendida de "tela limpa".

Tipos (espelham as categorias do problema):
  black_region  : faixa preta nas laterais/topo/baixo (tela dobravel nao expandida)
  empty_space   : regiao grande apagada com a cor de fundo da propria tela
  overlay       : um retalho da imagem colado sobre outra regiao (elementos sobrepostos)
  disorder      : blocos deslocados/desalinhados (layout quebrado)
  cropped       : conteudo cortado + deslocado deixando faixa vazia (elemento cortado)

Cada funcao recebe e devolve um PIL.Image RGB do mesmo tamanho.
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np
from PIL import Image

ERROR_TYPES = ["black_region", "empty_space", "overlay", "disorder", "cropped"]


def _bg_color(img: Image.Image) -> tuple[int, int, int]:
    """Estima a cor de fundo pela mediana das bordas (cantos/margens)."""
    a = np.asarray(img)
    edges = np.concatenate([
        a[:8, :, :].reshape(-1, 3), a[-8:, :, :].reshape(-1, 3),
        a[:, :8, :].reshape(-1, 3), a[:, -8:, :].reshape(-1, 3),
    ], axis=0)
    return tuple(int(v) for v in np.median(edges, axis=0))


def black_region(img: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    side = rng.choice(["left", "right", "top", "bottom", "both_sides"])
    frac = rng.uniform(0.12, 0.32)
    if side in ("left", "right", "both_sides"):
        bw = int(w * frac)
        if side in ("left", "both_sides"):
            a[:, :bw] = 0
        if side in ("right", "both_sides"):
            a[:, w - bw:] = 0
    elif side == "top":
        a[:int(h * frac)] = 0
    else:
        a[h - int(h * frac):] = 0
    return Image.fromarray(a)


def empty_space(img: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    bg = _bg_color(img)
    rh = int(h * rng.uniform(0.20, 0.45))
    rw = int(w * rng.uniform(0.40, 0.95))
    y = rng.randint(0, max(1, h - rh))
    x = rng.randint(0, max(1, w - rw))
    a[y:y + rh, x:x + rw] = bg
    return Image.fromarray(a)


def overlay(img: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    ph = int(h * rng.uniform(0.12, 0.30))
    pw = int(w * rng.uniform(0.25, 0.55))
    sy = rng.randint(0, max(1, h - ph)); sx = rng.randint(0, max(1, w - pw))
    dy = rng.randint(0, max(1, h - ph)); dx = rng.randint(0, max(1, w - pw))
    patch = a[sy:sy + ph, sx:sx + pw].copy()
    # leve translucidez para parecer sobreposicao
    alpha = rng.uniform(0.75, 1.0)
    a[dy:dy + ph, dx:dx + pw] = (alpha * patch + (1 - alpha) * a[dy:dy + ph, dx:dx + pw]).astype(np.uint8)
    return Image.fromarray(a)


def disorder(img: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    bg = _bg_color(img)
    n = rng.randint(2, 4)
    for _ in range(n):
        bh = int(h * rng.uniform(0.10, 0.22))
        y = rng.randint(0, max(1, h - bh))
        shift = int(w * rng.uniform(0.08, 0.25)) * rng.choice([-1, 1])
        band = a[y:y + bh].copy()
        a[y:y + bh] = bg
        if shift >= 0:
            a[y:y + bh, shift:] = band[:, :w - shift]
        else:
            a[y:y + bh, :w + shift] = band[:, -shift:]
    return Image.fromarray(a)


def cropped(img: Image.Image, rng: random.Random) -> Image.Image:
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    bg = _bg_color(img)
    frac = rng.uniform(0.15, 0.35)
    if rng.random() < 0.5:
        ch = int(h * frac)
        a[ch:] = a[:h - ch]
        a[:ch] = bg
    else:
        cw = int(w * frac)
        a[:, cw:] = a[:, :w - cw]
        a[:, :cw] = bg
    return Image.fromarray(a)


_FUNCS: dict[str, Callable] = {
    "black_region": black_region,
    "empty_space": empty_space,
    "overlay": overlay,
    "disorder": disorder,
    "cropped": cropped,
}


def inject(img: Image.Image, rng: random.Random, n_errors: int = 1,
           types: list[str] | None = None) -> tuple[Image.Image, list[str]]:
    """Aplica 1..n erros aleatorios e devolve (imagem_corrompida, tipos_aplicados)."""
    pool = types or ERROR_TYPES
    k = rng.randint(1, max(1, n_errors))
    chosen = rng.sample(pool, k=min(k, len(pool)))
    out = img
    for t in chosen:
        out = _FUNCS[t](out, rng)
    return out, chosen
