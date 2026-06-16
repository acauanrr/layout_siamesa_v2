"""Pre-processamento geometrico: resize anamorfico vs padding cinza + mascara de patch.

Dois modos (config backbone.preprocess):
- "resize": resize anamorfico direto para 518x518 (espreme o aspecto; nao injeta bordas).
- "pad":    padding ate quadrado preservando o aspecto, preenchido com CINZA NEUTRO, e
            so entao resize para 518x518. Preserva a GEOMETRIA real do erro (uma faixa
            preta nao e espremida). O cinza usado e a MEDIA do ImageNet -> vira ~0 apos a
            normalizacao, entao a area de padding quase nao influencia a rede; e distinto
            de preto (erro "black region") e do fundo das telas (erro "empty space").

Para o modo "pad" o risco e a AREA de cinza correlacionar com o aspect-ratio (e logo com
form factor / label). Mitigacao: a mascara de patches de conteudo permite calcular as
estatisticas de patch (mean/std) APENAS na regiao real, ignorando o padding.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Media/desvio ImageNet (mesma do data_config do DINOv2)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Cinza neutro = media ImageNet em 0-255 -> apos Normalize vira ~0 (influencia minima)
IMAGENET_MEAN_255 = tuple(int(round(m * 255)) for m in IMAGENET_MEAN)  # (124, 116, 104)


def pad_to_square(img: Image.Image, fill: tuple[int, int, int]) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if w == h:
        return img
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def content_patch_mask(w: int, h: int, size: int = 518, patch: int = 14,
                       mode: str = "resize") -> np.ndarray:
    """Mascara booleana [N] (N=(size/patch)^2, ordem row-major) dos patches de CONTEUDO.

    No modo "resize" todos os patches sao conteudo (sem padding). No modo "pad", os patches
    cujo centro cai fora da regiao da imagem original (dentro do quadrado) sao padding.
    """
    n = size // patch
    if mode != "pad" or w == h:
        return np.ones(n * n, dtype=bool)
    s = max(w, h)
    cw = size * w / s
    ch = size * h / s
    x0 = (size - cw) / 2.0
    y0 = (size - ch) / 2.0
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):                 # linha (y)
        cy = (i + 0.5) * patch
        inside_y = y0 <= cy <= y0 + ch
        for j in range(n):             # coluna (x)
            cx = (j + 0.5) * patch
            mask[i, j] = inside_y and (x0 <= cx <= x0 + cw)
    return mask.reshape(-1)


def preprocess_image(img: Image.Image, size: int = 518, mode: str = "resize",
                     pad_color: tuple[int, int, int] = IMAGENET_MEAN_255):
    """PIL RGB -> (tensor [3,size,size] normalizado, mascara_de_patch [N] bool)."""
    w, h = img.size
    if mode == "pad":
        img = pad_to_square(img, pad_color)
    img = img.resize((size, size), Image.BICUBIC)
    t = TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
    mask = content_patch_mask(w, h, size=size, mode=mode)
    return t, torch.from_numpy(mask)


def pad_fraction(w: int, h: int) -> float:
    """Fracao da area que seria padding cinza no modo 'pad' (0 = quadrado)."""
    s = max(w, h)
    return 1.0 - (w * h) / (s * s)
