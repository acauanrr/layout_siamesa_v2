"""Augmentacao de REFLOW: mudancas de layout LEGITIMAS aplicadas a telas LIMPAS.

POR QUE ISTO IMPORTA (a tecnica #1 portada do projeto legado, ~/iats/layout_siamesa/)
------------------------------------------------------------------------------------
O detector do v2 e' one-class: "tela limpa" forma um cluster e "tela com erro" cai fora.
Mas a classe limpa vem de UM unico device (2076x2152, telas de onboarding de uma sessao).
Isso cria DOIS problemas, ambos atacados aqui:

  1. FALSO-POSITIVO ESTRUTURAL (DESIGN.md §2.2): o modelo nunca viu "mesmo conteudo, layout
     DIFERENTE = ainda limpo". Logo confunde "tela diferente" com "tela errada" e dispara em
     qualquer app/tela novos -> especificidade 0.12 / FPR 0.88 no held-out.
  2. CONFOUND DE RESOLUCAO: como TODA limpa e' 2076x2152 e os erros sao heterogeneos, a regra
     trivial "resolucao != 2076x2152 => erro" da AUROC 0.99. A sintese de erros (synthetic.py)
     ataca o confound pelo lado do ERRO (injeta erro na MESMA resolucao). O reflow ataca pelo
     lado da CLASSE LIMPA: gera variantes limpas em OUTRAS resolucoes/aspectos -> destroi o
     atalho "resolucao -> erro" dos dois lados.

A ideia (legado, docs/v1_design.md): mudanca de layout LEGITIMA (scroll, recomposicao em 2
paineis, outro aspect-ratio, reflow de espacamento) e' NEGATIVA (label=clean). No legado isso
levou has-bug AUROC 0.62 -> 0.80 e dessaturou o limiar. Aqui as variantes de reflow entram no
treino como exemplos LIMPOS (label 0, category=clean), expandindo o manifold limpo.

PRINCIPIO DE NAO-COLISAO (critico — docs/v1_design.md §2.1)
----------------------------------------------------------
Reflow legitimo MOVE / REESCALA conteudo, preservando-o por inteiro. Bug MATA conteudo
(regiao morta preta/branca/fundo) ou TRUNCA no meio de um elemento. TODOS os operadores
abaixo preservam todo o conteudo (wrap, recomposicao, rescale) justamente para ficarem do
lado certo dessa fronteira — nenhuma variante de reflow pode "parecer um bug". (Inspecione
visualmente com `scripts/dump_synthetic.py --reflow`.)

Estilo: igual a synthetic.py — recebe e devolve PIL.Image RGB; usa numpy + PIL (sem cv2) e o
mesmo `random.Random` do pipeline (determinismo consistente com extract_synthetic).
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np
from PIL import Image

# Operadores conhecidos (espelham o legado). Pesos default replicam configs/v1 do legado.
REFLOW_OPS = ["scroll_shift", "two_pane", "ar_relayout", "band_jitter"]
DEFAULT_REFLOW_WEIGHTS = {
    "scroll_shift": 1.0,   # posicao de scroll diferente (wrap vertical, sem area morta)
    "two_pane": 0.6,       # single -> dual-pane (recomposicao lado a lado, emenda direta)
    "ar_relayout": 1.0,    # render em outro aspect-ratio (muda o canvas -> quebra o confound)
    "band_jitter": 1.0,    # reflow de texto/espacamento (estica/encolhe vaos de whitespace)
}


def _scroll_shift(img: Image.Image, rng: random.Random) -> Image.Image:
    """Posicao de scroll diferente: roll vertical com WRAP (sem area morta)."""
    a = np.asarray(img)
    h = a.shape[0]
    sign = 1 if rng.random() < 0.5 else -1
    shift = sign * int(h * rng.uniform(0.05, 0.25))
    return Image.fromarray(np.roll(a, shift, axis=0))


def _two_pane(img: Image.Image, rng: random.Random) -> Image.Image:
    """single -> dual-pane: metades superior/inferior reescaladas lado a lado, emenda DIRETA
    (sem vao que possa virar faixa preta/branca). Conteudo preservado (reescala, nao morte)."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    if h < 16 or w < 16:
        return img.copy()
    top = Image.fromarray(a[: h // 2])
    bottom = Image.fromarray(a[h // 2:])
    lw = max(8, w // 2)
    left = np.asarray(top.resize((lw, h), Image.BILINEAR))
    right = np.asarray(bottom.resize((w - lw, h), Image.BILINEAR))
    return Image.fromarray(np.concatenate([left, right], axis=1))


def _ar_relayout(img: Image.Image, rng: random.Random) -> Image.Image:
    """Render em outro aspect-ratio (cover <-> tela principal). MUDA as dimensoes do canvas;
    o pre-processamento 'pad' depois renderiza a diferenca de AR exatamente como os erros
    reais de AR distinto sao renderizados (padding cinza + mascara de patch) — sem padding
    interno falso que pudesse virar 'empty space'. Esta e' a operacao que tira a variante
    limpa da resolucao 2076x2152 -> quebra o confound pelo lado limpo."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    if min(h, w) < 64:
        return img.copy()
    f = float(rng.uniform(0.5, 2.0))           # multiplicador no aspecto h/w
    s = float(np.sqrt(f))
    nh = int(np.clip(round(h * s), 64, 2 * h))
    nw = int(np.clip(round(w / s), 64, 2 * w))
    return img.resize((nw, nh), Image.BILINEAR)


def _band_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    """Reflow de texto/espacamento: detecta faixas de WHITESPACE (linhas sem aresta vertical
    e horizontalmente uniformes), estica/encolhe SO esses vaos +-30% e reescala de volta a
    caixa original. Encolher/esticar texto seria distorcao (colisao com bug) — por isso so
    mexe em vaos reais (> 8px) de baixo gradiente."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    if h < 32:
        return img.copy()
    gray = a.astype(np.float32).mean(axis=2)
    grad = np.abs(np.diff(gray, axis=0)).mean(axis=1)          # (h-1,)
    calm = grad < 1.5                                          # sem aresta vertical entre linhas
    # linha de whitespace: sem aresta dos DOIS lados E horizontalmente uniforme (std baixo) ->
    # mantem interiores texturizados/foto de fora (estica-los pareceria distorcao de render).
    quiet = (np.concatenate([[True], calm]) & np.concatenate([calm, [True]])
             & (gray.std(axis=1) < 8.0))
    segs, start = [], 0
    for y in range(1, h + 1):
        if y == h or quiet[y] != quiet[start]:
            segs.append((start, y, bool(quiet[start])))
            start = y
    if not any(q and (y1 - y0) > 8 for y0, y1, q in segs):
        return _scroll_shift(img, rng)                        # sem whitespace usavel
    parts = []
    for y0, y1, q in segs:
        band = a[y0:y1]
        if q and (y1 - y0) > 8:                               # so reescala vaos reais
            nh = max(2, int(round((y1 - y0) * rng.uniform(0.7, 1.3))))
            band = np.asarray(Image.fromarray(band).resize((w, nh), Image.BILINEAR))
        parts.append(band)
    out = np.concatenate(parts, axis=0)
    return Image.fromarray(out).resize((w, h), Image.BILINEAR)


_FUNCS: dict[str, Callable[[Image.Image, random.Random], Image.Image]] = {
    "scroll_shift": _scroll_shift,
    "two_pane": _two_pane,
    "ar_relayout": _ar_relayout,
    "band_jitter": _band_jitter,
}


def _weighted_sample_no_replace(names: list[str], weights: list[float], k: int,
                                rng: random.Random) -> list[str]:
    """Amostra k nomes distintos proporcionalmente aos pesos (sem reposicao)."""
    pool = list(zip(names, weights))
    chosen: list[str] = []
    for _ in range(min(k, len(pool))):
        total = sum(w for _, w in pool)
        if total <= 0:
            break
        r = rng.uniform(0.0, total)
        acc = 0.0
        for i, (n, w) in enumerate(pool):
            acc += w
            if r <= acc:
                chosen.append(n)
                pool.pop(i)
                break
    return chosen


def reflow_augment(img: Image.Image, rng: random.Random, *,
                   ops_weights: dict[str, float] | None = None,
                   max_ops: int = 2) -> tuple[Image.Image, list[str]]:
    """Compoe 1..max_ops mudancas de layout legitimas (ponderadas por `ops_weights`).

    Devolve (imagem_reflowada, ops_aplicadas). `ar_relayout` vai por ULTIMO (muda o canvas).
    A imagem resultante e' um exemplo LIMPO (label 0) — nunca um bug.
    """
    w = ops_weights or DEFAULT_REFLOW_WEIGHTS
    names = [n for n in REFLOW_OPS if w.get(n, 0.0) > 0.0]
    if not names:
        return img.copy(), []
    weights = [float(w[n]) for n in names]
    k = rng.randint(1, max(1, max_ops))
    chosen = _weighted_sample_no_replace(names, weights, k, rng)
    chosen.sort(key=lambda n: n == "ar_relayout")             # mudanca de canvas por ultimo
    out = img
    for name in chosen:
        out = _FUNCS[name](out, rng)
    return out, chosen
