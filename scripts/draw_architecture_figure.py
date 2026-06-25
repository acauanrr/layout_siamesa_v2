#!/usr/bin/env python3
"""Diagrama de arquitetura "visual breakdown" do siamese-ui-error.

Inspirado no estilo das figuras de arquitetura do artigo de difusao (blocos limpos,
codigo de cor, motivo CONGELADO/TREINADO, fluxo esquerda->direita), porem aplicado ao
NOSSO modelo: rede siamesa (cabeca de projecao compartilhada) sobre DINOv2 congelado,
decisao em 2 estagios. Gera PNG + PDF prontos para apresentacao.

    python scripts/draw_architecture_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

# ------------------------------------------------------------------ paleta
INK = "#1f2430"          # texto principal
MUTE = "#5b6472"         # texto secundario
LINE = "#3a4250"         # setas
Z = "#7e64c0"           # roxo do vetor z

COL = {
    "clean":   ("#e7f1e6", "#4f9a59", "#2f6e3b"),   # fill, edge, title
    "real":    ("#fadfe0", "#cf5b62", "#9c2c33"),
    "synth":   ("#fdecd2", "#e0973a", "#a35e12"),
    "reflow":  ("#d9eeed", "#3fa3a0", "#176d6a"),
    "pre":     ("#eef0f4", "#9aa3b2", "#444c5a"),
    "frozen":  ("#dde9f7", "#3f76b8", "#22517f"),   # backbone congelado (azul frio)
    "trained": ("#fde6cd", "#e08a36", "#a4540f"),   # cabeca treinada (ambar quente)
    "zvec":    ("#ece6f7", "#7e64c0", "#4d3a85"),
    "stage":   ("#f6f7f9", "#7c8696", "#3a4250"),
    "sub":     ("#ffffff", "#b8c0cd", "#3a4250"),
    "ok":      ("#e7f1e6", "#4f9a59", "#236233"),
    "err":     ("#fadfe0", "#cf5b62", "#9c2c33"),
    "coarse":  ("#eef0f4", "#8b94a3", "#3a4250"),
}


def rbox(ax, x, y, w, h, key, *, lw=1.6, alpha=1.0, z=2, ls="-"):
    fill, edge, _ = COL[key]
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.10",
        linewidth=lw, edgecolor=edge, facecolor=fill, alpha=alpha,
        linestyle=ls, zorder=z, mutation_aspect=1.0,
    )
    ax.add_patch(p)
    return (x + w / 2, y + h / 2)


def title_body(ax, x, y, w, h, key, title, body=None, *, ts=10.5, bs=8.6,
               title_dy=0.0, lw=1.6, alpha=1.0, z=2):
    _, _, tcol = COL[key]
    rbox(ax, x, y, w, h, key, lw=lw, alpha=alpha, z=z)
    cx = x + w / 2
    if body:
        ax.text(cx, y + h - 0.27 + title_dy, title, ha="center", va="top",
                fontsize=ts, fontweight="bold", color=tcol, zorder=z + 1)
        ax.text(cx, y + h - 0.27 - 0.345 + title_dy, body, ha="center", va="top",
                fontsize=bs, color=INK, zorder=z + 1, linespacing=1.34)
    else:
        ax.text(cx, y + h / 2 + title_dy, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=tcol, zorder=z + 1,
                linespacing=1.3)
    return (cx, y + h / 2)


def badge(ax, x, y, text, fg, bg, *, fs=7.8, z=6):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold",
            color=fg, zorder=z,
            bbox=dict(boxstyle="round,pad=0.30", fc=bg, ec=fg, lw=1.0))


def arrow(ax, p0, p1, *, color=LINE, lw=1.9, ms=15, ls="-", conn=None, z=3):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
        linestyle=ls, zorder=z, shrinkA=2, shrinkB=2,
        connectionstyle=conn or "arc3,rad=0.0"))


def draw_flame(ax, x, y, s=0.20, z=9):
    outer = [(0, 0.95), (0.32, 0.46), (0.20, 0.06), (0.40, -0.34), (0.13, -0.30),
             (0, -0.55), (-0.13, -0.30), (-0.40, -0.34), (-0.20, 0.06), (-0.32, 0.46)]
    inner = [(0, 0.48), (0.17, 0.10), (0.09, -0.18), (0, -0.33), (-0.09, -0.18), (-0.17, 0.10)]
    ax.add_patch(Polygon([(x + px * s, y + py * s) for px, py in outer], closed=True,
                 fc="#ef7d22", ec="#b4530c", lw=0.8, zorder=z))
    ax.add_patch(Polygon([(x + px * s, y + py * s) for px, py in inner], closed=True,
                 fc="#ffd24d", ec="none", zorder=z + 1))


def state_tab(ax, x, y, kind, block_top):
    """Selo CONGELADO/TREINADO flutuando como aba acima do bloco."""
    ax.plot([x, x], [block_top, y - 0.215], color=COL[kind][1], lw=1.1, zorder=4)
    ax.add_patch(Circle((x, y), 0.215, fc="white", ec=COL[kind][1], lw=1.5, zorder=8))
    if kind == "frozen":
        ax.text(x, y - 0.012, "❄", ha="center", va="center", fontsize=13,
                color=COL["frozen"][2], zorder=9)
    else:
        draw_flame(ax, x, y, 0.205)


def main():
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig = plt.figure(figsize=(16.2, 9.1), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 9.1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---------------------------------------------------------- cabecalho
    ax.text(0.42, 8.74, "siamese-ui-error", fontsize=23, fontweight="bold", color=INK,
            ha="left", va="center")
    ax.text(0.42, 8.30,
            "detecção de erro de layout em UI  ·  rede siamesa sobre DINOv2 congelado",
            fontsize=11.0, color=MUTE, ha="left", va="center")

    # legenda CONGELADO / TREINADO (topo direita)
    badge(ax, 13.30, 8.64, "❄  congelado", COL["frozen"][2], COL["frozen"][0], fs=8.2)
    draw_flame(ax, 14.62, 8.64, 0.16)
    badge(ax, 15.18, 8.64, "treinado", COL["trained"][2], COL["trained"][0], fs=8.2)

    # ====================================================== 1) ENTRADAS (treino)
    ax.text(1.475, 7.74, "ENTRADAS  (treino)", fontsize=10.0, fontweight="bold",
            color=MUTE, ha="center", va="center")
    inputs = [
        ("clean",  "telas limpas",      "1 device · 2076×2152"),
        ("real",   "erros reais",        "6 categorias · 369 imgs"),
        ("synth",  "erros sintéticos",   "anti-confound (lado erro)"),
        ("reflow", "limpas-reflow",      "anti-confound (lado limpo)"),
    ]
    iy, ih, gap = 7.30, 0.78, 0.20
    in_anchor = []
    for key, t, sub in inputs:
        cy = iy - ih / 2
        rbox(ax, 0.40, iy - ih, 2.15, ih, key, lw=1.5)
        ax.text(1.475, cy + 0.16, t, ha="center", va="center", fontsize=9.3,
                fontweight="bold", color=COL[key][2])
        ax.text(1.475, cy - 0.175, sub, ha="center", va="center", fontsize=7.4, color=INK)
        in_anchor.append((2.55, cy))
        iy -= ih + gap

    ax.text(1.475, 3.36,
            "o confound: toda limpa vem de 1 device →\n"
            "regra trivial “resolução ≠ 2076×2152 ⇒ erro”\n"
            "já dá AUROC 0.98 (vê o device, não o erro).\n"
            "sintético + reflow forçam aprender o ERRO.",
            ha="center", va="top", fontsize=8.0, color=MUTE, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.5", fc="#fbfbfd", ec="#d7dbe2", lw=1.0))

    # ====================================================== 2) PRE + BACKBONE
    title_body(ax, 2.95, 4.45, 1.62, 1.32, "pre", "pré-processo",
               "padding CINZA\n518×518 (+ máscara)\n37×37 patches", ts=9.8, bs=7.9)
    for (ax0, ay0) in in_anchor:
        arrow(ax, (ax0, ay0), (2.92, 5.10), color="#9aa3b2", lw=1.3, ms=10,
              conn="arc3,rad=0.05", z=1)

    dcx = 5.825
    title_body(ax, 4.85, 4.30, 1.95, 1.66, "frozen", "DINOv2 ViT-S/14",
               "extrator de features\nsaída: CLS + média/desvio\ndos patches → 1152-d",
               ts=11.2, bs=8.3, title_dy=0.10, lw=2.0)
    state_tab(ax, dcx, 6.55, "frozen", 5.96)
    arrow(ax, (4.57, 5.10), (4.83, 5.10))

    hcx = 8.10
    title_body(ax, 7.05, 4.30, 2.10, 1.66, "trained", "cabeça de projeção  g(·)",
               "pesos compartilhados (siamesa)\nLinear 1152→256 → GELU → 64\ntreinada",
               ts=10.0, bs=8.1, title_dy=0.10, lw=2.0)
    state_tab(ax, hcx, 6.55, "trained", 5.96)
    arrow(ax, (6.80, 5.10), (7.03, 5.10))

    # z vector + perda
    title_body(ax, 7.10, 3.05, 2.00, 0.80, "zvec", "z  ·  64-d", "vetor L2-normalizado",
               ts=10.0, bs=8.0, title_dy=-0.02)
    arrow(ax, (hcx, 4.28), (hcx, 3.87), lw=1.6, ms=12)
    ax.text(hcx, 2.50, "perda = SupCon(z)  +  0.3 · CE(aux, 7 classes)",
            ha="center", va="center", fontsize=8.3, color=INK, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f4f1fb", ec=COL["zvec"][1], lw=1.1))

    # z -> estagios (uma origem, leque)
    arrow(ax, (9.12, 3.55), (9.53, 6.70), color=Z, lw=2.1, conn="arc3,rad=-0.16", z=4)
    arrow(ax, (9.12, 3.30), (9.53, 2.55), color=Z, lw=2.1, ls=(0, (5, 3)),
          conn="arc3,rad=0.16", z=4)
    ax.text(9.66, 4.30, "z", fontsize=10, color=COL["zvec"][2], fontweight="bold",
            ha="left", va="center")
    ax.text(9.30, 2.02, "se ERRO", fontsize=7.6, color=COL["zvec"][2],
            ha="center", va="center", fontstyle="italic")

    # ====================================================== 3) ESTAGIO 1
    s1x, s1y, s1w, s1h = 9.55, 4.78, 6.40, 3.55
    rbox(ax, s1x, s1y, s1w, s1h, "stage", lw=1.6)
    ax.text(s1x + 0.28, s1y + s1h - 0.30, "ESTÁGIO 1", fontsize=12.0, fontweight="bold",
            color="#2f6e3b", ha="left", va="center")
    ax.text(s1x + 1.78, s1y + s1h - 0.30, "— gate:  “tem erro?”", fontsize=10.6,
            color=INK, ha="left", va="center")

    # dois ramos (empilhados)
    title_body(ax, 9.85, 6.78, 2.55, 1.05, "sub", "protótipo LIMPO",
               "1 − cos(z, protótipo limpo)\n← clustering", ts=9.2, bs=8.0)
    title_body(ax, 9.85, 5.50, 2.55, 1.05, "sub", "cabeça auxiliar",
               "Linear 64→7 softmax\nP(erro) = 1 − P(clean)", ts=9.2, bs=8.0)
    # fork z -> ramos
    ax.plot([9.70, 9.70], [6.025, 7.305], color=Z, lw=1.6, zorder=4)
    arrow(ax, (9.53, 6.70), (9.70, 6.70), color=Z, lw=1.6, ms=1, z=4)
    arrow(ax, (9.70, 7.305), (9.85, 7.305), color=Z, lw=1.5, ms=11, z=4)
    arrow(ax, (9.70, 6.025), (9.85, 6.025), color=Z, lw=1.5, ms=11, z=4)

    # fusao
    title_body(ax, 12.78, 6.00, 2.95, 1.15, "coarse",
               "fusão calibrada (val livre de confound)",
               "→ p(erro) → limiar\n(balanceado / precisão / especificidade)",
               ts=9.2, bs=8.0, title_dy=0.05)
    arrow(ax, (12.40, 7.305), (12.75, 6.85), lw=1.5, ms=11, conn="arc3,rad=0.06")
    arrow(ax, (12.40, 6.025), (12.75, 6.30), lw=1.5, ms=11, conn="arc3,rad=-0.06")

    # saidas
    arrow(ax, (14.25, 5.98), (14.25, 5.62), lw=1.6, ms=12)
    badge(ax, 13.58, 5.32, "✓ limpo", COL["ok"][2], COL["ok"][0], fs=9.2)
    badge(ax, 14.95, 5.32, "✗ erro", COL["err"][2], COL["err"][0], fs=9.2)

    # ====================================================== 4) ESTAGIO 2
    s2x, s2y, s2w, s2h = 9.55, 0.70, 6.40, 3.30
    rbox(ax, s2x, s2y, s2w, s2h, "stage", lw=1.6)
    ax.text(s2x + 0.28, s2y + s2h - 0.30, "ESTÁGIO 2", fontsize=12.0, fontweight="bold",
            color="#a35e12", ha="left", va="center")
    ax.text(s2x + 1.78, s2y + s2h - 0.30, "— categoria  (só se E1 = erro)", fontsize=10.6,
            color=INK, ha="left", va="center")

    title_body(ax, 9.85, 1.74, 3.05, 1.06, "sub", "protótipo de CATEGORIA",
               "mais próximo (canônico)", ts=9.4, bs=8.2, title_dy=0.02)

    coarse = [
        ("região morta", "black_bars · empty_space"),
        ("deslocado", "overlay · disordered_layout"),
        ("geometria", "distortion · orientation"),
    ]
    cw, ch, cx0 = 2.40, 0.82, 13.30
    for i, (t, sub) in enumerate(coarse):
        cy = 2.95 - i * 0.92
        rbox(ax, cx0, cy - ch / 2, cw, ch, "coarse", lw=1.4)
        ax.text(cx0 + cw / 2, cy + 0.14, t, ha="center", va="center", fontsize=8.9,
                fontweight="bold", color="#a35e12")
        ax.text(cx0 + cw / 2, cy - 0.17, sub, ha="center", va="center", fontsize=6.8,
                color=MUTE)
        arrow(ax, (12.90, 2.27), (cx0 - 0.02, cy), color="#b8a06a", lw=1.2, ms=9,
              conn="arc3,rad=0.0", z=3)

    ax.text(s2x + 0.30, s2y + 0.36,
            "taxonomia GROSSA (3 super-classes) = primária   ·   fina (6 classes) = secundária",
            ha="left", va="center", fontsize=7.8, color=MUTE, fontstyle="italic")

    # ---------------------------------------------------------- rodape honesto
    ax.text(0.42, 0.28,
            "held-out honesto:  sintético livre de confound AUROC 0.71 (AP 0.89)  ·  "
            "subconjunto controlado 0.69 (> confound 0.38)  ·  "
            "global ainda NÃO vence a resolução trivial (0.99).   "
            "Alavanca decisiva = telas limpas diversas, não tuning.",
            ha="left", va="center", fontsize=7.8, color=MUTE)

    png = out_dir / "arquitetura_modelo.png"
    pdf = out_dir / "arquitetura_modelo.pdf"
    fig.savefig(png, dpi=200, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(f"OK -> {png}")
    print(f"OK -> {pdf}")


if __name__ == "__main__":
    main()
