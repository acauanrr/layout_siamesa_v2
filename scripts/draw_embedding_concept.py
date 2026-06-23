#!/usr/bin/env python3
"""Figura conceitual "antes × depois" do espaco de representacao do siamese-ui-error.

Companheira da figura de arquitetura (scripts/draw_architecture_figure.py). Analogo
da figura do "data manifold" do artigo de difusao, porem mostrando O QUE O NOSSO
MODELO FAZ ao espaco de features:

  ANTES  : DINOv2 cru (1152-d) -> limpas e erros sobrepostos (separa ~so por resolucao).
  DEPOIS : z aprendido (128-d, L2-norm) -> limpa colapsa num cluster em torno do
           PROTOTIPO; erros se afastam (3 super-classes); ANEL tracejado = limiar do gate.

Nuvens de pontos ILUSTRATIVAS/deterministicas (seed fixa) — esquematico, nao dados reais
(o scatter de dados reais ja existe em embedding_space.png via scripts/visualize.py).

    python scripts/draw_embedding_concept.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK = "#1f2430"
MUTE = "#5b6472"
PANEL_BG, PANEL_EC = "#fafbfc", "#d7dbe2"

C_CLEAN = "#4f9a59"
C_DEAD = "#d9544d"      # regiao morta (black_bars, empty_space)
C_DISP = "#e0913a"      # deslocado (overlay, disordered_layout)
C_GEOM = "#3f8ea3"      # geometria (distortion, orientation)
C_PROTO_E = "#16532a"
C_RING = "#2f6e3b"

ASPECT = 8.4 / 9.0      # in/unit_y ; usado p/ desenhar o anel "redondo"


def panel(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        fc=PANEL_BG, ec=PANEL_EC, lw=1.4, zorder=1))


def blob(rng, cx, cy, sx, sy, n, *, clip=None):
    p = np.column_stack([rng.normal(cx, sx, n), rng.normal(cy, sy, n)])
    if clip is not None:
        x0, x1, y0, y1 = clip
        p[:, 0] = np.clip(p[:, 0], x0, x1)
        p[:, 1] = np.clip(p[:, 1], y0, y1)
    return p


def scatter(ax, pts, color, *, s=68, z=3):
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c=color, edgecolors="white",
               linewidths=0.5, alpha=0.92, zorder=z)


def main():
    out_dir = Path("artifacts/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    rng = np.random.default_rng(7)

    fig = plt.figure(figsize=(16.0, 8.4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # -------------------------------------------------- cabecalho
    ax.text(0.42, 8.62, "siamese-ui-error", fontsize=22, fontweight="bold",
            color=INK, ha="left", va="center")
    ax.text(0.42, 8.20, "o que o modelo faz — espaço de representação (antes × depois)",
            fontsize=12.0, color=MUTE, ha="left", va="center")

    # legenda (faixa)
    leg = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_CLEAN,
               markeredgecolor="white", markersize=10, label="limpa"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_DEAD,
               markeredgecolor="white", markersize=10, label="erro · região morta"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_DISP,
               markeredgecolor="white", markersize=10, label="erro · deslocado"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_GEOM,
               markeredgecolor="white", markersize=10, label="erro · geometria"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=C_CLEAN,
               markeredgecolor=C_PROTO_E, markersize=16, label="protótipo limpo"),
        Line2D([0], [0], color=C_RING, lw=1.8, ls=(0, (5, 3)), label="limiar (gate)"),
    ]
    ax.legend(handles=leg, loc="upper right", bbox_to_anchor=(0.985, 0.965),
              ncol=3, frameon=True, fontsize=9.2, handletextpad=0.4,
              columnspacing=1.3, borderpad=0.7).set_zorder(20)

    # ================================================== PAINEL ANTES
    lx, ly, lw, lh = 0.50, 1.45, 6.65, 5.55
    panel(ax, lx, ly, lw, lh)
    ax.text(lx + 0.05, ly + lh + 0.30, "ANTES", fontsize=13, fontweight="bold",
            color=MUTE, ha="left", va="center")
    ax.text(lx + 1.15, ly + lh + 0.30, "— DINOv2 cru (1152-d)", fontsize=11,
            color=INK, ha="left", va="center")

    clip_l = (lx + 0.35, lx + lw - 0.35, ly + 0.35, ly + lh - 0.35)
    cxl, cyl = lx + lw / 2, ly + lh / 2 - 0.05
    scatter(ax, blob(rng, cxl, cyl, 1.55, 1.25, 42, clip=clip_l), C_CLEAN)
    scatter(ax, blob(rng, cxl + 0.1, cyl + 0.1, 1.6, 1.3, 26, clip=clip_l), C_DEAD)
    scatter(ax, blob(rng, cxl - 0.1, cyl - 0.05, 1.6, 1.3, 26, clip=clip_l), C_DISP)
    scatter(ax, blob(rng, cxl, cyl - 0.1, 1.55, 1.25, 22, clip=clip_l), C_GEOM)
    ax.text(cxl, ly + 0.30,
            "limpas e erros sobrepostos — o que mais separa é a\n"
            "resolução (o confound), não o conteúdo do erro",
            ha="center", va="bottom", fontsize=8.6, color=MUTE, linespacing=1.4, zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.85))

    # ================================================== seta g(.)
    arr = FancyArrowPatch((7.35, 4.30), (8.75, 4.30), arrowstyle="-|>",
                          mutation_scale=26, lw=3.0, color="#7e64c0", zorder=5)
    ax.add_patch(arr)
    ax.text(8.05, 4.92, "cabeça de projeção  g(·)", ha="center", va="center",
            fontsize=10.0, fontweight="bold", color="#4d3a85")
    ax.text(8.05, 3.72, "SupCon  +  0.6·CE", ha="center", va="center",
            fontsize=9.0, color="#4d3a85", fontstyle="italic")

    # ================================================== PAINEL DEPOIS
    rx, ry, rw, rh = 8.85, 1.45, 6.65, 5.55
    panel(ax, rx, ry, rw, rh)
    ax.text(rx + 0.05, ry + rh + 0.30, "DEPOIS", fontsize=13, fontweight="bold",
            color=MUTE, ha="left", va="center")
    ax.text(rx + 1.30, ry + rh + 0.30, "— z aprendido (128-d · L2-norm)", fontsize=11,
            color=INK, ha="left", va="center")

    clip_r = (rx + 0.35, rx + rw - 0.35, ry + 0.35, ry + rh - 0.35)
    px, py = rx + 1.55, ry + rh / 2 + 0.35           # prototipo / centro do cluster limpo

    # erros se afastam (3 lobos = super-classes)
    scatter(ax, blob(rng, rx + 4.95, ry + 4.35, 0.62, 0.55, 26, clip=clip_r), C_DEAD)
    scatter(ax, blob(rng, rx + 5.30, ry + 2.55, 0.66, 0.58, 26, clip=clip_r), C_DISP)
    scatter(ax, blob(rng, rx + 3.65, ry + 1.15, 0.66, 0.56, 22, clip=clip_r), C_GEOM)
    _lblbg = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.82)
    ax.text(rx + 4.95, ry + 5.05, "região morta", ha="center", fontsize=8.0,
            color=C_DEAD, fontweight="bold", zorder=7, bbox=_lblbg)
    ax.text(rx + 6.05, ry + 2.55, "deslocado", ha="center", fontsize=8.0,
            color="#a35e12", fontweight="bold", zorder=7, bbox=_lblbg)
    ax.text(rx + 3.65, ry + 0.46, "geometria", ha="center", fontsize=8.0,
            color="#1f5e6e", fontweight="bold", zorder=7, bbox=_lblbg)

    # limpa colapsa num cluster apertado em torno do prototipo
    scatter(ax, blob(rng, px, py, 0.50, 0.44, 42, clip=clip_r), C_CLEAN, z=4)

    # anel = limiar (desenhado "redondo" compensando o aspecto)
    rr = 1.28
    ax.add_patch(Ellipse((px, py), 2 * rr, 2 * rr / ASPECT, fill=False,
                         ec=C_RING, lw=1.9, ls=(0, (5, 3)), zorder=5))
    # prototipo
    ax.scatter([px], [py], marker="*", s=460, c=C_CLEAN, edgecolors=C_PROTO_E,
               linewidths=1.4, zorder=8)
    ax.annotate("protótipo limpo", (px, py), (px - 0.15, py + 1.55),
                fontsize=8.6, fontweight="bold", color=C_PROTO_E, ha="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=C_PROTO_E, lw=0.9))
    ax.annotate("anel = limiar\n(1 − cos(z, protótipo))", (px, py - rr),
                (px, py - rr - 0.95), fontsize=8.0, color=C_RING, ha="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color=C_RING, lw=0.9))

    ax.text(rx + rw / 2, ry + 0.30,
            "SupCon aproxima limpas do protótipo e afasta erros · gate: dentro do anel ⇒ ✓ limpo,  fora ⇒ ✗ erro",
            ha="center", va="bottom", fontsize=8.6, color=MUTE, zorder=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.85))

    # -------------------------------------------------- rodape honesto
    ax.text(0.42, 0.42,
            "esquemático/ilustrativo (não é o scatter de dados reais — esse está em "
            "embedding_space.png). A separação real held-out é modesta: sintético livre de "
            "confound AUROC 0.72 — a melhora é real, mas limitada pelos dados (conjunto limpo = 1 device).",
            ha="left", va="center", fontsize=7.8, color=MUTE)

    png = out_dir / "espaco_representacao.png"
    pdf = out_dir / "espaco_representacao.pdf"
    fig.savefig(png, dpi=200, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(f"OK -> {png}")
    print(f"OK -> {pdf}")


if __name__ == "__main__":
    main()
