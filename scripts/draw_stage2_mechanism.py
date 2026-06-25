#!/usr/bin/env python3
"""Estagio 2 — COMO a categoria e calculada (nearest category prototype). Bilingue.

Responde diretamente: usa os MESMOS embeddings z; nao aplica rede/siamesa de novo; nao
usa MLP como decisor (a cabeca aux e' so diagnostico). A separacao dos grupos de erro e'
um classificador de PROTOTIPO MAIS PROXIMO (nearest-centroid) por cosseno no espaco z
(que foi moldado no TREINO por SupCon + CE).

  (a) visao geometrica: z (novo erro) vs 6 protótipos de categoria (= media de classe).
  (b) o calculo: cos(z, protótipo_c) -> arg max -> categoria fina -> mapa fixo 6->3.

    python scripts/draw_stage2_mechanism.py            # PT -> estagio2_prototipo.*
    python scripts/draw_stage2_mechanism.py --lang en  # EN -> stage2_prototype_en.*
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTE = "#1f2430", "#5b6472"
PANEL_BG, PANEL_EC = "#fafbfc", "#d7dbe2"

# 6 categorias finas -> (slug, cor, cos ilustrativo)
CATS = [
    ("black_bars",        "#c0392b", 0.18),
    ("empty_space",       "#e06b5f", 0.10),
    ("overlay",           "#d68a2e", 0.63),   # <- arg max
    ("disordered_layout", "#e7b15a", 0.41),
    ("distortion",        "#2f7e8e", 0.22),
    ("orientation",       "#5aa9b6", 0.15),
]
WIN = 2  # overlay

STR = {
    "pt": {
        "_out": "estagio2_prototipo",
        "title": "Estágio 2 — como a categoria é calculada",
        "subtitle": ("“protótipo de categoria mais próximo” (nearest-centroid) no espaço z — "
                     "reaproveita o MESMO z do Estágio 1, sem rede nova e sem MLP decisor"),
        "pa_tag": "(a)", "pa_sub": "no espaço z: o protótipo mais próximo",
        "nearest": "mais próximo\n→ overlay",
        "proto_cap": "★ protótipo_c = média (re-normalizada) dos z de TREINO da categoria c  ·  k=1 ⇒ 1 protótipo/classe",
        "query": "z  (novo erro)",
        "pb_tag": "(b)", "pb_sub": "o cálculo, passo a passo",
        "step1": "1)  similaridade de cosseno  s_c = cos(z, protótipo_c)   para c = 1..6",
        "maxtag": "★ máx",
        "step2": "2)  arg max  ⇒  categoria fina",
        "step3": "3)  mapa fixo 6→3  ⇒  super-classe", "coarse_badge": "deslocado",
        "infer": "inferência = 1 produto escalar + arg max  ·  sem pesos treinados no Estágio 2  ·  sem k-means (k=1)",
        "callout": ("Reaproveita o MESMO z do Estágio 1  ·  nenhuma rede/siamesa aplicada de novo  ·  "
                    "a cabeça aux (MLP 64→7) é só DIAGNÓSTICO, não decide  ·  separação = nearest-centroid no z do SupCon"),
    },
    "en": {
        "_out": "stage2_prototype_en",
        "title": "Stage 2 — how the category is computed",
        "subtitle": ("“nearest category prototype” (nearest-centroid) in z-space — "
                     "reuses the SAME z from Stage 1, no new network and no MLP decider"),
        "pa_tag": "(a)", "pa_sub": "in z-space: the nearest prototype",
        "nearest": "nearest\n→ overlay",
        "proto_cap": "★ prototype_c = (re-normalized) mean of TRAIN z of category c  ·  k=1 ⇒ 1 prototype/class",
        "query": "z  (new error)",
        "pb_tag": "(b)", "pb_sub": "the computation, step by step",
        "step1": "1)  cosine similarity  s_c = cos(z, prototype_c)   for c = 1..6",
        "maxtag": "★ max",
        "step2": "2)  arg max  ⇒  fine category",
        "step3": "3)  fixed map 6→3  ⇒  super-class", "coarse_badge": "displaced",
        "infer": "inference = 1 dot product + arg max  ·  no trained weights in Stage 2  ·  no k-means (k=1)",
        "callout": ("Reuses the SAME z from Stage 1  ·  no network/siamese applied again  ·  "
                    "the aux head (MLP 64→7) is DIAGNOSTIC only, doesn't decide  ·  separation = nearest-centroid in SupCon's z"),
    },
}


def panel(ax, x, y, w, h, title_main, title_sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=PANEL_BG, ec=PANEL_EC, lw=1.4, zorder=1))
    ax.text(x + 0.05, y + h + 0.27, title_main, fontsize=12.5, fontweight="bold",
            color=MUTE, ha="left", va="center")
    ax.text(x + 1.0, y + h + 0.27, title_sub, fontsize=10.4, color=INK, ha="left", va="center")


def badge(ax, x, y, text, fg, bg, *, fs=9.6, z=8):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=fg,
            zorder=z, bbox=dict(boxstyle="round,pad=0.36", fc=bg, ec=fg, lw=1.2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="pt", choices=["pt", "en"])
    t = STR[ap.parse_args().lang]

    out = Path("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    rng = np.random.default_rng(11)

    fig = plt.figure(figsize=(16.0, 8.4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.42, 8.02, t["title"], fontsize=21, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(0.42, 7.60, t["subtitle"], fontsize=10.8, color=MUTE, ha="left", va="center")

    # (a) GEOMETRICO
    lx, ly, lw, lh = 0.50, 0.95, 7.05, 5.85
    panel(ax, lx, ly, lw, lh, t["pa_tag"], t["pa_sub"])

    protos = [(2.05, 5.55), (3.35, 5.95), (5.85, 4.55), (6.15, 3.05), (2.85, 1.95), (4.45, 1.75)]
    for (pxc, pyc), (slug, color, _cv) in zip(protos, CATS):
        pts = np.column_stack([rng.normal(pxc, 0.34, 7), rng.normal(pyc, 0.30, 7)])
        ax.scatter(pts[:, 0], pts[:, 1], s=34, c=color, edgecolors="white", linewidths=0.4, alpha=0.55, zorder=3)
        ax.scatter([pxc], [pyc], marker="*", s=300, c=color, edgecolors="white", linewidths=1.1, zorder=6)
        ax.text(pxc, pyc - 0.52, slug, ha="center", va="center", fontsize=6.8, color=color,
                fontweight="bold", zorder=7, bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.8))

    qx, qy = 4.55, 3.85
    for i, (pxc, pyc) in enumerate(protos):
        win = i == WIN
        ax.add_patch(FancyArrowPatch((qx, qy), (pxc, pyc), arrowstyle="-",
                     lw=2.4 if win else 0.8, color=CATS[i][1] if win else "#c2c8d2",
                     alpha=1.0 if win else 0.8, zorder=5 if win else 2, shrinkA=8, shrinkB=10))
    ax.scatter([qx], [qy], marker="D", s=150, c="#222831", edgecolors="white", linewidths=1.2, zorder=8)
    ax.text(qx + 0.05, qy - 0.42, t["query"], ha="center", va="center", fontsize=8.4, fontweight="bold",
            color="#222831", zorder=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.annotate(t["nearest"], (protos[WIN][0], protos[WIN][1]),
                (protos[WIN][0] + 0.55, protos[WIN][1] + 1.05), fontsize=8.4, fontweight="bold",
                color=CATS[WIN][1], ha="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=CATS[WIN][1], lw=1.0),
                arrowprops=dict(arrowstyle="-", color=CATS[WIN][1], lw=1.0))

    ax.text(lx + lw / 2, ly + 0.32, t["proto_cap"], ha="center", va="bottom", fontsize=7.8,
            color=MUTE, zorder=7, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

    # (b) CALCULO
    rx, ry, rw, rh = 8.30, 0.95, 7.25, 5.85
    panel(ax, rx, ry, rw, rh, t["pb_tag"], t["pb_sub"])

    ax.text(rx + 0.30, ry + rh - 0.40, t["step1"], fontsize=9.4, fontweight="bold", color=INK, ha="left", va="center")

    bx0, bx1 = rx + 2.95, rx + 6.85
    bw = bx1 - bx0
    top, dy = ry + rh - 0.95, 0.52
    for i, (slug, color, cv) in enumerate(CATS):
        yy = top - i * dy
        win = i == WIN
        ax.text(bx0 - 0.12, yy, slug, ha="right", va="center", fontsize=7.8, color=color, fontweight="bold")
        ax.add_patch(FancyBboxPatch((bx0, yy - 0.135), bw, 0.27, boxstyle="round,pad=0.0,rounding_size=0.05",
                     fc="#eef0f4", ec="none", zorder=2))
        ax.add_patch(FancyBboxPatch((bx0, yy - 0.135), max(bw * cv, 0.06), 0.27,
                     boxstyle="round,pad=0.0,rounding_size=0.05", fc=color,
                     ec=("#1f2430" if win else "none"), lw=1.4 if win else 0, zorder=3))
        val = f"{cv:.2f}   {t['maxtag']}" if win else f"{cv:.2f}"
        ax.text(bx0 + bw * cv + 0.12, yy, val, ha="left", va="center", fontsize=7.8,
                color=("#a35e12" if win else INK), fontweight="bold" if win else "normal")

    y2 = ry + 1.55
    ax.text(rx + 0.30, y2 + 0.40, t["step2"], fontsize=9.4, fontweight="bold", color=INK, ha="left", va="center")
    badge(ax, rx + 4.6, y2 + 0.40, "overlay", "#a35e12", "#fbeede", fs=9.6)
    ax.text(rx + 0.30, y2 - 0.35, t["step3"], fontsize=9.4, fontweight="bold", color=INK, ha="left", va="center")
    badge(ax, rx + 4.6, y2 - 0.35, t["coarse_badge"], "#a35e12", "#f6e4cb", fs=9.6)
    ax.add_patch(FancyArrowPatch((rx + 4.6, y2 + 0.16), (rx + 4.6, y2 - 0.08), arrowstyle="-|>",
                 mutation_scale=12, lw=1.6, color="#8b94a3", zorder=4))

    ax.text(rx + rw / 2, ry + 0.32, t["infer"], ha="center", va="bottom", fontsize=7.8, color=MUTE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

    # rodape callout
    ax.add_patch(FancyBboxPatch((0.50, 0.28), 15.05, 0.44, boxstyle="round,pad=0.0,rounding_size=0.08",
                 fc="#fffaf0", ec="#d9b15e", lw=1.3, zorder=2))
    ax.text(8.02, 0.50, t["callout"], ha="center", va="center", fontsize=8.4,
            color="#7a5a14", fontweight="bold", zorder=3)

    for ext in ("png", "pdf"):
        fig.savefig(out / f"{t['_out']}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print("OK ->", out / f"{t['_out']}.png")


if __name__ == "__main__":
    main()
