#!/usr/bin/env python3
"""Diagrama DETALHADO da decisao em dois estagios (bloco S6 do pipeline). Bilingue.

Foco: responder COMO funciona o Estagio 2 (categoria). Mostra que ambos os estagios
reaproveitam o MESMO vetor z (128-d) ja produzido pela cabeca de projecao g(.):
  - Estagio 1 (gate): fusao por REGRESSAO LOGISTICA de [1-cos(z,prototipo limpo), 1-P(clean)].
  - Estagio 2 (categoria): NEAREST CATEGORY PROTOTYPE por cosseno (protótipo = media do z
    por categoria no treino). Sem rede nova, sem MLP decisor, sem clustering ao vivo.

    python scripts/draw_two_stage_decision.py            # PT  -> decisao_two_stage.*
    python scripts/draw_two_stage_decision.py --lang en  # EN  -> two_stage_decision_en.*
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTE, LINE, Z = "#1f2430", "#5b6472", "#3a4250", "#7e64c0"
COL = {
    "z":      ("#ece6f7", "#7e64c0", "#4d3a85"),
    "s1":     ("#eef4ee", "#6f9f78", "#2f6e3b"),
    "s2":     ("#fbeede", "#d3973f", "#a35e12"),
    "sub":    ("#ffffff", "#b8c0cd", "#3a4250"),
    "fus":    ("#eaf0f7", "#5a86b8", "#234e7d"),
    "thr":    ("#f3eef9", "#9170c4", "#4d3a85"),
    "ok":     ("#e7f1e6", "#4f9a59", "#236233"),
    "err":    ("#fadfe0", "#cf5b62", "#9c2c33"),
    "coarse": ("#ffffff", "#c69b54", "#a35e12"),
    "call":   ("#fffaf0", "#d9b15e", "#7a5a14"),
}

STR = {
    "pt": {
        "_out": "decisao_two_stage",
        "title": "Decisão em dois estágios",
        "subtitle": ("ambos os estágios reaproveitam o MESMO vetor z (128-d, L2-norm) já produzido "
                     "pela cabeça de projeção g(·) — nenhuma rede é aplicada de novo"),
        "z_title": "z · 128-d", "z_body": "L2-norm\n(saída de g(·))", "z_note": "1 vetor por imagem",
        "s1_tag": "ESTÁGIO 1", "s1_sub": "— GATE: “tem erro?”  (binário)",
        "proto_t": "ramo PROTÓTIPO",
        "proto_b": "score_proto = 1 − cos(z, protótipo limpo)\nprotótipo limpo = média dos z\nlimpos de treino (k=1)",
        "aux_t": "ramo AUXILIAR", "aux_b": "aux_err = 1 − P(clean)\ncabeça Linear 128→7 · softmax",
        "fus_t": "FUSÃO",
        "fus_b": "Regressão Logística\n( [score_proto , aux_err] )\n→  p(erro)\ncalibrada na VAL livre de confound",
        "thr_t": "limiar", "thr_b": "max-F1 (padrão) /\nprecisão / especificidade",
        "ok": "✓ limpo", "err": "✗ erro",
        "if_err": "se ✗ erro  (o MESMO z segue para o Estágio 2)",
        "s2_tag": "ESTÁGIO 2", "s2_sub": "— CATEGORIA  (só se ✗ erro)",
        "cp_t": "protótipos de CATEGORIA",
        "cp_b": "6 protótipos =\nmédia do z por categoria\n(treino · k=1 · sem k-means)",
        "as_t": "atribuição de categoria",
        "as_b": "categoria = argmax_c  cos(z, protótipo_c)\n“protótipo mais próximo” (cosseno)\nreaproveita o MESMO z · nearest-centroid",
        "co_t": "6 finas → 3 grossas  (mapa fixo)",
        "co_items": [("região morta", "black_bars · empty_space"),
                     ("deslocado", "overlay · disordered_layout"),
                     ("geometria", "distortion · orientation")],
        "callout": ("Estágio 2 reaproveita o MESMO z  ·  NENHUMA rede/siamesa aplicada de novo  ·  "
                    "NENHUM MLP decisor (a cabeça aux é só diagnóstico)  ·  sem k-means (k=1 = média de classe)"),
        "footer": ("treino (offline): g(·) molda z por SupCon + 0.6·CE; protótipos = médias de classe no z.   "
                   "inferência: 1 produto escalar + arg max — sem pesos treinados no Estágio 2."),
    },
    "en": {
        "_out": "two_stage_decision_en",
        "title": "Two-stage decision",
        "subtitle": ("both stages reuse the SAME z vector (128-d, L2-norm) already produced "
                     "by the projection head g(·) — no network is applied again"),
        "z_title": "z · 128-d", "z_body": "L2-norm\n(output of g(·))", "z_note": "1 vector per image",
        "s1_tag": "STAGE 1", "s1_sub": "— GATE: “is there an error?”  (binary)",
        "proto_t": "PROTOTYPE branch",
        "proto_b": "score_proto = 1 − cos(z, clean prototype)\nclean prototype = mean of clean\ntrain z (k=1)",
        "aux_t": "AUXILIARY branch", "aux_b": "aux_err = 1 − P(clean)\nLinear 128→7 head · softmax",
        "fus_t": "FUSION",
        "fus_b": "Logistic Regression\n( [score_proto , aux_err] )\n→  p(error)\ncalibrated on confound-free VAL",
        "thr_t": "threshold", "thr_b": "max-F1 (default) /\nprecision / specificity",
        "ok": "✓ clean", "err": "✗ error",
        "if_err": "if ✗ error  (the SAME z flows to Stage 2)",
        "s2_tag": "STAGE 2", "s2_sub": "— CATEGORY  (only if ✗ error)",
        "cp_t": "CATEGORY prototypes",
        "cp_b": "6 prototypes =\nmean z per category\n(train · k=1 · no k-means)",
        "as_t": "category assignment",
        "as_b": "category = argmax_c  cos(z, prototype_c)\n“nearest prototype” (cosine)\nreuses the SAME z · nearest-centroid",
        "co_t": "6 fine → 3 coarse  (fixed map)",
        "co_items": [("dead region", "black_bars · empty_space"),
                     ("displaced", "overlay · disordered_layout"),
                     ("geometry", "distortion · orientation")],
        "callout": ("Stage 2 reuses the SAME z  ·  NO network/siamese applied again  ·  "
                    "NO MLP decider (the aux head is diagnostic only)  ·  no k-means (k=1 = class mean)"),
        "footer": ("training (offline): g(·) shapes z via SupCon + 0.6·CE; prototypes = class means in z.   "
                   "inference: 1 dot product + arg max — no trained weights in Stage 2."),
    },
}


def rbox(ax, x, y, w, h, key, *, lw=1.6, z=2):
    fill, edge, _ = COL[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.10",
                 linewidth=lw, edgecolor=edge, facecolor=fill, zorder=z))


def tbox(ax, x, y, w, h, key, title, body=None, *, ts=10.0, bs=8.4, tdy=0.0, lw=1.6, z=2):
    _, _, tc = COL[key]
    rbox(ax, x, y, w, h, key, lw=lw, z=z)
    cx = x + w / 2
    if body:
        ax.text(cx, y + h - 0.26 + tdy, title, ha="center", va="top", fontsize=ts,
                fontweight="bold", color=tc, zorder=z + 1)
        ax.text(cx, y + h - 0.26 - 0.33 + tdy, body, ha="center", va="top", fontsize=bs,
                color=INK, zorder=z + 1, linespacing=1.34)
    else:
        ax.text(cx, y + h / 2 + tdy, title, ha="center", va="center", fontsize=ts,
                fontweight="bold", color=tc, zorder=z + 1, linespacing=1.3)
    return (cx, y + h / 2)


def badge(ax, x, y, text, fg, bg, *, fs=9.4, z=6):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold", color=fg,
            zorder=z, bbox=dict(boxstyle="round,pad=0.34", fc=bg, ec=fg, lw=1.1))


def arrow(ax, p0, p1, *, color=LINE, lw=1.9, ms=15, ls="-", conn=None, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                 color=color, linestyle=ls, zorder=z, shrinkA=2, shrinkB=2,
                 connectionstyle=conn or "arc3,rad=0.0"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="pt", choices=["pt", "en"])
    t = STR[ap.parse_args().lang]

    out = Path("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    fig = plt.figure(figsize=(16.2, 9.1), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 9.1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.42, 8.72, t["title"], fontsize=22, fontweight="bold", color=INK, ha="left", va="center")
    ax.text(0.42, 8.28, t["subtitle"], fontsize=10.8, color=MUTE, ha="left", va="center")

    # z de entrada
    tbox(ax, 0.40, 5.95, 1.85, 1.25, "z", t["z_title"], t["z_body"], ts=11.0, bs=8.4, tdy=0.02, lw=2.0)
    ax.text(1.325, 5.72, t["z_note"], ha="center", va="top", fontsize=7.4, color=MUTE)

    # ESTAGIO 1
    rbox(ax, 2.65, 5.00, 13.15, 2.95, "s1", lw=1.7)
    ax.text(2.92, 7.66, t["s1_tag"], fontsize=12.5, fontweight="bold", color="#2f6e3b", ha="left", va="center")
    ax.text(4.30, 7.66, t["s1_sub"], fontsize=10.6, color=INK, ha="left", va="center")

    tbox(ax, 2.95, 6.20, 3.30, 1.18, "sub", t["proto_t"], t["proto_b"], ts=9.2, bs=7.9)
    tbox(ax, 2.95, 5.18, 3.30, 0.92, "sub", t["aux_t"], t["aux_b"], ts=9.2, bs=7.9, tdy=-0.02)
    tbox(ax, 6.85, 5.55, 3.55, 1.55, "fus", t["fus_t"], t["fus_b"], ts=10.0, bs=8.2, tdy=0.06, lw=1.8)
    tbox(ax, 10.95, 5.78, 2.05, 1.08, "thr", t["thr_t"], t["thr_b"], ts=9.6, bs=7.8)

    badge(ax, 14.30, 6.62, t["ok"], COL["ok"][2], COL["ok"][0])
    badge(ax, 14.30, 5.95, t["err"], COL["err"][2], COL["err"][0])

    arrow(ax, (2.27, 6.55), (2.92, 6.78), color=Z, lw=1.9, conn="arc3,rad=-0.05")
    arrow(ax, (2.27, 6.45), (2.92, 5.64), color=Z, lw=1.9, conn="arc3,rad=0.08")
    arrow(ax, (6.25, 6.78), (6.82, 6.55), conn="arc3,rad=-0.05", lw=1.6, ms=12)
    arrow(ax, (6.25, 5.64), (6.82, 5.95), conn="arc3,rad=0.05", lw=1.6, ms=12)
    arrow(ax, (10.40, 6.33), (10.92, 6.33), lw=1.7, ms=13)
    arrow(ax, (13.00, 6.33), (13.55, 6.40), lw=1.6, ms=12, conn="arc3,rad=-0.08")
    arrow(ax, (13.00, 6.33), (13.55, 6.05), lw=1.6, ms=12, conn="arc3,rad=0.08")

    # ESTAGIO 2
    rbox(ax, 2.65, 1.30, 13.15, 3.10, "s2", lw=1.7)
    ax.text(2.92, 4.10, t["s2_tag"], fontsize=12.5, fontweight="bold", color="#a35e12", ha="left", va="center")
    ax.text(4.30, 4.10, t["s2_sub"], fontsize=10.6, color=INK, ha="left", va="center")

    tbox(ax, 2.95, 2.18, 3.55, 1.45, "sub", t["cp_t"], t["cp_b"], ts=9.4, bs=8.0, tdy=0.04)
    tbox(ax, 7.05, 2.18, 3.55, 1.45, "sub", t["as_t"], t["as_b"], ts=9.6, bs=7.7, tdy=0.04)

    tbox(ax, 11.15, 1.95, 4.45, 1.95, "coarse", t["co_t"], None, ts=9.6, tdy=0.74, lw=1.6)
    for i, (tt, ss) in enumerate(t["co_items"]):
        yy = 3.05 - i * 0.52
        ax.text(11.45, yy, "•", fontsize=12, color="#d3973f", ha="left", va="center")
        ax.text(11.75, yy + 0.07, tt, fontsize=8.8, fontweight="bold", color="#a35e12", ha="left", va="center")
        ax.text(11.75, yy - 0.16, ss, fontsize=6.9, color=MUTE, ha="left", va="center")

    arrow(ax, (6.50, 2.905), (7.02, 2.905), lw=1.7, ms=13)
    arrow(ax, (10.60, 2.905), (11.12, 2.905), lw=1.7, ms=13)

    arrow(ax, (14.30, 5.66), (5.55, 3.66), color=COL["err"][1], lw=2.0, ls=(0, (5, 3)),
          conn="arc3,rad=0.16", z=4)
    ax.text(9.7, 4.66, t["if_err"], fontsize=8.6, color=COL["err"][2], ha="center", va="center", fontstyle="italic")

    # CALLOUT
    rbox(ax, 2.65, 0.52, 13.15, 0.62, "call", lw=1.4)
    ax.text(9.225, 0.83, t["callout"], ha="center", va="center", fontsize=8.6,
            color="#7a5a14", fontweight="bold")
    ax.text(0.42, 0.26, t["footer"], ha="left", va="center", fontsize=7.7, color=MUTE)

    for ext in ("png", "pdf"):
        fig.savefig(out / f"{t['_out']}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print("OK ->", out / f"{t['_out']}.png")


if __name__ == "__main__":
    main()
