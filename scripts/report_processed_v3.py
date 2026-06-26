#!/usr/bin/env python
"""Relatorio VISUAL do setup principal (processed_v3) -> artifacts/reports/processed_v3/.

Gera, a partir do modelo CONGELADO (artifacts/models/siamese_head.pt + decision.npz):

  clusters_treino.png            cluster do TREINO no espaco aprendido z, por categoria
  clusters_teste.png             cluster do TESTE, com o TREINO em CINZA ao fundo
  confusion_matrix_binaria_treino.png   erro/nao-erro (gate), treino   — acc/prec/rec/F1/AUROC
  confusion_matrix_binaria_teste.png    erro/nao-erro (gate), teste    — acc/prec/rec/F1/AUROC
  confusion_matrix_categoria_treino.png clean + 4 erros (5 classes), treino — acc/F1/AUROC macro
  confusion_matrix_categoria_teste.png  clean + 4 erros (5 classes), teste
  metricas_por_classe.png        precisao / recall / AUROC por classe (clean + 4 erros)
  metricas_por_classe.json       numeros planos
  RELATORIO_processed_v3.md      junta tudo + tabelas

Decisoes (consistentes com a producao):
- Gate erro/nao-erro = fusao [score_proto, aux_err] + limiar, CARREGADOS de decision.npz
  (exatamente a decisao de producao calibrada na val). AUROC = roc_auc do score de fusao.
- Classificador 5-classes = PROTOTIPO mais proximo entre {clean, black_bars, disordered_layout,
  empty_space, overlay} (mesmos protótipos congelados: clean=gate, erros=cat_prototypes do bundle).
  AUROC por classe = one-vs-rest com a similaridade ao protótipo da classe.

Uso: python scripts/report_processed_v3.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import normalize
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, balanced_accuracy_score)

from siamese.protocol import allow_test_access
from siamese.config import Config
from siamese.features import load_embeddings
from siamese.train import load_model
from siamese.evaluate import model_embeddings, _aux_err
from siamese.decision import assign_category
from siamese.manifest import CATEGORIES, ID_TO_CATEGORY, category_id

CAT_COLORS = {"clean": "#2ca02c", "black_bars": "#1f77b4", "disordered_layout": "#9467bd",
              "empty_space": "#ff7f0e", "overlay": "#d62728"}
CATS = list(CATEGORIES)                       # ['clean','black_bars','disordered_layout','empty_space','overlay']
ERR_CATS = CATS[1:]


def _cat_ids(z: dict) -> np.ndarray:
    return np.array([category_id(str(c)) for c in z["category"]], dtype=int)


def reduce_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    Xn = normalize(X)
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                         random_state=seed).fit_transform(Xn)
    except Exception as e:
        print(f"  (umap indisponivel: {e}; usando t-SNE)")
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(Xn)


# ------------------------------------------------------------------ matrizes de confusao
def confusion_binary_png(out: Path, y, pred, score, *, split: str):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y, pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = f1_score(y, pred, zero_division=0)
    auroc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else float("nan")
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.imshow(cm, cmap="Blues")
    mx = cm.max()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=28,
                    color="white" if cm[i, j] > mx / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["sem erro", "erro"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["sem erro", "erro"])
    ax.set_xlabel("predito"); ax.set_ylabel("real")
    ax.set_title(f"Erro vs sem-erro (gate) — {split}\n"
                 f"ACC={acc:.2f}  ·  Precisão={prec:.2f}  ·  Recall={rec:.2f}  ·  "
                 f"F1={f1:.2f}  ·  AUROC={auroc:.2f}", fontsize=10.5)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return dict(acc=acc, precisao=prec, recall=rec, f1=f1, auroc=auroc,
                bAcc=balanced_accuracy_score(y, pred), TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))


def confusion_category_png(out: Path, y, pred, score_mat, *, split: str):
    labels = list(range(len(CATS)))
    cm = confusion_matrix(y, pred, labels=labels)
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, labels=labels, average="macro", zero_division=0)
    # AUROC macro one-vs-rest (so classes presentes no y)
    aurocs = []
    for c in labels:
        yc = (y == c).astype(int)
        if yc.sum() and yc.sum() < len(yc):
            aurocs.append(roc_auc_score(yc, score_mat[:, c]))
    auroc_m = float(np.mean(aurocs)) if aurocs else float("nan")
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.imshow(cm, cmap="Blues")
    mx = cm.max()
    for i in range(len(CATS)):
        for j in range(len(CATS)):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > mx / 2 else "black")
    ax.set_xticks(labels); ax.set_xticklabels(CATS, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(labels); ax.set_yticklabels(CATS, fontsize=9)
    ax.set_xlabel("predito (sistema 2 estágios: gate + categoria)"); ax.set_ylabel("real")
    ax.set_title(f"Categoria: clean + 4 erros — {split}\n"
                 f"Acurácia={acc:.2f}  F1-macro={f1m:.2f}  AUROC-macro={auroc_m:.2f}", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return dict(accuracy=acc, f1_macro=f1m, auroc_macro=auroc_m, confusion=cm.tolist())


# ------------------------------------------------------------------ metricas por classe
def per_class_metrics(y, pred, score_mat):
    labels = list(range(len(CATS)))
    prec = precision_score(y, pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    rows = {}
    for c, name in enumerate(CATS):
        yc = (y == c).astype(int)
        sup = int(yc.sum())
        auroc = roc_auc_score(yc, score_mat[:, c]) if 0 < sup < len(yc) else float("nan")
        acc_ovr = accuracy_score(yc, (pred == c).astype(int))
        rows[name] = dict(precisao=float(prec[c]), recall=float(rec[c]), f1=float(f1[c]),
                          auroc=float(auroc), acuracia_ovr=float(acc_ovr), suporte=sup)
    return rows


def per_class_chart(out: Path, rows: dict, *, split: str):
    names = ERR_CATS                                       # destaque nos 4 erros
    prec = [rows[c]["precisao"] for c in names]
    rec = [rows[c]["recall"] for c in names]
    auroc = [rows[c]["auroc"] for c in names]
    sup = [rows[c]["suporte"] for c in names]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.2))
    b1 = ax.bar(x - w, prec, w, label="precisão", color="#2e7d32")
    b2 = ax.bar(x, rec, w, label="recall (acerto na classe)", color="#f0a030")
    b3 = ax.bar(x + w, auroc, w, label="AUROC (one-vs-rest)", color="#1f77b4")
    for bars in (b1, b2, b3):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(names, sup)], fontsize=10)
    ax.set_ylim(0, 1.1); ax.set_ylabel("métrica")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_title(f"Quão bem o sistema IDENTIFICA cada classe de erro — {split}\n"
                 "precisão = dos que chamou de X, quantos eram X · recall = dos X reais, quantos achou · "
                 "AUROC = ranqueia X acima do resto", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ------------------------------------------------------------------ clusters
def _scatter_categories(ax, xy, cats, *, big=False):
    for c, name in enumerate(CATS):
        m = cats == c
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=46 if big else 30, c=CAT_COLORS[name],
                   edgecolors="white", linewidths=0.5, alpha=0.9, label=name, zorder=3)


def cluster_pngs(out_dir: Path, xy_tr, cat_tr, xy_te, cat_te, xy_ce):
    xy = np.concatenate([xy_tr, xy_te, xy_ce])
    lims = [xy[:, 0].min() - 1, xy[:, 0].max() + 1, xy[:, 1].min() - 1, xy[:, 1].max() + 1]

    def _proto_stars(ax):
        for c, name in enumerate(CATS):
            ax.scatter(xy_ce[c, 0], xy_ce[c, 1], s=360, marker="*", c=CAT_COLORS[name],
                       edgecolors="black", linewidths=1.3, zorder=5)

    def _finish(ax, title):
        ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=12)
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=CAT_COLORS[c],
                          markersize=9, label=c) for c in CATS]
        handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                              markeredgecolor='black', markersize=15, label='protótipo'))
        ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.92)

    # TREINO
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    _scatter_categories(ax, xy_tr, cat_tr, big=False)
    _proto_stars(ax)
    _finish(ax, "Clusters de TREINO no espaço aprendido (z) — por categoria")
    fig.tight_layout(); fig.savefig(out_dir / "clusters_treino.png", dpi=140); plt.close(fig)

    # TESTE (treino em cinza ao fundo)
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.scatter(xy_tr[:, 0], xy_tr[:, 1], s=22, c="#cfd4da", alpha=0.55, linewidths=0,
               zorder=1, label="treino (fundo)")
    _scatter_categories(ax, xy_te, cat_te, big=True)
    _proto_stars(ax)
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor="#cfd4da", markersize=9,
                      label='treino (fundo)')]
    handles += [Line2D([0], [0], marker='o', color='w', markerfacecolor=CAT_COLORS[c],
                       markersize=9, label=f"teste · {c}") for c in CATS]
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
                          markeredgecolor='black', markersize=15, label='protótipo'))
    ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Clusters de TESTE (held-out) — treino em cinza ao fundo", fontsize=12)
    ax.legend(handles=handles, loc="best", fontsize=8.5, framealpha=0.92)
    fig.tight_layout(); fig.savefig(out_dir / "clusters_teste.png", dpi=140); plt.close(fig)


def _cluster_fig(xy_pts, cats, names, xy_ce, *, title, symbol, bg_xy=None, p_err=None, prefix=""):
    """Figura plotly interativa: pontos por categoria + protótipos (★) + (opcional) treino cinza."""
    import plotly.graph_objects as go
    fig = go.Figure()
    if bg_xy is not None:
        fig.add_trace(go.Scatter(
            x=bg_xy[:, 0], y=bg_xy[:, 1], mode="markers", name="treino (fundo)",
            marker=dict(size=6, color="#d9dee5", opacity=0.5, line=dict(width=0)),
            hoverinfo="skip"))
    for c, name in enumerate(CATS):
        m = cats == c
        if not m.any():
            continue
        if p_err is None:
            cd = names[m].reshape(-1, 1)
            ht = "<b>%{customdata[0]}</b><extra>" + name + "</extra>"
        else:
            cd = np.column_stack([names[m], np.round(p_err[m], 3)])
            ht = "<b>%{customdata[0]}</b><br>p(erro)=%{customdata[1]}<extra>" + name + "</extra>"
        fig.add_trace(go.Scatter(
            x=xy_pts[m, 0], y=xy_pts[m, 1], mode="markers",
            name=f"{prefix}{name} ({int(m.sum())})",
            marker=dict(size=12, color=CAT_COLORS[name], symbol=symbol,
                        line=dict(width=1, color="white"), opacity=0.9),
            customdata=cd, hovertemplate=ht))
    for c, name in enumerate(CATS):
        fig.add_trace(go.Scatter(
            x=[xy_ce[c, 0]], y=[xy_ce[c, 1]], mode="markers", showlegend=False,
            marker=dict(size=22, color=CAT_COLORS[name], symbol="star",
                        line=dict(width=1.5, color="black")),
            hovertemplate=f"★ protótipo {name}<extra></extra>"))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(
        title=title, height=660, hovermode="closest", plot_bgcolor="#FAFBFD", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.14, xanchor="center", x=0.5),
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13),
        margin=dict(l=16, r=16, t=56, b=20))
    return fig


def cluster_htmls(out_dir: Path, xy_tr, cat_tr, names_tr, xy_te, cat_te, names_te, p_err_te, xy_ce):
    """clusters_treino.html / clusters_teste.html (plotly): treino=círculo, teste=losango,
    treino em cinza ao fundo do teste; ★ = protótipo de cada categoria. Hover = arquivo + p(erro)."""
    fig_tr = _cluster_fig(
        xy_tr, cat_tr, names_tr, xy_ce, symbol="circle",
        title="Clusters de TREINO no espaço aprendido (z) — por categoria  (★ = protótipo)")
    fig_tr.write_html(out_dir / "clusters_treino.html", include_plotlyjs=True, full_html=True,
                      config={"displaylogo": False, "responsive": True})
    fig_te = _cluster_fig(
        xy_te, cat_te, names_te, xy_ce, symbol="diamond", bg_xy=xy_tr, p_err=p_err_te, prefix="teste · ",
        title="Clusters de TESTE (held-out) — treino em cinza ao fundo  (★ = protótipo)")
    fig_te.write_html(out_dir / "clusters_teste.html", include_plotlyjs=True, full_html=True,
                      config={"displaylogo": False, "responsive": True})


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = ap.parse_args()
    cfg = Config.load(args.config)
    allow_test_access(True)                        # analise post-hoc do modelo congelado
    device = "cuda" if torch.cuda.is_available() else "cpu"

    emb = Path(cfg.paths.emb_dir)
    out_dir = Path(cfg.paths.reports_dir) / "processed_v3"; out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(cfg.paths.models_dir) / "siamese_head.pt", device=device)
    dn = np.load(Path(cfg.paths.models_dir) / "decision.npz", allow_pickle=True)
    protos_clean = dn["prototypes"]; fcoef = dn["fusion_coef"]; fint = float(dn["fusion_intercept"][0])
    thr = float(dn["threshold"][0])
    cat_protos = dn["cat_prototypes"]; cat_proto_ids = dn["cat_proto_ids"]

    # 5 protótipos: clean (gate) + 4 de erro (bundle)
    protos5 = np.concatenate([protos_clean, cat_protos], axis=0)
    ids5 = np.concatenate([np.zeros(len(protos_clean), int), cat_proto_ids]).astype(int)

    tr = load_embeddings(emb / "train.npz"); te = load_embeddings(emb / "test.npz")
    cat_tr, cat_te = _cat_ids(tr), _cat_ids(te)
    ybin_tr, ybin_te = (cat_tr != 0).astype(int), (cat_te != 0).astype(int)

    z_tr, aux_tr = model_embeddings(model, tr["emb"], device)
    z_te, aux_te = model_embeddings(model, te["emb"], device)

    def fused(z, aux):
        zc = normalize(z); sp = 1.0 - (zc @ normalize(protos_clean).T).max(1)
        ae = _aux_err(aux, True)
        return 1.0 / (1.0 + np.exp(-(fcoef[0] * sp + fcoef[1] * ae + fint)))

    fu_tr, fu_te = fused(z_tr, aux_tr), fused(z_te, aux_te)
    pred_bin_tr, pred_bin_te = (fu_tr > thr).astype(int), (fu_te > thr).astype(int)

    # 5-classes = SISTEMA REAL DE 2 ESTAGIOS (clean se o gate disser sem-erro; senao a categoria do
    # protótipo de erro mais proximo). Assim a coluna 'clean' bate EXATAMENTE com a matriz binaria.
    def cat_pred_2stage(z, fu):
        pred = np.zeros(len(z), int)             # 0 = clean (gate disse sem-erro)
        err = fu > thr
        if err.any():
            pred[err] = assign_category(z[err], cat_protos, cat_proto_ids)
        return pred

    # score por classe (similaridade ao protótipo da classe) -> AUROC one-vs-rest, livre de limiar
    def class_scores(z):
        zc = normalize(z); sims = zc @ protos5.T
        return np.column_stack([sims[:, ids5 == c].max(1) for c in range(len(CATS))])

    pred5_tr, sc5_tr = cat_pred_2stage(z_tr, fu_tr), class_scores(z_tr)
    pred5_te, sc5_te = cat_pred_2stage(z_te, fu_te), class_scores(z_te)

    print(f"saida: {out_dir}")
    # --- matrizes binarias (erro/nao-erro) ---
    bin_tr = confusion_binary_png(out_dir / "confusion_matrix_binaria_treino.png",
                                  ybin_tr, pred_bin_tr, fu_tr, split="TREINO (in-sample)")
    bin_te = confusion_binary_png(out_dir / "confusion_matrix_binaria_teste.png",
                                  ybin_te, pred_bin_te, fu_te, split="TESTE (held-out)")
    # --- matrizes por categoria (clean + 4 erros) ---
    cat_tr_m = confusion_category_png(out_dir / "confusion_matrix_categoria_treino.png",
                                      cat_tr, pred5_tr, sc5_tr, split="TREINO (in-sample)")
    cat_te_m = confusion_category_png(out_dir / "confusion_matrix_categoria_teste.png",
                                      cat_te, pred5_te, sc5_te, split="TESTE (held-out)")
    # --- metricas por classe (teste = held-out, vinculante) ---
    pc_te = per_class_metrics(cat_te, pred5_te, sc5_te)
    pc_tr = per_class_metrics(cat_tr, pred5_tr, sc5_tr)
    per_class_chart(out_dir / "metricas_por_classe.png", pc_te, split="TESTE (held-out)")
    # --- clusters (UMAP uma vez; MESMAS coordenadas no PNG estatico e no HTML interativo) ---
    centroids = np.stack([normalize(protos5[ids5 == c]).mean(0) for c in range(len(CATS))])
    Z = np.concatenate([z_tr, z_te, centroids])
    xy = reduce_2d(Z, seed=cfg.seed)
    n_tr, n_te = len(z_tr), len(z_te)
    xy_tr, xy_te, xy_ce = xy[:n_tr], xy[n_tr:n_tr + n_te], xy[n_tr + n_te:]
    cluster_pngs(out_dir, xy_tr, cat_tr, xy_te, cat_te, xy_ce)
    names_tr = np.array([Path(str(p)).name for p in tr["path"]])
    names_te = np.array([Path(str(p)).name for p in te["path"]])
    cluster_htmls(out_dir, xy_tr, cat_tr, names_tr, xy_te, cat_te, names_te, fu_te, xy_ce)

    flat = {
        "n_treino": int(len(cat_tr)), "n_teste": int(len(cat_te)),
        "binaria": {"treino": bin_tr, "teste": bin_te},
        "categoria_5classes": {"treino": cat_tr_m, "teste": cat_te_m},
        "por_classe_teste": pc_te, "por_classe_treino": pc_tr,
        "limiar_gate": thr, "categorias": CATS,
    }
    (out_dir / "metricas_por_classe.json").write_text(json.dumps(flat, indent=2, ensure_ascii=False))
    _write_md(out_dir, flat)
    print("OK — artefatos gerados:")
    for p in sorted(out_dir.iterdir()):
        print("  ", p.name)


def _f(v, n=2):
    try:
        return f"{float(v):.{n}f}"
    except (TypeError, ValueError):
        return "—"


def _write_md(out_dir: Path, flat: dict):
    b_tr, b_te = flat["binaria"]["treino"], flat["binaria"]["teste"]
    c_tr, c_te = flat["categoria_5classes"]["treino"], flat["categoria_5classes"]["teste"]
    pc = flat["por_classe_teste"]
    rows = ""
    for name in CATS:
        v = pc[name]
        rows += (f"| `{name}` | {v['suporte']} | {_f(v['precisao'])} | {_f(v['recall'])} | "
                 f"{_f(v['f1'])} | {_f(v['auroc'])} | {_f(v['acuracia_ovr'])} |\n")
    md = f"""# Relatório visual — `processed_v3` (setup principal)

Gerado por `scripts/report_processed_v3.py` a partir do modelo congelado (treino+teste completos).
Teste = held-out ({flat['n_teste']} imagens); treino = in-sample ({flat['n_treino']}), mostrado só como referência.

## 1. Clusters no espaço aprendido (z)
- `clusters_treino.png` / `clusters_treino.html` — treino por categoria (clean + 4 erros) + protótipos (★).
- `clusters_teste.png` / `clusters_teste.html` — teste por categoria, **treino em cinza ao fundo** + protótipos (★).
- Os `.html` são **interativos** (plotly): zoom/pan, ligar/desligar classes na legenda, e hover com o
  nome do arquivo (e `p(erro)` no teste). Mesmas coordenadas UMAP dos `.png`.

## 2. Erro vs sem-erro (gate) — matriz de confusão
- `confusion_matrix_binaria_treino.png` · `confusion_matrix_binaria_teste.png`

| split | ACC | Precisão | Recall | F1 | AUROC | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---|
| TREINO (in-sample) | {_f(b_tr['acc'])} | {_f(b_tr['precisao'])} | {_f(b_tr['recall'])} | {_f(b_tr['f1'])} | {_f(b_tr['auroc'])} | {b_tr['TP']}/{b_tr['TN']}/{b_tr['FP']}/{b_tr['FN']} |
| **TESTE (held-out)** | **{_f(b_te['acc'])}** | **{_f(b_te['precisao'])}** | {_f(b_te['recall'])} | {_f(b_te['f1'])} | **{_f(b_te['auroc'])}** | {b_te['TP']}/{b_te['TN']}/{b_te['FP']}/{b_te['FN']} |

> O TREINO é ressubstituição (o modelo já viu) → quase perfeito; **o número que vale é o TESTE**.

## 3. Categoria (clean + 4 erros) — matriz de confusão 5×5
- `confusion_matrix_categoria_treino.png` · `confusion_matrix_categoria_teste.png`
- **Sistema fim-a-fim (2 estágios):** prediz `clean` se o gate disser sem-erro; senão atribui a
  categoria do protótipo de erro mais próximo. (Por isso a coluna `clean` aqui **bate** com a
  matriz binária da §2 — esta é aquela, refinada por categoria.)

| split | Acurácia | F1-macro | AUROC-macro |
|---|---:|---:|---:|
| TREINO (in-sample) | {_f(c_tr['accuracy'])} | {_f(c_tr['f1_macro'])} | {_f(c_tr['auroc_macro'])} |
| **TESTE (held-out)** | **{_f(c_te['accuracy'])}** | **{_f(c_te['f1_macro'])}** | **{_f(c_te['auroc_macro'])}** |

## 4. Quão bem identifica CADA classe (teste held-out)
- `metricas_por_classe.png`

| classe | n | precisão | recall (acerto) | F1 | AUROC | acurácia one-vs-rest |
|---|---:|---:|---:|---:|---:|---:|
{rows}
> **Leitura por classe:** *precisão* = dos que o sistema chamou de X, quantos eram X; *recall* = dos
> X reais, quantos o sistema identificou como X **fim-a-fim** (passar no gate **e** acertar a
> categoria) — é o "quão bom em identificar X"; *AUROC* = quão bem a proximidade ao protótipo de X
> ranqueia os X acima do resto (independe de limiar); *acurácia one-vs-rest* = acerto no problema
> binário "é X ou não" (alta e pouco informativa porque a maioria não é X — use recall/AUROC).

*Números planos: `metricas_por_classe.json`.*
"""
    (out_dir / "RELATORIO_processed_v3.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
