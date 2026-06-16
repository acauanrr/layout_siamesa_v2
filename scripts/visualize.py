#!/usr/bin/env python
"""Visualiza o modelo funcionando: clusters do espaco aprendido, prototipo e decisao.

Gera em artifacts/reports/:
  embedding_space.png       DINOv2 cru (antes)  vs  z aprendido (depois) — 2D (UMAP)
  decision_space.png        histograma da distancia ao prototipo limpo + limiar; curva PR
  embedding_interactive.html  scatter interativo (hover mostra arquivo, classe, scores)

Uso: python scripts/visualize.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_curve

from siamese.config import Config
from siamese.features import load_embeddings, read_manifest
from siamese.train import load_model
from siamese.evaluate import model_embeddings
from siamese.decision import (fit_prototypes, PrototypeDecider,
                              select_threshold_for_precision, select_threshold_max_f1)

COLORS = {"limpo": "#2ca02c", "erro_real": "#d62728", "erro_sintetico": "#ff7f0e", "prototipo": "#000000"}
OUTCOME_COLORS = {
    "TP_acerto_erro": "#2ca02c",    # erro detectado corretamente (verde)
    "TN_acerto_limpo": "#1f77b4",   # limpo correto (azul)
    "FP_falso_alarme": "#ff7f0e",   # limpo marcado como erro (laranja)
    "FN_erro_perdido": "#d62728",   # erro NAO detectado (vermelho)
}
SYMBOL = {"train": "circle", "test": "diamond"}


def _outcome(fused: np.ndarray, true: np.ndarray, thr: float) -> np.ndarray:
    pred = (fused > thr).astype(int)
    oc = np.empty(len(true), dtype=object)
    oc[(pred == 1) & (true == 1)] = "TP_acerto_erro"
    oc[(pred == 0) & (true == 0)] = "TN_acerto_limpo"
    oc[(pred == 1) & (true == 0)] = "FP_falso_alarme"
    oc[(pred == 0) & (true == 1)] = "FN_erro_perdido"
    return oc


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--target-precisions", default="0.85,0.95",
                    help="lista p/ comparar o tradeoff precisao×recall lado a lado")
    args = ap.parse_args()
    precisions = [float(x) for x in args.target_precisions.split(",")]
    cfg = Config.load(args.config)
    emb_dir = Path(cfg.paths.emb_dir)
    rep = Path(cfg.paths.reports_dir); rep.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    tr = load_embeddings(emb_dir / "train.npz")
    te = load_embeddings(emb_dir / "test.npz")
    va = load_embeddings(emb_dir / "val.npz")
    syn = load_embeddings(emb_dir / "train_synth.npz")
    model = load_model(Path(cfg.paths.models_dir) / "siamese_head.pt", device=device)

    z_tr, aux_tr = model_embeddings(model, tr["emb"], device)
    z_te, aux_te = model_embeddings(model, te["emb"], device)
    z_va, aux_va = model_embeddings(model, va["emb"], device)
    z_sy, aux_sy = model_embeddings(model, syn["emb"], device)

    # prototipo do cluster limpo + decisao
    protos = fit_prototypes(z_tr[tr["label"] == 0], k=cfg.decision.k_prototypes, seed=cfg.seed)
    dec = PrototypeDecider(protos, 0.0, cfg.decision.target_precision)
    fus = LogisticRegression(max_iter=1000).fit(
        np.stack([dec.scores(z_va), aux_va], 1), va["label"])
    fused_va = fus.predict_proba(np.stack([dec.scores(z_va), aux_va], 1))[:, 1]
    # limiar primario = ponto de operacao do config (f1 balanceado por padrao)
    if cfg.decision.objective == "precision":
        thr = select_threshold_for_precision(fused_va, va["label"], cfg.decision.target_precision)[0]
    else:
        thr = select_threshold_max_f1(fused_va, va["label"])[0]

    # ---- monta os grupos de pontos ----
    clean_train_rows = [r for r in read_manifest(Path(cfg.paths.splits_dir) / "train.csv")
                        if int(r["label"]) == 0]

    groups = []  # (z, raw, classe, split, label_texto)
    def add(zz, raw, classe, split, names):
        groups.append({"z": zz, "raw": raw, "classe": classe, "split": split, "names": names})

    m0 = tr["label"] == 0; m1 = tr["label"] == 1
    add(z_tr[m0], tr["emb"][m0], "limpo", "train", [Path(p).name for p in tr["path"][m0]])
    add(z_tr[m1], tr["emb"][m1], "erro_real", "train", [Path(p).name for p in tr["path"][m1]])
    syn_names = [f"synth[{a}] <- {Path(clean_train_rows[p]['path']).stem}"
                 for p, a in zip(syn["parent"], syn["applied"])]
    add(z_sy, syn["emb"], "erro_sintetico", "train", syn_names)
    mt0 = te["label"] == 0; mt1 = te["label"] == 1
    add(z_te[mt0], te["emb"][mt0], "limpo", "test", [Path(p).name for p in te["path"][mt0]])
    add(z_te[mt1], te["emb"][mt1], "erro_real", "test", [Path(p).name for p in te["path"][mt1]])

    Z = np.concatenate([g["z"] for g in groups])
    RAW = np.concatenate([g["raw"] for g in groups])
    classe = np.concatenate([[g["classe"]] * len(g["z"]) for g in groups])
    split = np.concatenate([[g["split"]] * len(g["z"]) for g in groups])
    names = sum([g["names"] for g in groups], [])
    sp = dec.scores(Z)
    fused = fus.predict_proba(np.stack([sp, np.concatenate(
        [aux_tr[m0], aux_tr[m1], aux_sy, aux_te[mt0], aux_te[mt1]])], 1))[:, 1]

    print("Reduzindo z aprendido (UMAP/t-SNE)...")
    emb2_after = reduce_2d(np.concatenate([Z, protos]))
    proto2 = emb2_after[len(Z):]; z2 = emb2_after[:len(Z)]
    print("Reduzindo DINOv2 cru...")
    raw2 = reduce_2d(RAW)

    true = np.where(classe == "limpo", 0, 1)
    outcome = _outcome(fused, true, thr)   # no limiar primario (config)

    _static_embedding(rep, raw2, z2, proto2, classe)
    _static_decision(rep, sp, classe, thr, te, z_te, dec, fus, aux_te)
    _interactive(rep, z2, proto2, classe, split, names, sp, fused, thr)
    _interactive_outcome(rep, z2, proto2, outcome, classe, split, names, sp, fused, thr)
    _static_outcome(rep, z2, proto2, outcome, split)

    # ---- TRADEOFF precisao×recall: varios limiares lado a lado ----
    thr_by_p = {p: select_threshold_for_precision(fused_va, va["label"], p)[0] for p in precisions}
    _tradeoff_static(rep, z2, proto2, split, true, fused, thr_by_p)
    for p, t in thr_by_p.items():
        oc = _outcome(fused, true, t)
        _interactive_outcome(rep, z2, proto2, oc, classe, split, names, sp, fused, t,
                             suffix=f"_p{p:.2f}")

    # resumo no terminal: metrica de test por limiar
    print("\n  Tradeoff (TEST held-out):")
    te_mask = split == "test"
    for p in precisions:
        oc = _outcome(fused, true, thr_by_p[p])[te_mask]
        from collections import Counter
        c = Counter(oc)
        tp, fp, fn = c.get("TP_acerto_erro", 0), c.get("FP_falso_alarme", 0), c.get("FN_erro_perdido", 0)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"    alvo={p:.2f} thr={thr_by_p[p]:.3f}: precisao={prec:.3f} recall={rec:.3f} "
              f"(TP={tp} FP={fp} FN={fn})")
    print(f"\nVisualizacoes em {rep}/:")
    print("  embedding_space.png  decision_space.png  outcome_space.png")
    print("  tradeoff_outcome.png  <- comparacao lado a lado dos limiares")
    print("  embedding_interactive.html  embedding_interactive_outcome.html")
    print("  " + "  ".join(f"embedding_interactive_outcome_p{p:.2f}.html" for p in precisions))


def _static_embedding(rep, raw2, z2, proto2, classe):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    for title, xy, a in [("DINOv2 cru (ANTES) — limpo e erro misturados", raw2, ax[0]),
                         ("z aprendido pela siamesa (DEPOIS) — limpo vira cluster", z2, ax[1])]:
        for c in ["erro_sintetico", "erro_real", "limpo"]:
            mk = classe == c
            a.scatter(xy[mk, 0], xy[mk, 1], s=14, alpha=0.6, c=COLORS[c], label=c, edgecolors="none")
        a.set_title(title); a.set_xticks([]); a.set_yticks([]); a.legend(loc="best", fontsize=8)
    ax[1].scatter(proto2[:, 0], proto2[:, 1], s=400, marker="*", c=COLORS["prototipo"],
                  edgecolors="white", linewidths=1.5, label="prototipo limpo", zorder=5)
    ax[1].legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(rep / "embedding_space.png", dpi=120); plt.close(fig)


def _static_decision(rep, sp, classe, thr, te, z_te, dec, fus, aux_te):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for c in ["limpo", "erro_real", "erro_sintetico"]:
        ax[0].hist(sp[classe == c], bins=30, alpha=0.55, color=COLORS[c], label=c)
    ax[0].set_title("Distancia ao prototipo LIMPO (regra de decisao)")
    ax[0].set_xlabel("score = 1 - cos(z, prototipo)"); ax[0].legend(fontsize=8)
    # curva PR no test (fusao)
    fused_te = fus.predict_proba(np.stack([dec.scores(z_te), aux_te], 1))[:, 1]
    p, r, _ = precision_recall_curve(te["label"], fused_te)
    ax[1].plot(r, p); ax[1].set_title("Precision-Recall (TEST real, fusao)")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(rep / "decision_space.png", dpi=120); plt.close(fig)


def _interactive(rep, z2, proto2, classe, split, names, sp, fused, thr):
    import plotly.graph_objects as go
    fig = go.Figure()
    for c in ["limpo", "erro_real", "erro_sintetico"]:
        mk = classe == c
        custom = np.stack([np.array(names)[mk], split[mk], sp[mk].round(3),
                           fused[mk].round(3)], axis=1)
        fig.add_trace(go.Scatter(
            x=z2[mk, 0], y=z2[mk, 1], mode="markers", name=c,
            marker=dict(size=7, color=COLORS[c], opacity=0.7, line=dict(width=0)),
            customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>split=%{customdata[1]}<br>"
                          "dist_prototipo=%{customdata[2]}<br>p(erro)=%{customdata[3]}<extra>" + c + "</extra>"))
    fig.add_trace(go.Scatter(x=proto2[:, 0], y=proto2[:, 1], mode="markers",
                             name="prototipo limpo", marker=dict(size=20, color="black", symbol="star")))
    fig.update_layout(title=f"Espaco aprendido z (UMAP) — limiar p(erro)={thr:.3f} | "
                            "limpo=verde, erro real=vermelho, sintetico=laranja",
                      width=1100, height=750, hovermode="closest")
    fig.write_html(rep / "embedding_interactive.html")


def _interactive_outcome(rep, z2, proto2, outcome, classe, split, names, sp, fused, thr, suffix=""):
    """Scatter colorido por ACERTO/ERRO do modelo (TP/TN/FP/FN). Simbolo = split.
    Use a legenda (clique) p/ isolar, ex.: 'FN_erro_perdido | test' = erros reais perdidos."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for oc in ["TN_acerto_limpo", "TP_acerto_erro", "FP_falso_alarme", "FN_erro_perdido"]:
        for s in ["train", "test"]:
            mk = (outcome == oc) & (split == s)
            if mk.sum() == 0:
                continue
            custom = np.stack([np.array(names)[mk], classe[mk], sp[mk].round(3),
                               fused[mk].round(3)], axis=1)
            fig.add_trace(go.Scatter(
                x=z2[mk, 0], y=z2[mk, 1], mode="markers", name=f"{oc} | {s}",
                marker=dict(size=11 if s == "test" else 6, color=OUTCOME_COLORS[oc],
                            symbol=SYMBOL[s], opacity=0.85 if s == "test" else 0.45,
                            line=dict(width=1 if s == "test" else 0, color="black")),
                customdata=custom,
                hovertemplate="<b>%{customdata[0]}</b><br>classe=%{customdata[1]}<br>"
                              "dist_prototipo=%{customdata[2]}<br>p(erro)=%{customdata[3]}"
                              f"<extra>{oc} | {s}</extra>"))
    fig.add_trace(go.Scatter(x=proto2[:, 0], y=proto2[:, 1], mode="markers",
                             name="prototipo limpo", marker=dict(size=20, color="black", symbol="star")))
    fig.update_layout(
        title=f"ACERTO/ERRO do modelo (limiar p(erro)={thr:.3f}) — circulo=train (in-sample), "
              "losango=test (held-out). Verde/azul=acerto, laranja=falso-alarme, vermelho=erro perdido",
        width=1150, height=780, hovermode="closest")
    fig.write_html(rep / f"embedding_interactive_outcome{suffix}.html")


def _static_outcome(rep, z2, proto2, outcome, split):
    """PNG: held-out (test) colorido por TP/TN/FP/FN sobre o fundo do train (cinza)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 7))
    tr = split == "train"; teM = split == "test"
    ax.scatter(z2[tr, 0], z2[tr, 1], s=8, c="#dddddd", label="train (fundo)", edgecolors="none")
    for oc in ["TN_acerto_limpo", "TP_acerto_erro", "FP_falso_alarme", "FN_erro_perdido"]:
        mk = teM & (outcome == oc)
        if mk.sum():
            ax.scatter(z2[mk, 0], z2[mk, 1], s=70, c=OUTCOME_COLORS[oc], marker="D",
                       edgecolors="black", linewidths=0.6, label=f"{oc} ({int(mk.sum())})")
    ax.scatter(proto2[:, 0], proto2[:, 1], s=400, marker="*", c="black",
               edgecolors="white", linewidths=1.5, label="prototipo", zorder=5)
    ax.set_title("TEST (held-out) por acerto/erro do modelo — onde estao FP e FN")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(rep / "outcome_space.png", dpi=120); plt.close(fig)


def _tradeoff_static(rep, z2, proto2, split, true, fused, thr_by_p):
    """Painel lado a lado: TEST held-out por TP/TN/FP/FN em cada limiar (precisao-alvo)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import Counter
    ps = sorted(thr_by_p)
    fig, axes = plt.subplots(1, len(ps), figsize=(7.2 * len(ps), 7), squeeze=False)
    tr = split == "train"; teM = split == "test"
    for ax, p in zip(axes[0], ps):
        thr = thr_by_p[p]
        oc = _outcome(fused, true, thr)
        c = Counter(oc[teM])
        tp, fp, fn = c.get("TP_acerto_erro", 0), c.get("FP_falso_alarme", 0), c.get("FN_erro_perdido", 0)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        ax.scatter(z2[tr, 0], z2[tr, 1], s=7, c="#dddddd", edgecolors="none")
        for o in ["TN_acerto_limpo", "TP_acerto_erro", "FP_falso_alarme", "FN_erro_perdido"]:
            mk = teM & (oc == o)
            if mk.sum():
                ax.scatter(z2[mk, 0], z2[mk, 1], s=70, c=OUTCOME_COLORS[o], marker="D",
                           edgecolors="black", linewidths=0.6, label=f"{o} ({int(mk.sum())})")
        ax.scatter(proto2[:, 0], proto2[:, 1], s=350, marker="*", c="black",
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.set_title(f"precisao-alvo {p:.2f}  (thr={thr:.3f})\n"
                     f"TEST: precisao={prec:.2f}  recall={rec:.2f}  | TP={tp} FP={fp} FN={fn}")
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="best", fontsize=8)
    fig.suptitle("Tradeoff precisao×recall — mesmo espaco z, limiares diferentes "
                 "(losango=test held-out, ★=prototipo)", fontsize=12)
    fig.tight_layout(); fig.savefig(rep / "tradeoff_outcome.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
