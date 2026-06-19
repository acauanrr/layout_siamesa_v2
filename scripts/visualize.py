#!/usr/bin/env python
"""Visualiza o modelo funcionando: clusters do espaco aprendido, prototipo e decisao.

PRINCIPAL (para a reuniao):
  clusters_apresentacao.html  ANTES x DEPOIS do treino, lado a lado, interativo e
                              autoexplicativo (com roteiro do que falar). E o arquivo
                              a ser apresentado: mostra o PROPOSITO da clusterizacao.

Apoio (figuras estaticas do relatorio):
  embedding_space.png       DINOv2 cru (antes)  vs  z aprendido (depois) — 2D (UMAP)
  decision_space.png        histograma da distancia ao prototipo limpo + limiar; curva PR
  outcome_space.png         TEST por TP/TN/FP/FN; tradeoff_outcome.png limiares lado a lado

Aprofundamento (opcional, --extra-html): embedding_interactive*.html por acerto/erro.

Uso: python scripts/visualize.py --config configs/default.yaml
     python scripts/visualize.py --config configs/default.yaml --extra-html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_recall_curve, roc_auc_score

from siamese.config import Config
from siamese.features import load_embeddings, read_manifest
from siamese.train import load_model
from siamese.evaluate import model_embeddings, _aux_err
from siamese.decision import (fit_prototypes, PrototypeDecider,
                              select_threshold_for_precision, select_threshold_max_f1,
                              fit_category_prototypes)
from siamese.manifest import CATEGORY_TO_ID, ID_TO_CATEGORY, CATEGORIES

COLORS = {"limpo": "#2ca02c", "erro_real": "#d62728", "erro_sintetico": "#ff7f0e", "prototipo": "#000000"}
# paleta por CATEGORIA (multi-cluster): clean + 6 categorias de erro
CATEGORY_COLORS = {
    "clean": "#2ca02c", "black_bars": "#1f77b4", "disordered_layout": "#9467bd",
    "distortion": "#8c564b", "empty_space": "#ff7f0e", "orientation": "#e377c2",
    "overlay": "#d62728",
}
OUTCOME_COLORS = {
    "TP_acerto_erro": "#2ca02c",    # erro detectado corretamente (verde)
    "TN_acerto_limpo": "#1f77b4",   # limpo correto (azul)
    "FP_falso_alarme": "#ff7f0e",   # limpo marcado como erro (laranja)
    "FN_erro_perdido": "#d62728",   # erro NAO detectado (vermelho)
}
SYMBOL = {"train": "circle", "test": "diamond"}
# rotulos exibidos (EN) — as CHAVES acima permanecem internas (indexam cores/logica)
CLASS_LABEL = {"limpo": "Clean", "erro_real": "Real error", "erro_sintetico": "Synthetic error"}
OUTCOME_LABEL = {
    "TP_acerto_erro": "TP · error detected",
    "TN_acerto_limpo": "TN · clean correct",
    "FP_falso_alarme": "FP · false alarm",
    "FN_erro_perdido": "FN · missed error",
}


def _outcome(fused: np.ndarray, true: np.ndarray, thr: float) -> np.ndarray:
    pred = (fused > thr).astype(int)
    oc = np.empty(len(true), dtype=object)
    oc[(pred == 1) & (true == 1)] = "TP_acerto_erro"
    oc[(pred == 0) & (true == 0)] = "TN_acerto_limpo"
    oc[(pred == 1) & (true == 0)] = "FP_falso_alarme"
    oc[(pred == 0) & (true == 1)] = "FN_erro_perdido"
    return oc


def _centroid_auroc(X: np.ndarray, true: np.ndarray) -> float:
    """AUROC de 'distancia ao centro das telas limpas' como detector de erro.

    Mede, de forma objetiva e fiel a regra de decisao (distancia ao prototipo),
    o quao bem o espaco separa erro de nao-erro SO pela distancia. E o numero que
    resume o proposito da clusterizacao: deve subir do espaco cru (ANTES) para o
    espaco z aprendido (DEPOIS)."""
    Xn = normalize(X)
    c = Xn[true == 0].mean(0)
    c = c / (np.linalg.norm(c) + 1e-9)
    score = 1.0 - Xn @ c            # distancia (1 - cos) ao centro do limpo
    return float(roc_auc_score(true, score))


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
    ap.add_argument("--extra-html", action="store_true",
                    help="gera tambem os HTML de acerto/erro (TP/TN/FP/FN) por limiar")
    ap.add_argument("--final-test", action="store_true",
                    help="visualiza o TESTE held-out (destrava; uma vez). Sem a flag, usa a VAL.")
    args = ap.parse_args()
    precisions = [float(x) for x in args.target_precisions.split(",")]
    cfg = Config.load(args.config)
    emb_dir = Path(cfg.paths.emb_dir)
    rep = Path(cfg.paths.reports_dir); rep.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    tr = load_embeddings(emb_dir / "train.npz")
    va = load_embeddings(emb_dir / "val.npz")
    # Fase 0: o teste e' blindado. Sem --final-test, visualiza a VAL no lugar do teste.
    if args.final_test:
        from siamese.protocol import allow_test_access
        allow_test_access(True)
        te = load_embeddings(emb_dir / "test.npz")
    else:
        print("[DEV] sem --final-test: visualizando a VAL no lugar do TESTE (teste blindado).")
        te = va
    syn = load_embeddings(emb_dir / "train_synth.npz")
    model = load_model(Path(cfg.paths.models_dir) / "siamese_head.pt", device=device)
    multiclass = getattr(model, "num_classes", 1) > 1

    z_tr, aux_tr = model_embeddings(model, tr["emb"], device)
    z_te, aux_te = model_embeddings(model, te["emb"], device)
    z_va, aux_va = model_embeddings(model, va["emb"], device)
    z_sy, aux_sy = model_embeddings(model, syn["emb"], device)
    # score escalar de erro da aux (binario: logit; multi-classe: 1 - P(clean))
    ae_tr, ae_te, ae_va, ae_sy = (_aux_err(aux_tr, multiclass), _aux_err(aux_te, multiclass),
                                  _aux_err(aux_va, multiclass), _aux_err(aux_sy, multiclass))

    # prototipo do cluster limpo + decisao (gate / Estagio 1)
    protos = fit_prototypes(z_tr[tr["label"] == 0], k=cfg.decision.k_prototypes, seed=cfg.seed)
    dec = PrototypeDecider(protos, 0.0, cfg.decision.target_precision)
    fus = LogisticRegression(max_iter=1000).fit(
        np.stack([dec.scores(z_va), ae_va], 1), va["label"])
    fused_va = fus.predict_proba(np.stack([dec.scores(z_va), ae_va], 1))[:, 1]
    # limiar primario = ponto de operacao do config (f1 balanceado por padrao)
    if cfg.decision.objective == "precision":
        thr = select_threshold_for_precision(fused_va, va["label"], cfg.decision.target_precision)[0]
    else:
        thr = select_threshold_max_f1(fused_va, va["label"])[0]

    # ---- monta os grupos de pontos ----
    groups = []  # (z, raw, classe, split, label_texto)
    def add(zz, raw, classe, split, names):
        groups.append({"z": zz, "raw": raw, "classe": classe, "split": split, "names": names})

    m0 = tr["label"] == 0; m1 = tr["label"] == 1
    add(z_tr[m0], tr["emb"][m0], "limpo", "train", [Path(p).name for p in tr["path"][m0]])
    add(z_tr[m1], tr["emb"][m1], "erro_real", "train", [Path(p).name for p in tr["path"][m1]])
    # 'parent' agora e' o stem da imagem-mae (string), 'applied' o tipo sintetico
    syn_names = [f"synth[{a}] <- {p}" for p, a in zip(syn["parent"], syn["applied"])]
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
    ae_all = np.concatenate([ae_tr[m0], ae_tr[m1], ae_sy, ae_te[mt0], ae_te[mt1]])
    fused = fus.predict_proba(np.stack([sp, ae_all], 1))[:, 1]

    # categoria por ponto (alinhada a Z): clean / categoria real / categoria sintetica
    categoria = np.concatenate([
        np.array(["clean"] * int(m0.sum())),
        tr["category"][m1].astype(str),
        syn["category"].astype(str),
        np.array(["clean"] * int(mt0.sum())),
        te["category"][mt1].astype(str),
    ])
    # prototipos POR categoria (Estagio 2) — fit sobre os erros REAIS de treino
    cat_protos = cat_proto_ids = cat_proto2 = None
    if multiclass:
        cat_ids_tr = np.array([CATEGORY_TO_ID.get(str(c), 0) for c in tr["category"]])
        err = tr["label"] == 1
        cat_protos, cat_proto_ids = fit_category_prototypes(
            z_tr[err], cat_ids_tr[err], k=cfg.decision.k_prototypes, seed=cfg.seed)

    print("Reduzindo z aprendido (UMAP/t-SNE)...")
    extras = [protos] + ([cat_protos] if cat_protos is not None else [])
    emb2_after = reduce_2d(np.concatenate([Z] + extras))
    z2 = emb2_after[:len(Z)]
    proto2 = emb2_after[len(Z):len(Z) + len(protos)]
    if cat_protos is not None:
        cat_proto2 = emb2_after[len(Z) + len(protos):]
    print("Reduzindo DINOv2 cru...")
    raw2 = reduce_2d(RAW)

    true = np.where(classe == "limpo", 0, 1)
    outcome = _outcome(fused, true, thr)   # no limiar primario (config)

    # separabilidade objetiva por distancia ao "normal": ANTES (cru) vs DEPOIS (z)
    sep_raw = _centroid_auroc(RAW, true)
    sep_z = _centroid_auroc(Z, true)

    # ---- PRINCIPAL: antes x depois, interativo e autoexplicativo (apresentar isto) ----
    _interactive_presentation(rep, raw2, z2, proto2, classe, split, names, sp, fused,
                              sep_raw, sep_z)

    # ---- figuras estaticas de apoio (usadas no relatorio) ----
    _static_embedding(rep, raw2, z2, proto2, classe)
    _static_decision(rep, sp, classe, thr, te, z_te, dec, fus, ae_te)
    _static_outcome(rep, z2, proto2, outcome, split)

    # ---- MULTI-CLUSTER: espaco colorido por CATEGORIA + protótipos de categoria ----
    if multiclass:
        _static_categories(rep, raw2, z2, cat_proto2, cat_proto_ids, categoria, split)
        _interactive_categories(rep, raw2, z2, cat_proto2, cat_proto_ids, categoria,
                                split, names, fused)

    # ---- TRADEOFF precisao×recall: varios limiares lado a lado ----
    thr_by_p = {p: select_threshold_for_precision(fused_va, va["label"], p)[0] for p in precisions}
    _tradeoff_static(rep, z2, proto2, split, true, fused, thr_by_p)

    # ---- HTML de aprofundamento (acerto/erro TP/TN/FP/FN), apenas com --extra-html ----
    if args.extra_html:
        _interactive(rep, z2, proto2, classe, split, names, sp, fused, thr)
        _interactive_outcome(rep, z2, proto2, outcome, classe, split, names, sp, fused, thr)
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
    print(f"  >>> clusters_apresentacao.html  <- gate 'tem erro?' (Estagio 1, apresentar)")
    if multiclass:
        print(f"  >>> categorias_apresentacao.html  <- MULTI-CLUSTER por categoria (Estagio 2)")
        print(f"      embedding_categorias.png  (espaco z colorido pelas {len(CATEGORIES)} classes + protótipos)")
    print(f"      separabilidade por distancia ao normal (AUROC): ANTES {sep_raw:.2f} -> DEPOIS {sep_z:.2f}")
    print("  embedding_space.png  decision_space.png  outcome_space.png  tradeoff_outcome.png  (apoio)")
    if args.extra_html:
        print("  embedding_interactive.html  embedding_interactive_outcome.html  "
              + "  ".join(f"embedding_interactive_outcome_p{p:.2f}.html" for p in precisions))
    else:
        print("  (use --extra-html p/ os HTML de acerto/erro TP/TN/FP/FN por limiar)")


def _static_embedding(rep, raw2, z2, proto2, classe):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    for title, xy, a in [("Raw DINOv2 (BEFORE) — clean and errors mixed", raw2, ax[0]),
                         ("z learned by the siamese head (AFTER) — clean forms a cluster", z2, ax[1])]:
        for c in ["erro_sintetico", "erro_real", "limpo"]:
            mk = classe == c
            a.scatter(xy[mk, 0], xy[mk, 1], s=14, alpha=0.6, c=COLORS[c], label=CLASS_LABEL[c], edgecolors="none")
        a.set_title(title); a.set_xticks([]); a.set_yticks([]); a.legend(loc="best", fontsize=8)
    ax[1].scatter(proto2[:, 0], proto2[:, 1], s=400, marker="*", c=COLORS["prototipo"],
                  edgecolors="white", linewidths=1.5, label="clean prototype", zorder=5)
    ax[1].legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(rep / "embedding_space.png", dpi=120); plt.close(fig)


def _static_categories(rep, raw2, z2, cat_proto2, cat_proto_ids, categoria, split):
    """MULTI-CLUSTER (PNG): espaco cru (antes) vs z aprendido (depois), colorido por CATEGORIA.
    treino = circulo pequeno (fundo); teste held-out = losango destacado; ★ = protótipo de categoria."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cats = [c for c in CATEGORIES if (categoria == c).any()]
    tr = split == "train"; teM = split == "test"
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    for title, xy, a in [("Raw DINOv2 (BEFORE) — categories mixed", raw2, ax[0]),
                         ("z learned (AFTER) — per-category clusters · ● train  ◆ test  ★ prototype", z2, ax[1])]:
        for c in cats:
            col = CATEGORY_COLORS.get(c, "#999999")
            mk = (categoria == c) & tr
            a.scatter(xy[mk, 0], xy[mk, 1], s=12, alpha=0.45, c=col, label=c, edgecolors="none")
            mk = (categoria == c) & teM
            a.scatter(xy[mk, 0], xy[mk, 1], s=60, alpha=0.95, c=col, marker="D",
                      edgecolors="black", linewidths=0.6)
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([]); a.legend(loc="best", fontsize=8)
    if cat_proto2 is not None:
        for i, cid in enumerate(cat_proto_ids):
            nm = ID_TO_CATEGORY[int(cid)]
            ax[1].scatter(cat_proto2[i, 0], cat_proto2[i, 1], s=320, marker="*",
                          c=CATEGORY_COLORS.get(nm, "#000000"), edgecolors="black",
                          linewidths=1.2, zorder=5)
    fig.tight_layout(); fig.savefig(rep / "embedding_categorias.png", dpi=120); plt.close(fig)


def _interactive_categories(rep, raw2, z2, cat_proto2, cat_proto_ids, categoria, split, names, fused):
    """MULTI-CLUSTER (HTML interativo): antes×depois por categoria, com protótipos de categoria.
    Clique na legenda p/ isolar uma categoria; passe o mouse p/ ver arquivo, split e p(erro)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    names = np.asarray(names)
    cats = [c for c in CATEGORIES if (categoria == c).any()]
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.05,
        subplot_titles=("BEFORE · raw DINOv2 — categories mixed",
                        "AFTER · z-space (siamese) — clusters per category · ★ = prototype"))
    for c in cats:
        mk = categoria == c
        col = CATEGORY_COLORS.get(c, "#999999")
        sym = np.where(split[mk] == "test", "diamond", "circle")   # ● train · ◆ test
        sz = np.where(split[mk] == "test", 11, 6)
        ln = np.where(split[mk] == "test", 0.8, 0.0)
        fig.add_trace(go.Scatter(
            x=raw2[mk, 0], y=raw2[mk, 1], mode="markers", name=c, legendgroup=c, showlegend=False,
            marker=dict(size=sz, color=col, symbol=sym, opacity=0.6, line=dict(width=ln, color="black")),
            customdata=np.stack([names[mk], split[mk]], 1),
            hovertemplate="<b>%{customdata[0]}</b><br>" + c + " · %{customdata[1]}<extra></extra>"),
            row=1, col=1)
        fig.add_trace(go.Scatter(
            x=z2[mk, 0], y=z2[mk, 1], mode="markers", name=c, legendgroup=c, showlegend=True,
            marker=dict(size=sz, color=col, symbol=sym, opacity=0.78, line=dict(width=ln, color="black")),
            customdata=np.stack([names[mk], split[mk], fused[mk].round(3)], 1),
            hovertemplate="<b>%{customdata[0]}</b><br>" + c +
                          " · %{customdata[1]} · p(error)=%{customdata[2]}<extra></extra>"),
            row=1, col=2)
    if cat_proto2 is not None:
        for i, cid in enumerate(cat_proto_ids):
            nm = ID_TO_CATEGORY[int(cid)]
            fig.add_trace(go.Scatter(
                x=[cat_proto2[i, 0]], y=[cat_proto2[i, 1]], mode="markers", name=f"★ {nm}",
                legendgroup=nm, showlegend=False,
                marker=dict(size=18, color=CATEGORY_COLORS.get(nm, "#000000"), symbol="star",
                            line=dict(width=1.4, color="black"))), row=1, col=2)
    fig.update_layout(
        title="Multi-cluster: error categories in the learned z-space  (● train · ◆ test held-out)",
        height=620, hovermode="closest",
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        margin=dict(l=16, r=16, t=64, b=24), plot_bgcolor="#FAFBFD", paper_bgcolor="white")
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.write_html(rep / "categorias_apresentacao.html")


def _static_decision(rep, sp, classe, thr, te, z_te, dec, fus, ae_te):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for c in ["limpo", "erro_real", "erro_sintetico"]:
        ax[0].hist(sp[classe == c], bins=30, alpha=0.55, color=COLORS[c], label=CLASS_LABEL[c])
    ax[0].set_title("Distance to the CLEAN prototype (decision rule)")
    ax[0].set_xlabel("score = 1 − cos(z, prototype)"); ax[0].legend(fontsize=8)
    # curva PR no test (fusao)
    fused_te = fus.predict_proba(np.stack([dec.scores(z_te), ae_te], 1))[:, 1]
    p, r, _ = precision_recall_curve(te["label"], fused_te)
    ax[1].plot(r, p); ax[1].set_title("Precision–Recall (real TEST, fusion)")
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
            x=z2[mk, 0], y=z2[mk, 1], mode="markers", name=CLASS_LABEL[c],
            marker=dict(size=7, color=COLORS[c], opacity=0.7, line=dict(width=0)),
            customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>split=%{customdata[1]}<br>"
                          "dist_to_prototype=%{customdata[2]}<br>p(error)=%{customdata[3]}<extra>" + CLASS_LABEL[c] + "</extra>"))
    fig.add_trace(go.Scatter(x=proto2[:, 0], y=proto2[:, 1], mode="markers",
                             name="clean prototype", marker=dict(size=20, color="black", symbol="star")))
    fig.update_layout(title=f"Learned z-space (UMAP) — threshold p(error)={thr:.3f} | "
                            "clean=green, real error=red, synthetic=orange",
                      width=1100, height=750, hovermode="closest")
    fig.write_html(rep / "embedding_interactive.html")


_PAGE_TEMPLATE = """<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clusters — before × after (presentation)</title>
<style>
  :root { --ink:#1A2027; --muted:#5b6b78; --line:#e3e8ee;
          --clean:#2ca02c; --real:#d62728; --synth:#ff7f0e; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Segoe UI",Helvetica,Arial,sans-serif; color:var(--ink);
         background:#f4f6f9; -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1220px; margin:0 auto; padding:24px 20px 52px; }
  header h1 { font-size:23px; margin:0 0 8px; letter-spacing:-0.01em; }
  header p.lead { font-size:15px; color:var(--muted); margin:0; line-height:1.55; max-width:980px; }
  header p.lead b { color:var(--ink); }
  .card { background:#fff; border:1px solid var(--line); border-radius:14px;
          box-shadow:0 1px 3px rgba(16,24,40,.05); }
  .plot { padding:10px 10px 2px; margin:18px 0; }
  .guia { padding:22px 26px; }
  .guia h2 { font-size:13px; letter-spacing:.06em; text-transform:uppercase;
             color:var(--muted); margin:0 0 16px; }
  ol.fala { margin:0; padding:0; list-style:none; counter-reset:n; }
  ol.fala li { counter-increment:n; position:relative; padding:0 0 16px 46px; line-height:1.6; font-size:15px; }
  ol.fala li:last-child { padding-bottom:0; }
  ol.fala li::before { content:counter(n); position:absolute; left:0; top:-1px; width:30px; height:30px;
                       border-radius:50%; background:var(--ink); color:#fff; font-weight:600;
                       display:flex; align-items:center; justify-content:center; font-size:14px; }
  .chip { display:inline-block; padding:1px 9px; border-radius:999px; font-size:12.5px; font-weight:600;
          color:#fff; }
  .num { font-variant-numeric:tabular-nums; font-weight:700; }
  .tips { margin:18px 0 0; padding-top:16px; border-top:1px dashed var(--line);
          font-size:13.5px; color:var(--muted); line-height:1.6; }
  .tips b { color:var(--ink); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Embedding space — before × after training</h1>
    <p class="lead">The siamese head learns a space where <b>clean screens cluster together</b> (near the ★)
       and <b>errors move away</b>. The decision is simply the <b>distance to the ★ (clean prototype)</b>:
       near ⇒ clean, far ⇒ error.</p>
  </header>

  <div class="card plot">__PLOT__</div>

  <div class="card guia">
    <h2>What to say in the presentation</h2>
    <ol class="fala">
      <li><b>Before (left).</b> In raw DINOv2, <span class="chip" style="background:var(--clean)">clean</span>
        and <span class="chip" style="background:var(--real)">errors</span> are <b>mixed</b> — separating
        by distance to "normal" barely works (AUROC <span class="num">__SEP_RAW__</span>).</li>
      <li><b>After (right).</b> The siamese head <b>reshapes the space</b>: clean screens
        <b>concentrate in one region</b> (around the ★) and errors (real and synthetic) <b>move away</b>. The
        <b>★</b> is the <b>prototype</b> — the center of "normal".</li>
      <li><b>How we decide.</b> We measure <b>each screen's distance to the ★</b>. In this geometry, that
        distance separates error from non-error with <b>AUROC <span class="num">__SEP_Z__</span></b>
        (it was __SEP_RAW__ in the raw space). It is the clustering that makes the decision <b>simple and explainable</b>
        — without looking at resolution or device.</li>
    </ol>
    <p class="tips">Interact live: <b>click</b> a class in the legend to isolate it ·
       <b>drag</b> to zoom · <b>hover</b> a point to see the file,
       its distance to the prototype and p(error).<br>
       <span style="opacity:.85">The 2D map is a <b>projection</b> (UMAP) for visualization only; the real separation is
       the distance above. Held-out <b>test</b> performance (in the report) is AUROC 0.90.</span></p>
  </div>
</div>
</body>
</html>"""


def _interactive_presentation(rep, raw2, z2, proto2, classe, split, names, sp, fused,
                              sep_raw, sep_z):
    """PRINCIPAL: antes x depois lado a lado, interativo, com roteiro do que falar.

    Conta a historia da clusterizacao em uma figura: no espaco cru (ANTES) limpas e
    erros estao misturados; no espaco z (DEPOIS) as limpas formam um cluster compacto,
    os erros caem fora, e a estrela e o prototipo. A decisao e a distancia ao prototipo."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    LABEL = {"limpo": "Clean (no error)", "erro_real": "Real errors",
             "erro_sintetico": "Synthetic errors"}
    order = ["limpo", "erro_real", "erro_sintetico"]
    names = np.asarray(names)

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.05,
        subplot_titles=(
            f"BEFORE · raw DINOv2 (untrained) — separates by distance: AUROC {sep_raw:.2f}",
            f"AFTER · z-space (siamese) — AUROC {sep_z:.2f} · ★ = prototype"))

    for c in order:
        mk = classe == c
        # ANTES (col 1): hover simples (distancia/p(erro) nao se aplicam ao espaco cru)
        fig.add_trace(go.Scatter(
            x=raw2[mk, 0], y=raw2[mk, 1], mode="markers", name=LABEL[c],
            legendgroup=c, showlegend=False,
            marker=dict(size=6, color=COLORS[c], opacity=0.60, line=dict(width=0)),
            customdata=np.stack([names[mk], split[mk]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>" + LABEL[c]
                          + " · %{customdata[1]}<extra></extra>"), row=1, col=1)
        # AFTER (col 2): hover completo; a legenda aparece aqui (legendgroup sincroniza os dois lados)
        fig.add_trace(go.Scatter(
            x=z2[mk, 0], y=z2[mk, 1], mode="markers", name=LABEL[c],
            legendgroup=c, showlegend=True,
            marker=dict(size=6, color=COLORS[c], opacity=0.65, line=dict(width=0)),
            customdata=np.stack([names[mk], split[mk], sp[mk].round(3), fused[mk].round(3)], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>" + LABEL[c]
                          + " · %{customdata[1]}<br>distance to prototype=%{customdata[2]}"
                            " · p(error)=%{customdata[3]}<extra></extra>"), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=proto2[:, 0], y=proto2[:, 1], mode="markers", name="★ Clean prototype",
        marker=dict(size=22, color="#111111", symbol="star",
                    line=dict(width=1.6, color="white"))), row=1, col=2)
    fig.add_annotation(x=proto2[0, 0], y=proto2[0, 1], xref="x2", yref="y2",
                       text='center of "normal"', showarrow=True, arrowhead=2,
                       ax=40, ay=-40, arrowcolor="#111", font=dict(size=12, color="#111"),
                       bgcolor="rgba(255,255,255,0.75)")

    fig.update_layout(
        height=600, hovermode="closest",
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.10, xanchor="center", x=0.5),
        margin=dict(l=16, r=16, t=64, b=24),
        plot_bgcolor="#FAFBFD", paper_bgcolor="white")
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    plot = fig.to_html(full_html=False, include_plotlyjs=True,
                       config={"displaylogo": False, "responsive": True})
    page = (_PAGE_TEMPLATE.replace("__SEP_RAW__", f"{sep_raw:.2f}")
                          .replace("__SEP_Z__", f"{sep_z:.2f}")
                          .replace("__PLOT__", plot))
    (rep / "clusters_apresentacao.html").write_text(page, encoding="utf-8")


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
                x=z2[mk, 0], y=z2[mk, 1], mode="markers", name=f"{OUTCOME_LABEL[oc]} | {s}",
                marker=dict(size=11 if s == "test" else 6, color=OUTCOME_COLORS[oc],
                            symbol=SYMBOL[s], opacity=0.85 if s == "test" else 0.45,
                            line=dict(width=1 if s == "test" else 0, color="black")),
                customdata=custom,
                hovertemplate="<b>%{customdata[0]}</b><br>class=%{customdata[1]}<br>"
                              "dist_to_prototype=%{customdata[2]}<br>p(error)=%{customdata[3]}"
                              f"<extra>{OUTCOME_LABEL[oc]} | {s}</extra>"))
    fig.add_trace(go.Scatter(x=proto2[:, 0], y=proto2[:, 1], mode="markers",
                             name="clean prototype", marker=dict(size=20, color="black", symbol="star")))
    fig.update_layout(
        title=f"Model outcome (threshold p(error)={thr:.3f}) — circle=train (in-sample), "
              "diamond=test (held-out). Green/blue=correct, orange=false alarm, red=missed error",
        width=1150, height=780, hovermode="closest")
    fig.write_html(rep / f"embedding_interactive_outcome{suffix}.html")


def _static_outcome(rep, z2, proto2, outcome, split):
    """PNG: held-out (test) colorido por TP/TN/FP/FN sobre o fundo do train (cinza)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 7))
    tr = split == "train"; teM = split == "test"
    ax.scatter(z2[tr, 0], z2[tr, 1], s=8, c="#dddddd", label="train (background)", edgecolors="none")
    for oc in ["TN_acerto_limpo", "TP_acerto_erro", "FP_falso_alarme", "FN_erro_perdido"]:
        mk = teM & (outcome == oc)
        if mk.sum():
            ax.scatter(z2[mk, 0], z2[mk, 1], s=70, c=OUTCOME_COLORS[oc], marker="D",
                       edgecolors="black", linewidths=0.6, label=f"{OUTCOME_LABEL[oc]} ({int(mk.sum())})")
    ax.scatter(proto2[:, 0], proto2[:, 1], s=400, marker="*", c="black",
               edgecolors="white", linewidths=1.5, label="prototype", zorder=5)
    ax.set_title("Held-out TEST by model outcome — where FP and FN are")
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
                           edgecolors="black", linewidths=0.6, label=f"{OUTCOME_LABEL[o]} ({int(mk.sum())})")
        ax.scatter(proto2[:, 0], proto2[:, 1], s=350, marker="*", c="black",
                   edgecolors="white", linewidths=1.5, zorder=5)
        ax.set_title(f"target precision {p:.2f}  (thr={thr:.3f})\n"
                     f"TEST: precision={prec:.2f}  recall={rec:.2f}  | TP={tp} FP={fp} FN={fn}")
        ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="best", fontsize=8)
    fig.suptitle("Precision×recall tradeoff — same z-space, different thresholds "
                 "(diamond=held-out test, ★=prototype)", fontsize=12)
    fig.tight_layout(); fig.savefig(rep / "tradeoff_outcome.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
