#!/usr/bin/env python
"""Cluster do TEST (held-out) + matriz de confusao no MESMO ponto de operacao.

POR QUE ESTE ARQUITETO (a versao justa, clara e defensavel):
- O clusters_apresentacao.html mistura train+test e reporta separabilidade IN-SAMPLE
  (AUROC ~0.94), otimista. Aqui usamos SO o test held-out e o numero HONESTO (AUROC ~0.90).
- O tradeoff_outcome.png mostra DOIS limiares de alta-precisao -> confunde. Aqui ha UM
  ponto de operacao (balanceado/F1 por padrao) e cluster + regua + matriz contam a MESMA
  historia, no MESMO limiar.

Gera (em artifacts/reports/):
  clusters_test.html         -> PRINCIPAL p/ a reuniao. Numa pagina: (1) cluster do test no
                                espaco z, colorido por acerto/erro; (2) "regua de decisao"
                                p(erro) com o limiar (a decisao REAL); (3) a matriz de
                                confusao embutida + metricas; (4) roteiro do que falar.
  confusion_matrix_test.png  -> a mesma matriz em PNG, para slide/relatorio.

O limiar e escolhido na VALIDACAO (sem data-snooping) e medido no TEST.

Uso:
  python scripts/visualize_test.py --config configs/default.yaml          # balanceado (F1)
  python scripts/visualize_test.py --objective precision --target-precision 0.95
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score)

from siamese.config import Config
from siamese.features import load_embeddings
from siamese.train import load_model
from siamese.evaluate import model_embeddings, grouped_bootstrap_ci, _aux_err
from siamese.decision import (fit_prototypes, PrototypeDecider, select_threshold_for_specificity,
                              select_threshold_for_precision, select_threshold_max_f1)

# acerto/erro -> (rotulo amigavel EN, cor)
OUTCOME = {
    "TP": ("✔ Error detected", "#2ca02c"),
    "TN": ("✔ Clean correct", "#1f77b4"),
    "FP": ("✘ False alarm", "#ff7f0e"),
    "FN": ("✘ Missed error", "#d62728"),
}


def reduce_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    Xn = normalize(X)
    try:
        import umap
        return umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                         random_state=seed).fit_transform(Xn)
    except Exception as e:  # pragma: no cover
        print(f"  (umap indisponivel: {e}; usando t-SNE)")
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="pca", perplexity=30,
                    random_state=seed).fit_transform(Xn)


def _centroid_score(query_raw: np.ndarray, clean_ref_raw: np.ndarray) -> np.ndarray:
    """1 - cos(query, centro_das_limpas_de_referencia). Mesma REGRA da decisao, mas no
    espaco bruto (DINOv2 cru) -> serve p/ medir 'antes x depois' de forma honesta no test."""
    ref = normalize(clean_ref_raw).mean(0)
    ref = ref / (np.linalg.norm(ref) + 1e-9)
    return 1.0 - normalize(query_raw) @ ref


def _outcome_codes(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    oc = np.empty(len(true), dtype=object)
    oc[(pred == 1) & (true == 1)] = "TP"
    oc[(pred == 0) & (true == 0)] = "TN"
    oc[(pred == 1) & (true == 0)] = "FP"
    oc[(pred == 0) & (true == 1)] = "FN"
    return oc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--objective", choices=["f1", "precision", "specificity"], default=None,
                    help="ponto de operacao; default = o do config (f1 = balanceado)")
    ap.add_argument("--target-precision", type=float, default=None)
    ap.add_argument("--final-test", action="store_true",
                    help="OBRIGATORIO: este script visualiza o TESTE held-out (destrava; uma vez).")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    objective = args.objective or cfg.decision.objective
    target = args.target_precision or cfg.decision.target_precision
    emb_dir = Path(cfg.paths.emb_dir)
    rep = Path(cfg.paths.reports_dir); rep.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    # Fase 0: visualizar o TESTE exige --final-test (blindagem; processar o teste uma so vez).
    if not args.final_test:
        print("visualize_test.py mostra o TESTE held-out. Rode com --final-test "
              "(uma vez, apos congelar a config). Para iterar, use scripts/visualize.py (val).")
        return
    from siamese.protocol import allow_test_access
    allow_test_access(True)

    tr = load_embeddings(emb_dir / "train.npz")
    va = load_embeddings(emb_dir / "val.npz")
    te = load_embeddings(emb_dir / "test.npz")
    model = load_model(Path(cfg.paths.models_dir) / "siamese_head.pt", device=device)

    z_tr, aux_tr_raw = model_embeddings(model, tr["emb"], device)
    z_va, aux_va_raw = model_embeddings(model, va["emb"], device)
    z_te, aux_te_raw = model_embeddings(model, te["emb"], device)
    # cabeca aux multi-classe -> logits [N,7]; reduz a score ESCALAR de erro (1 - softmax[clean]),
    # mesma logica de evaluate._aux_err, p/ casar a forma com dec.scores() na fusao (corrige o
    # ValueError de np.stack em multi-classe).
    _mc = cfg.train.multiclass
    aux_tr = _aux_err(aux_tr_raw, _mc); aux_va = _aux_err(aux_va_raw, _mc); aux_te = _aux_err(aux_te_raw, _mc)

    # --- decisao: prototipo do limpo (treino) + fusao calibrada na VAL ---
    protos = fit_prototypes(z_tr[tr["label"] == 0], k=cfg.decision.k_prototypes, seed=cfg.seed)
    dec = PrototypeDecider(protos, 0.0, target)
    Xva = np.stack([dec.scores(z_va), aux_va], axis=1)
    fus = LogisticRegression(max_iter=1000).fit(Xva, va["label"])
    fused_va = fus.predict_proba(Xva)[:, 1]
    sp_te = dec.scores(z_te)
    fused_te = fus.predict_proba(np.stack([sp_te, aux_te], axis=1))[:, 1]

    # --- limiar fixado na VAL (sem data-snooping) ---
    if objective == "precision":
        thr, info = select_threshold_for_precision(fused_va, va["label"], target)
    elif objective == "specificity":
        thr, info = select_threshold_for_specificity(fused_va, va["label"], cfg.decision.target_specificity)
    else:
        thr, info = select_threshold_max_f1(fused_va, va["label"])

    # --- metricas HONESTAS no TEST ---
    y = te["label"].astype(int)
    pred = (fused_te > thr).astype(int)
    oc = _outcome_codes(pred, y)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = accuracy_score(y, pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = f1_score(y, pred, zero_division=0)
    auroc = roc_auc_score(y, fused_te)
    ap = average_precision_score(y, fused_te)
    ci = grouped_bootstrap_ci(accuracy_score, y, pred, te["group"])

    # separabilidade por DISTANCIA ao normal, medida SO no test (antes x depois, honesto)
    sep_raw = roc_auc_score(y, _centroid_score(te["emb"], tr["emb"][tr["label"] == 0]))
    sep_z = roc_auc_score(y, sp_te)  # = AUROC do score de prototipo no test

    n_err = int((y == 1).sum()); n_ok = int((y == 0).sum())
    print(f"[test] n={len(y)} (erros={n_err}, limpas={n_ok})  objetivo={objective}  thr={thr:.3f}")
    print(f"       acc={acc:.3f} (IC95 {ci[0]:.2f}-{ci[1]:.2f})  prec={prec:.3f} rec={rec:.3f} "
          f"f1={f1:.3f}  AUROC={auroc:.3f} AP={ap:.3f}")
    print(f"       separabilidade por distancia ao normal NO TEST: cru {sep_raw:.2f} -> z {sep_z:.2f}")
    print(f"       confusao: TP={tp} TN={tn} FP={fp} FN={fn}")

    # --- projecao 2D: train (fundo) + test (colorido) + prototipo, no MESMO mapa ---
    print("Reduzindo para 2D (UMAP/t-SNE)...")
    Z = np.concatenate([z_tr, z_te, protos])
    xy = reduce_2d(Z, seed=cfg.seed)
    n_tr = len(z_tr)
    xy_tr, xy_te, xy_proto = xy[:n_tr], xy[n_tr:n_tr + len(z_te)], xy[n_tr + len(z_te):]

    _confusion_png(rep, tn, fp, fn, tp, dict(acc=acc, prec=prec, rec=rec, f1=f1,
                                             auroc=auroc, ap=ap, thr=thr, objective=objective))
    _build_html(rep, te, xy_tr, xy_te, xy_proto, oc, fused_te, sp_te, thr,
                dict(tn=tn, fp=fp, fn=fn, tp=tp, acc=acc, prec=prec, rec=rec, f1=f1,
                     auroc=auroc, ap=ap, ci=ci, sep_raw=sep_raw, sep_z=sep_z,
                     n=len(y), n_err=n_err, n_ok=n_ok, objective=objective), cfg.seed)

    print(f"\n>>> {rep/'clusters_test.html'}  <- APRESENTAR (cluster do test + matriz, mesmo limiar)")
    print(f"    {rep/'confusion_matrix_test.png'}  <- matriz p/ slide")


# ----------------------------------------------------------------------------- matriz PNG
def _confusion_png(rep: Path, tn, fp, fn, tp, m: dict) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array([[tn, fp], [fn, tp]])
    cell_color = np.array([["#1f77b4", "#ff7f0e"], ["#d62728", "#2ca02c"]])  # TN FP / FN TP
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=cell_color[i, j], alpha=0.18))
            ax.text(j + 0.5, 1 - i + 0.60, str(cm[i, j]), ha="center", va="center",
                    fontsize=30, fontweight="bold", color=cell_color[i, j])
            ax.text(j + 0.5, 1 - i + 0.28, labels[i, j], ha="center", va="center",
                    fontsize=12, color="#444")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["predicted CLEAN", "predicted ERROR"], fontsize=11)
    ax.set_yticks([1.5, 0.5]); ax.set_yticklabels(["actual CLEAN", "actual ERROR"], fontsize=11)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Confusion matrix — held-out TEST ({'balanced/F1' if m['objective']=='f1' else 'high precision'})\n"
                 f"acc={m['acc']:.2f} (95% CI)  ·  prec={m['prec']:.2f}  rec={m['rec']:.2f}  "
                 f"F1={m['f1']:.2f}  ·  AUROC={m['auroc']:.2f} AP={m['ap']:.2f}",
                 fontsize=10.5, pad=12)
    fig.tight_layout(); fig.savefig(rep / "confusion_matrix_test.png", dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- HTML
_TEMPLATE = """<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Held-out TEST — cluster + matrix (same threshold)</title>
<style>
  :root { --ink:#1A2027; --muted:#5b6b78; --line:#e3e8ee;
          --tp:#2ca02c; --tn:#1f77b4; --fp:#ff7f0e; --fn:#d62728; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Segoe UI",Helvetica,Arial,sans-serif; color:var(--ink);
         background:#f4f6f9; -webkit-font-smoothing:antialiased; }
  .wrap { max-width:1240px; margin:0 auto; padding:24px 20px 52px; }
  header h1 { font-size:23px; margin:0 0 8px; letter-spacing:-0.01em; }
  header p.lead { font-size:15px; color:var(--muted); margin:0; line-height:1.55; max-width:1020px; }
  header p.lead b { color:var(--ink); }
  .card { background:#fff; border:1px solid var(--line); border-radius:14px;
          box-shadow:0 1px 3px rgba(16,24,40,.05); margin:18px 0; }
  .plot { padding:10px 10px 2px; }
  .row { display:flex; gap:18px; flex-wrap:wrap; }
  .col { flex:1 1 360px; }
  .pad { padding:22px 26px; }
  h2.sec { font-size:13px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin:0 0 16px; }
  table.cm { border-collapse:collapse; margin:6px auto; }
  table.cm td, table.cm th { border:1px solid var(--line); padding:14px 18px; text-align:center; }
  table.cm th { color:var(--muted); font-weight:600; font-size:13px; }
  table.cm .v { font-size:30px; font-weight:800; font-variant-numeric:tabular-nums; }
  table.cm .lab { font-size:12px; color:#667; }
  .tp { background:rgba(44,160,44,.12); } .tn { background:rgba(31,119,180,.12); }
  .fp { background:rgba(255,127,14,.14); } .fn { background:rgba(214,39,40,.12); }
  .metrics { display:flex; gap:10px; flex-wrap:wrap; margin:14px 0 0; }
  .metric { background:#f7f9fc; border:1px solid var(--line); border-radius:10px; padding:10px 14px; min-width:96px; }
  .metric .k { font-size:11.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .metric .n { font-size:22px; font-weight:800; font-variant-numeric:tabular-nums; }
  .metric.head { background:#eef4ff; border-color:#cfe0ff; }
  ol.fala { margin:0; padding:0; list-style:none; counter-reset:n; }
  ol.fala li { counter-increment:n; position:relative; padding:0 0 14px 44px; line-height:1.6; font-size:14.5px; }
  ol.fala li:last-child { padding-bottom:0; }
  ol.fala li::before { content:counter(n); position:absolute; left:0; top:-1px; width:28px; height:28px;
                       border-radius:50%; background:var(--ink); color:#fff; font-weight:600;
                       display:flex; align-items:center; justify-content:center; font-size:13px; }
  .chip { display:inline-block; padding:1px 9px; border-radius:999px; font-size:12.5px; font-weight:600; color:#fff; }
  .tips { margin:16px 0 0; padding-top:14px; border-top:1px dashed var(--line); font-size:13px; color:var(--muted); line-height:1.6; }
  .tips b { color:var(--ink); }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Held-out TEST — the comparable, defensible evidence</h1>
    <p class="lead">These <b>__N__ screens</b> (__N_ERR__ with errors · __N_OK__ clean) were <b>never seen during training</b>.
       The <b>cluster</b> (left), the <b>decision ruler</b> (right) and the <b>confusion matrix</b> (below) all describe
       <b>the same decision</b>, at the <b>same threshold</b> (chosen on validation). Threshold-free comparison number:
       <b>AUROC __AUROC__ · AP __AP__</b>.</p>
  </header>

  <div class="card plot">__PLOT__</div>

  <div class="row">
    <div class="card pad col">
      <h2 class="sec">Confusion matrix · TEST · same threshold (__OBJ__)</h2>
      <table class="cm">
        <tr><th></th><th>predicted CLEAN</th><th>predicted ERROR</th></tr>
        <tr><th>actual CLEAN</th>
            <td class="tn"><div class="v" style="color:var(--tn)">__TN__</div><div class="lab">TN · correct</div></td>
            <td class="fp"><div class="v" style="color:var(--fp)">__FP__</div><div class="lab">FP · false alarm</div></td></tr>
        <tr><th>actual ERROR</th>
            <td class="fn"><div class="v" style="color:var(--fn)">__FN__</div><div class="lab">FN · missed error</div></td>
            <td class="tp"><div class="v" style="color:var(--tp)">__TP__</div><div class="lab">TP · error detected</div></td></tr>
      </table>
      <div class="metrics">
        <div class="metric head"><div class="k">AUROC</div><div class="n">__AUROC__</div></div>
        <div class="metric head"><div class="k">AP</div><div class="n">__AP__</div></div>
        <div class="metric"><div class="k">Accuracy</div><div class="n">__ACC__</div></div>
        <div class="metric"><div class="k">Precision</div><div class="n">__PREC__</div></div>
        <div class="metric"><div class="k">Recall</div><div class="n">__REC__</div></div>
        <div class="metric"><div class="k">F1</div><div class="n">__F1__</div></div>
      </div>
      <p class="tips">Accuracy with <b>95% CI</b> (per-ticket bootstrap): <b>__CI_LO__ – __CI_HI__</b>.
         <b>AUROC/AP do not depend on the threshold</b> → lead the model comparison with them. The matrix is the
         <b>balanced</b> operating point (threshold that maximizes F1 on validation) — it favors no single metric.</p>
    </div>

    <div class="card pad col">
      <h2 class="sec">What to say in the presentation</h2>
      <ol class="fala">
        <li><b>It is held-out.</b> These screens were not in training — the fair test of generalization.</li>
        <li><b>Cluster (left).</b> Clean screens fall near the <b>★ prototype</b>; errors move away.
            Color is the <b>outcome at the matrix threshold</b>:
            <span class="chip" style="background:var(--tn)">clean ok</span>
            <span class="chip" style="background:var(--tp)">error detected</span>
            <span class="chip" style="background:var(--fp)">false alarm</span>
            <span class="chip" style="background:var(--fn)">missed error</span>.</li>
        <li><b>Ruler (right) = the real decision.</b> Each screen has a <b>p(error)</b>; the dashed line is the
            <b>threshold</b>. Bottom row = clean screens; top row = error screens.
            Whoever crosses the threshold to the wrong side becomes FP (clean) or is lost as FN (error).</li>
        <li><b>Matrix (below) = the count.</b> Same points, tallied up. <b>Same threshold</b> as the cluster and the ruler.</li>
        <li><b>To compare with other models:</b> use <b>AUROC __AUROC__ / AP __AP__</b> (threshold-free). The matrix shows
            <b>one</b> honest operating point; the threshold was fixed on <b>validation</b> and measured here — no data-snooping.</li>
      </ol>
      <p class="tips"><b>Honesty of the 2D map:</b> the cluster is a projection (UMAP) for intuition only. The true decision
         is the <b>p(error) ruler</b> on the right (which uses distance to the prototype <i>+</i> the auxiliary head) — so a point
         may look "out of place" in 2D yet still be on the correct side of the threshold.
         <br>Separability by distance-to-normal alone, <b>measured on this test set</b>: raw space <b>__SEP_RAW__</b> →
         learned space <b>__SEP_Z__</b> (the final fusion reaches AUROC __AUROC__).</p>
    </div>
  </div>
</div>
</body>
</html>"""


def _build_html(rep, te, xy_tr, xy_te, xy_proto, oc, fused_te, sp_te, thr, m, seed) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = np.array([Path(str(p)).name for p in te["path"]])
    true = te["label"].astype(int)
    rng = np.random.RandomState(seed)
    jitter = (rng.rand(len(true)) - 0.5) * 0.5  # espalha verticalmente na regua

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.60, 0.40], horizontal_spacing=0.07,
        subplot_titles=("TEST cluster in z-space (★ = clean prototype)",
                        "Decision ruler · p(error) and the threshold"))

    # fundo: train (cinza) p/ dar o contexto do cluster limpo
    fig.add_trace(go.Scatter(
        x=xy_tr[:, 0], y=xy_tr[:, 1], mode="markers", name="train (background)",
        marker=dict(size=5, color="#d9dee5", opacity=0.55, line=dict(width=0)),
        hoverinfo="skip", showlegend=True), row=1, col=1)

    # test colorido por resultado (cluster, col1) + regua (col2)
    for code in ["TN", "TP", "FP", "FN"]:
        mk = oc == code
        if mk.sum() == 0:
            continue
        label, color = OUTCOME[code]
        custom = np.stack([names[mk], sp_te[mk].round(3), fused_te[mk].round(3)], axis=1)
        fig.add_trace(go.Scatter(
            x=xy_te[mk, 0], y=xy_te[mk, 1], mode="markers", name=f"{label} ({int(mk.sum())})",
            legendgroup=code,
            marker=dict(size=12, color=color, symbol="diamond",
                        line=dict(width=1, color="white"), opacity=0.92),
            customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>dist. to prototype=%{customdata[1]}"
                          "<br>p(error)=%{customdata[2]}<extra>" + label + "</extra>"),
            row=1, col=1)
        # regua: x = p(erro), y = classe real (0 limpa / 1 erro) + jitter
        fig.add_trace(go.Scatter(
            x=fused_te[mk], y=true[mk] + jitter[mk], mode="markers", name=label,
            legendgroup=code, showlegend=False,
            marker=dict(size=11, color=color, symbol="diamond",
                        line=dict(width=1, color="white"), opacity=0.92),
            customdata=custom,
            hovertemplate="<b>%{customdata[0]}</b><br>p(error)=%{customdata[2]}<extra>" + label + "</extra>"),
            row=1, col=2)

    # estrela do prototipo (col1)
    fig.add_trace(go.Scatter(
        x=xy_proto[:, 0], y=xy_proto[:, 1], mode="markers", name="★ clean prototype",
        marker=dict(size=22, color="#111", symbol="star", line=dict(width=1.6, color="white"))),
        row=1, col=1)

    # linha do limiar na regua (col2)
    fig.add_vline(x=thr, line=dict(color="#111", width=2, dash="dash"), row=1, col=2)
    fig.add_annotation(x=thr, y=1.62, xref="x2", yref="y2", text=f"threshold = {thr:.3f}",
                       showarrow=False, font=dict(size=12, color="#111"),
                       bgcolor="rgba(255,255,255,0.8)")
    fig.add_annotation(x=thr, y=-0.62, xref="x2", yref="y2", ax=-60, ay=0, axref="x2",
                       text="← decides CLEAN", showarrow=False, xanchor="right",
                       font=dict(size=11, color="#5b6b78"))
    fig.add_annotation(x=thr, y=-0.62, xref="x2", yref="y2", text="decides ERROR →",
                       showarrow=False, xanchor="left", font=dict(size=11, color="#5b6b78"))

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(title_text="p(error)", range=[-0.02, 1.02], row=1, col=2)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["CLEAN<br>(actual)", "ERROR<br>(actual)"],
                     range=[-0.9, 1.9], row=1, col=2)
    fig.update_layout(
        height=560, hovermode="closest",
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.16, xanchor="center", x=0.5),
        margin=dict(l=16, r=16, t=60, b=20), plot_bgcolor="#FAFBFD", paper_bgcolor="white")

    plot = fig.to_html(full_html=False, include_plotlyjs=True,
                       config={"displaylogo": False, "responsive": True})

    repl = {
        "__PLOT__": plot, "__N__": str(m["n"]), "__N_ERR__": str(m["n_err"]),
        "__N_OK__": str(m["n_ok"]), "__OBJ__": ("balanced / F1" if m["objective"] == "f1"
                                                else "high precision"),
        "__TN__": str(m["tn"]), "__FP__": str(m["fp"]), "__FN__": str(m["fn"]), "__TP__": str(m["tp"]),
        "__ACC__": f"{m['acc']:.2f}", "__PREC__": f"{m['prec']:.2f}", "__REC__": f"{m['rec']:.2f}",
        "__F1__": f"{m['f1']:.2f}", "__AUROC__": f"{m['auroc']:.2f}", "__AP__": f"{m['ap']:.2f}",
        "__CI_LO__": f"{m['ci'][0]:.2f}", "__CI_HI__": f"{m['ci'][1]:.2f}",
        "__SEP_RAW__": f"{m['sep_raw']:.2f}", "__SEP_Z__": f"{m['sep_z']:.2f}",
    }
    page = _TEMPLATE
    for k, v in repl.items():
        page = page.replace(k, v)
    (rep / "clusters_test.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
