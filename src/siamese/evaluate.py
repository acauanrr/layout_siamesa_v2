"""Avaliacao HONESTA do detector siames.

Principio central (verificado nos dados): um classificador que usa SO metadados de
confound atinge AUROC~0.98 no teste. Logo a metrica GLOBAL e quase pura trapaca. Este
modulo reporta, lado a lado:

  1. Metricas globais do modelo (proto, aux, fusao) vs BASELINES de confound.
  2. METRICA PRIMARIA: subconjunto controlado (unfold-portrait-screenshot) onde
     form-factor/orientacao/kind sao constantes -> separacao = layout real.
  3. Deteccao SINTETICA livre de confound (erros injetados em imagens limpas held-out,
     mesma resolucao): mede deteccao de CONTEUDO de erro sem confound geometrico.
  4. Auditoria same-resolution (erros reais 2076x2152): unico sinal real sem confound #1.
  5. Testes de FALSEABILIDADE (placebo de resolucao; label-shuffle dentro do estrato).
  6. Selecao de limiar por precisao-alvo + IC bootstrap agrupado por ticket; precision@K.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             accuracy_score, f1_score, balanced_accuracy_score)

from .features import load_embeddings
from .decision import (fit_prototypes, PrototypeDecider,
                       select_threshold_for_precision, select_threshold_max_f1)

CLEAN_RES = (2076, 2152)


# ---------- utilitarios ----------
def _resolutions(paths) -> np.ndarray:
    out = []
    for p in paths:
        try:
            with Image.open(p) as im:
                out.append(im.size)
        except Exception:
            out.append((-1, -1))
    return np.array(out)


def _onehot(values: np.ndarray) -> np.ndarray:
    cats = sorted(set(values.tolist()))
    return np.stack([(values == c).astype(float) for c in cats], axis=1)


def confound_matrix(z: dict) -> np.ndarray:
    """Features SO de confound (zero conteudo): kind, form_factor, orientation, resolucao."""
    res = _resolutions(z["path"]).astype(float)
    aspect = (res[:, 0] / np.maximum(res[:, 1], 1)).reshape(-1, 1)
    is_clean_res = ((res[:, 0] == CLEAN_RES[0]) & (res[:, 1] == CLEAN_RES[1])).astype(float).reshape(-1, 1)
    return np.concatenate([
        _onehot(z["kind"]), _onehot(z["form_factor"]), _onehot(z["orientation"]),
        res, aspect, is_clean_res,
    ], axis=1)


def model_embeddings(model, emb: np.ndarray, device="cpu"):
    with torch.no_grad():
        z, aux = model(torch.from_numpy(emb.astype(np.float32)).to(device))
    return z.cpu().numpy(), aux.cpu().numpy()


def grouped_bootstrap_ci(metric_fn, y, score, groups, n=1000, seed=0):
    """IC 95% reamostrando GRUPOS (tickets) com reposicao."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    vals = []
    for _ in range(n):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in chosen])
        yb, sb = y[idx], score[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(metric_fn(yb, sb))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def precision_at_k(y, score, ks=(5, 10, 20)):
    order = np.argsort(-score)
    return {int(k): float(y[order[:k]].mean()) for k in ks if k <= len(y)}


# ---------- avaliacao principal ----------
def evaluate(cfg, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from .train import load_model
    emb_dir = Path(cfg.paths.emb_dir)
    rep_dir = Path(cfg.paths.reports_dir); rep_dir.mkdir(parents=True, exist_ok=True)

    tr = load_embeddings(emb_dir / "train.npz")
    va = load_embeddings(emb_dir / "val.npz")
    te = load_embeddings(emb_dir / "test.npz")
    model = load_model(Path(cfg.paths.models_dir) / "siamese_head.pt", device=device)

    # embeddings projetados
    z_tr, aux_tr = model_embeddings(model, tr["emb"], device)
    z_va, aux_va = model_embeddings(model, va["emb"], device)
    z_te, aux_te = model_embeddings(model, te["emb"], device)

    # prototipos do cluster LIMPO (so train, label 0)
    protos = fit_prototypes(z_tr[tr["label"] == 0], k=cfg.decision.k_prototypes, seed=cfg.seed)
    decider = PrototypeDecider(protos, threshold=0.0, target_precision=cfg.decision.target_precision)
    sp_va, sp_te = decider.scores(z_va), decider.scores(z_te)

    # fusao calibrada [score_proto, aux_logit] via LogReg na VAL (avaliada no test held-out)
    fus = LogisticRegression(max_iter=1000)
    Xv = np.stack([sp_va, aux_va], axis=1)
    fus.fit(Xv, va["label"])
    fused_va = fus.predict_proba(Xv)[:, 1]
    fused_te = fus.predict_proba(np.stack([sp_te, aux_te], axis=1))[:, 1]

    report: dict = {"n_test": int(len(te["label"])), "target_precision": cfg.decision.target_precision}

    # ---- 1) global: modelo vs baselines de confound ----
    def auroc_ap(y, s):
        return {"auroc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}

    glob = {
        "modelo_proto": auroc_ap(te["label"], sp_te),
        "modelo_aux": auroc_ap(te["label"], aux_te),
        "modelo_fusao": auroc_ap(te["label"], fused_te),
    }
    # baseline confound-only (treina na train, testa no test)
    Ctr, Cte = confound_matrix(tr), confound_matrix(te)
    # alinhar colunas (categorias podem diferir entre splits): usa so resolucao+aspect+is_clean_res (numericas, sempre presentes)
    def numeric_confound(z):
        res = _resolutions(z["path"]).astype(float)
        aspect = (res[:, 0] / np.maximum(res[:, 1], 1)).reshape(-1, 1)
        is_photo = (z["kind"] == "photo").astype(float).reshape(-1, 1)
        return np.concatenate([res, aspect, is_photo], axis=1)
    sc = StandardScaler()
    lr_conf = LogisticRegression(max_iter=2000).fit(sc.fit_transform(numeric_confound(tr)), tr["label"])
    s_conf = lr_conf.predict_proba(sc.transform(numeric_confound(te)))[:, 1]
    glob["baseline_confound_resaspect_photo"] = auroc_ap(te["label"], s_conf)
    # baseline resolucao trivial
    res_te = _resolutions(te["path"])
    s_res = (~((res_te[:, 0] == CLEAN_RES[0]) & (res_te[:, 1] == CLEAN_RES[1]))).astype(float)
    glob["baseline_resolucao_trivial"] = auroc_ap(te["label"], s_res)
    # baseline LogReg sobre DINOv2 cru (CLS)
    lr_raw = LogisticRegression(max_iter=2000).fit(normalize(tr["emb"]), tr["label"])
    s_raw = lr_raw.predict_proba(normalize(te["emb"]))[:, 1]
    glob["baseline_logreg_dinov2_cru"] = auroc_ap(te["label"], s_raw)
    # baseline one-class kNN sobre DINOv2 cru (so imagens limpas de treino)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(normalize(tr["emb"][tr["label"] == 0]))
    s_knn = nn.kneighbors(normalize(te["emb"]))[0].mean(axis=1)
    glob["baseline_oneclass_knn_dinov2"] = auroc_ap(te["label"], s_knn)
    # baseline FRACAO DE PADDING/CINZA (auditoria do modo 'pad': se isto separa as classes,
    # o padding reintroduziu o confound de aspect-ratio)
    from .geometry import pad_fraction
    pf_te = np.array([pad_fraction(int(w), int(h)) for w, h in res_te])
    if len(np.unique(pf_te)) > 1:
        glob["baseline_fracao_padding_cinza"] = auroc_ap(te["label"], pf_te)
    report["global_vs_baselines"] = glob

    # ---- 2) METRICA PRIMARIA: subconjunto controlado ----
    def controlled_mask(z):
        clean = z["label"] == 0
        ctrl_err = (z["label"] == 1) & (z["form_factor"] == "unfold") & \
                   (z["orientation"] == "portrait") & (z["kind"] == "screenshot")
        return clean | ctrl_err
    m = controlled_mask(te)
    if m.sum() > 0 and len(np.unique(te["label"][m])) == 2:
        report["primaria_subconjunto_controlado"] = {
            "n": int(m.sum()), "n_erro": int(te["label"][m].sum()),
            "modelo_fusao": auroc_ap(te["label"][m], fused_te[m]),
            "modelo_proto": auroc_ap(te["label"][m], sp_te[m]),
            "baseline_confound": auroc_ap(te["label"][m], s_conf[m]),
            "ci95_fusao_auroc": grouped_bootstrap_ci(roc_auc_score, te["label"][m], fused_te[m], te["group"][m]),
        }

    # ---- 3) deteccao SINTETICA livre de confound (clean-test vs synth-test) ----
    syn = load_embeddings(emb_dir / "test_synth.npz")
    z_syn, aux_syn = model_embeddings(model, syn["emb"], device)
    sp_syn = decider.scores(z_syn)
    clean_te = te["label"] == 0
    y_sy = np.concatenate([np.zeros(int(clean_te.sum())), np.ones(len(z_syn))])
    s_sy_proto = np.concatenate([sp_te[clean_te], sp_syn])
    fused_syn = fus.predict_proba(np.stack([sp_syn, aux_syn], axis=1))[:, 1]
    s_sy_fus = np.concatenate([fused_te[clean_te], fused_syn])
    report["sintetico_livre_de_confound"] = {
        "n_clean": int(clean_te.sum()), "n_synth": int(len(z_syn)),
        "modelo_proto": auroc_ap(y_sy, s_sy_proto),
        "modelo_fusao": auroc_ap(y_sy, s_sy_fus),
        "nota": "erros injetados nas proprias imagens limpas de teste (mesma resolucao) -> sem confound geometrico",
    }

    # ---- 4) auditoria same-resolution (erros reais 2076x2152, held-out val+test) ----
    sameres = []
    for z, sp, fu, aux in [(va, sp_va, fused_va, aux_va), (te, sp_te, fused_te, aux_te)]:
        res = _resolutions(z["path"])
        for i in range(len(z["label"])):
            if z["label"][i] == 1 and res[i, 0] == CLEAN_RES[0] and res[i, 1] == CLEAN_RES[1]:
                name = Path(str(z["path"][i])).name
                sameres.append({"file": name, "score_proto": float(sp[i]), "fused": float(fu[i]),
                                "independente": not name.startswith("Screenshot_")})
    report["auditoria_same_resolution"] = {
        "n": len(sameres), "itens": sameres,
        "nota": "Screenshot_* sao quase-duplicatas da sessao no_erros; 'independente'=True e o unico teste real anti-confound",
    }

    # ---- 5) falseabilidade ----
    # (a) o score do modelo prediz o CONFOUND de resolucao tao bem quanto o erro?
    res_clean_te = ((res_te[:, 0] == CLEAN_RES[0]) & (res_te[:, 1] == CLEAN_RES[1])).astype(int)
    placebo_y = 1 - res_clean_te  # 1 = resolucao nao-canonica (confound)
    falsi = {}
    if len(np.unique(placebo_y)) == 2:
        falsi["auroc_modelo_predizendo_resolucao"] = float(roc_auc_score(placebo_y, fused_te))
        falsi["auroc_modelo_predizendo_erro"] = glob["modelo_fusao"]["auroc"]
        falsi["interpretacao"] = ("Se os dois AUROC sao parecidos, o modelo rastreia resolucao, "
                                  "nao erro. Diferenca grande favorece deteccao de conteudo.")
    # (b) label-shuffle dentro do estrato de resolucao -> AUROC deve cair a ~0.5
    rng = np.random.default_rng(cfg.seed)
    y_shuf = tr["label"].copy()
    for val_res in [1, 0]:
        idx = np.where(((_resolutions(tr["path"])[:, 0] == CLEAN_RES[0]) &
                        (_resolutions(tr["path"])[:, 1] == CLEAN_RES[1])).astype(int) == val_res)[0]
        y_shuf[idx] = rng.permutation(y_shuf[idx])
    lr_sh = LogisticRegression(max_iter=2000).fit(normalize(tr["emb"]), y_shuf)
    s_sh = lr_sh.predict_proba(normalize(te["emb"]))[:, 1]
    falsi["auroc_label_shuffle_no_estrato"] = float(roc_auc_score(te["label"], s_sh))
    report["falseabilidade"] = falsi

    # ---- 6) limiar por precisao-alvo (fixado na VAL) + metricas no test ----
    thr_results = {}
    for target in (0.90, 0.95, 0.99):
        thr, info = select_threshold_for_precision(fused_va, va["label"], target)
        pred = (fused_te > thr).astype(int)
        tp = int(((pred == 1) & (te["label"] == 1)).sum())
        fp = int(((pred == 1) & (te["label"] == 0)).sum())
        fn = int(((pred == 0) & (te["label"] == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        thr_results[f"precisao_alvo_{target}"] = {
            "threshold": thr, "val": info,
            "test_precision": prec, "test_recall": rec,
            "test_tp": tp, "test_fp": fp, "test_fn": fn,
        }
    report["limiar_por_precisao"] = thr_results
    report["precision_at_k"] = precision_at_k(te["label"], fused_te)
    report["ci95_global_auroc_fusao"] = grouped_bootstrap_ci(
        roc_auc_score, te["label"], fused_te, te["group"])

    # ---- PONTO DE OPERACAO (objetivo do config) — metricas COMPLETAS p/ comparacao ----
    if cfg.decision.objective == "precision":
        op_thr, op_val = select_threshold_for_precision(fused_va, va["label"], cfg.decision.target_precision)
    else:  # f1 (balanceado) — padrao
        op_thr, op_val = select_threshold_max_f1(fused_va, va["label"])
    pred = (fused_te > op_thr).astype(int)
    y = te["label"]
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    report["ponto_operacao"] = {
        "objetivo": cfg.decision.objective, "threshold": float(op_thr),
        "acuracia": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precisao": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, fused_te)),
        "ap": float(average_precision_score(y, fused_te)),
        "confusao": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "ci95_acuracia": grouped_bootstrap_ci(accuracy_score, y, pred, te["group"]),
    }

    # ---- decision bundle (para inferencia self-contained) ----
    np.savez(
        Path(cfg.paths.models_dir) / "decision.npz",
        prototypes=protos,
        fusion_coef=fus.coef_.ravel(),
        fusion_intercept=np.array([fus.intercept_[0]]),
        threshold=np.array([op_thr]),
        use_patch_stats=np.array([int(cfg.backbone.use_patch_stats)]),
        size=np.array([cfg.backbone.size]),
        preprocess=np.array([cfg.backbone.preprocess]),
        target_precision=np.array([cfg.decision.target_precision]),
    )
    _confusion_plot(rep_dir, tp, tn, fp, fn, report["ponto_operacao"])

    # ---- plots ----
    _plots(rep_dir, te, sp_te, fused_te, va, fused_va, y_sy, s_sy_fus, thr_results)

    with open(rep_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def _confusion_plot(rep_dir, tp, tn, fp, fn, op):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np
    cm = _np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=22, color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["sem-erro", "erro"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["sem-erro", "erro"])
    ax.set_xlabel("predito"); ax.set_ylabel("real")
    ax.set_title(f"Matriz de confusao (TEST, ponto balanceado)\n"
                 f"ACC={op['acuracia']:.2f}  Precisao={op['precisao']:.2f}  "
                 f"Recall={op['recall']:.2f}  F1={op['f1']:.2f}")
    fig.tight_layout(); fig.savefig(rep_dir / "confusion_matrix.png", dpi=120); plt.close(fig)


def _plots(rep_dir, te, sp_te, fused_te, va, fused_va, y_sy, s_sy_fus, thr_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # distribuicao de score por classe (test)
    ax[0].hist(fused_te[te["label"] == 0], bins=20, alpha=0.6, label="sem-erro", color="tab:green")
    ax[0].hist(fused_te[te["label"] == 1], bins=20, alpha=0.6, label="erro", color="tab:red")
    ax[0].set_title("Score de fusao no TEST (real)"); ax[0].set_xlabel("p(erro)"); ax[0].legend()
    # PR curve
    p, r, _ = precision_recall_curve(te["label"], fused_te)
    ax[1].plot(r, p); ax[1].set_title("Precision-Recall (TEST real)")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision"); ax[1].set_ylim(0, 1.02)
    # sintetico livre de confound
    ax[2].hist(s_sy_fus[y_sy == 0], bins=20, alpha=0.6, label="limpo", color="tab:green")
    ax[2].hist(s_sy_fus[y_sy == 1], bins=20, alpha=0.6, label="erro sintetico", color="tab:red")
    ax[2].set_title("Sintetico livre de confound"); ax[2].set_xlabel("p(erro)"); ax[2].legend()
    fig.tight_layout(); fig.savefig(rep_dir / "evaluation_plots.png", dpi=110)
    plt.close(fig)
