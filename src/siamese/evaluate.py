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
                             accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef, brier_score_loss)

from .features import load_embeddings
from .decision import (fit_prototypes, PrototypeDecider,
                       select_threshold_for_precision, select_threshold_max_f1,
                       fit_category_prototypes, assign_category)
from .manifest import category_id, CATEGORY_TO_ID, ID_TO_CATEGORY, CATEGORIES, CLEAN_CATEGORY

CLEAN_RES = (2076, 2152)


def _aux_err(aux: np.ndarray, multiclass: bool) -> np.ndarray:
    """Score escalar de 'erro' a partir da cabeca auxiliar.
    binario: o proprio logit. multi-classe: 1 - softmax[clean] = P(tem erro)."""
    if multiclass:
        e = np.exp(aux - aux.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return 1.0 - p[:, 0]
    return aux


def _cat_ids(z: dict) -> np.ndarray:
    """Vetor de id de categoria (0=clean,1..N) a partir da coluna 'category' do npz.
    Categoria fora do escopo levanta (nunca vira 'clean' silenciosamente — Fase 6)."""
    return np.array([category_id(str(c)) for c in z.get("category", [])], dtype=int)


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


def _expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE: |acuracia - confianca| media ponderada por bin de probabilidade (Fase 5)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(ece)


# ---------- avaliacao principal ----------
def evaluate(cfg, device: str | None = None, final_test: bool = False) -> dict:
    """Avalia o detector. Por PADRAO (final_test=False) reporta sobre a VALIDACAO (modo DEV):
    o TESTE NAO e' tocado — assim grid_search/ablation/visualize ficam livres de snooping. O
    TESTE held-out so e' avaliado com final_test=True, que exige a trava liberada (apenas
    `scripts/evaluate.py --final-test`, UMA vez, apos congelar a config). A fusao logistica e
    o limiar sao SEMPRE calibrados na VAL (set de calibracao); em modo DEV o conjunto de
    relatorio TAMBEM e' a val (in-sample — sinalizado em report['_modo']); a unica metrica
    DEV honesta e' o gate sintetico livre de confound (prototipos vem do TRAIN)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from .train import load_model
    from .protocol import test_access_allowed, TestSetAccessError
    emb_dir = Path(cfg.paths.emb_dir)
    rep_dir = Path(cfg.paths.reports_dir); rep_dir.mkdir(parents=True, exist_ok=True)

    tr = load_embeddings(emb_dir / "train.npz")
    va = load_embeddings(emb_dir / "val.npz")                 # set de CALIBRACAO (fusao + limiar)
    if final_test:
        if not test_access_allowed():
            raise TestSetAccessError(
                "evaluate(final_test=True) exige a trava liberada. Use `scripts/evaluate.py "
                "--final-test` (uma unica vez, apos congelar arquitetura/pesos/limiares).")
        ho = load_embeddings(emb_dir / "test.npz")            # HELD-OUT real
        ho_name, ho_syn_path = "test", emb_dir / "test_synth.npz"
    else:
        ho = va                                               # DEV: relatorio sobre a propria val
        ho_name, ho_syn_path = "val", emb_dir / "val_synth.npz"
    model = load_model(Path(cfg.paths.models_dir) / "siamese_head.pt", device=device)
    multiclass = getattr(model, "num_classes", 1) > 1

    # embeddings projetados
    z_tr, aux_tr = model_embeddings(model, tr["emb"], device)
    z_va, aux_va = model_embeddings(model, va["emb"], device)
    z_ho, aux_ho = model_embeddings(model, ho["emb"], device)
    # score escalar de erro da cabeca auxiliar (binario: logit; multi-classe: 1-P(clean))
    ae_tr, ae_va, ae_ho = (_aux_err(aux_tr, multiclass), _aux_err(aux_va, multiclass),
                           _aux_err(aux_ho, multiclass))

    # ESTAGIO 1 — prototipos do cluster LIMPO (so train, label 0) -> gate "tem erro?"
    protos = fit_prototypes(z_tr[tr["label"] == 0], k=cfg.decision.k_prototypes, seed=cfg.seed)
    decider = PrototypeDecider(protos, threshold=0.0, target_precision=cfg.decision.target_precision)
    sp_tr, sp_va, sp_ho = decider.scores(z_tr), decider.scores(z_va), decider.scores(z_ho)

    # fusao calibrada [score_proto, aux_err] via LogReg na VAL (avaliada no held-out)
    fus = LogisticRegression(max_iter=1000)
    Xv = np.stack([sp_va, ae_va], axis=1)
    fus.fit(Xv, va["label"])
    fused_va = fus.predict_proba(Xv)[:, 1]
    fused_ho = fus.predict_proba(np.stack([sp_ho, ae_ho], axis=1))[:, 1]
    fused_tr = fus.predict_proba(np.stack([sp_tr, ae_tr], axis=1))[:, 1]   # p/ metricas de TREINO

    modo = ("FINAL (teste held-out; calibrado na val)" if final_test
            else "DEV (val in-sample — NAO held-out; numeros vinculantes so via --final-test)")
    report: dict = {"n_test": int(len(ho["label"])), "multiclass": bool(multiclass),
                    "target_precision": cfg.decision.target_precision,
                    "_modo": modo, "_holdout": ho_name}

    # ---- 1) global: modelo vs baselines de confound ----
    def auroc_ap(y, s):
        return {"auroc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}

    glob = {
        "modelo_proto": auroc_ap(ho["label"], sp_ho),
        "modelo_aux": auroc_ap(ho["label"], ae_ho),
        "modelo_fusao": auroc_ap(ho["label"], fused_ho),
    }
    # baseline confound-only (treina na train, testa no held-out): usa so numericas (resolucao+
    # aspect+is_photo, sempre presentes; categorias one-hot podem diferir entre splits).
    def numeric_confound(z):
        res = _resolutions(z["path"]).astype(float)
        aspect = (res[:, 0] / np.maximum(res[:, 1], 1)).reshape(-1, 1)
        is_photo = (z["kind"] == "photo").astype(float).reshape(-1, 1)
        return np.concatenate([res, aspect, is_photo], axis=1)
    sc = StandardScaler()
    lr_conf = LogisticRegression(max_iter=2000).fit(sc.fit_transform(numeric_confound(tr)), tr["label"])
    s_conf = lr_conf.predict_proba(sc.transform(numeric_confound(ho)))[:, 1]
    glob["baseline_confound_resaspect_photo"] = auroc_ap(ho["label"], s_conf)
    # baseline resolucao trivial
    res_ho = _resolutions(ho["path"])
    s_res = (~((res_ho[:, 0] == CLEAN_RES[0]) & (res_ho[:, 1] == CLEAN_RES[1]))).astype(float)
    glob["baseline_resolucao_trivial"] = auroc_ap(ho["label"], s_res)
    # baseline LogReg sobre DINOv2 cru (CLS)
    lr_raw = LogisticRegression(max_iter=2000).fit(normalize(tr["emb"]), tr["label"])
    s_raw = lr_raw.predict_proba(normalize(ho["emb"]))[:, 1]
    glob["baseline_logreg_dinov2_cru"] = auroc_ap(ho["label"], s_raw)
    # baseline one-class kNN sobre DINOv2 cru (so imagens limpas de treino)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(normalize(tr["emb"][tr["label"] == 0]))
    s_knn = nn.kneighbors(normalize(ho["emb"]))[0].mean(axis=1)
    glob["baseline_oneclass_knn_dinov2"] = auroc_ap(ho["label"], s_knn)
    # baseline FRACAO DE PADDING/CINZA (auditoria do modo 'pad': se isto separa as classes,
    # o padding reintroduziu o confound de aspect-ratio)
    from .geometry import pad_fraction
    pf_ho = np.array([pad_fraction(int(w), int(h)) for w, h in res_ho])
    if len(np.unique(pf_ho)) > 1:
        glob["baseline_fracao_padding_cinza"] = auroc_ap(ho["label"], pf_ho)
    report["global_vs_baselines"] = glob

    # ---- 2) METRICA PRIMARIA: subconjunto controlado ----
    def controlled_mask(z):
        clean = z["label"] == 0
        ctrl_err = (z["label"] == 1) & (z["form_factor"] == "unfold") & \
                   (z["orientation"] == "portrait") & (z["kind"] == "screenshot")
        return clean | ctrl_err
    m = controlled_mask(ho)
    if m.sum() > 0 and len(np.unique(ho["label"][m])) == 2:
        report["primaria_subconjunto_controlado"] = {
            "n": int(m.sum()), "n_erro": int(ho["label"][m].sum()),
            "modelo_fusao": auroc_ap(ho["label"][m], fused_ho[m]),
            "modelo_proto": auroc_ap(ho["label"][m], sp_ho[m]),
            "baseline_confound": auroc_ap(ho["label"][m], s_conf[m]),
            "ci95_fusao_auroc": grouped_bootstrap_ci(roc_auc_score, ho["label"][m], fused_ho[m], ho["group"][m]),
        }

    # ---- 3) deteccao SINTETICA livre de confound (clean-holdout vs synth-holdout) ----
    # Metrica DEV honesta: prototipos vem do TRAIN, entao mesmo com ho=val nao ha resubstituicao
    # no score de prototipo. So roda se synthetic.enabled e o arquivo de sonda existir.
    y_sy = s_sy_fus = None
    if cfg.synthetic.enabled and ho_syn_path.exists():
        syn = load_embeddings(ho_syn_path)
        z_syn, aux_syn = model_embeddings(model, syn["emb"], device)
        ae_syn = _aux_err(aux_syn, multiclass)
        sp_syn = decider.scores(z_syn)
        clean_ho = ho["label"] == 0
        y_sy = np.concatenate([np.zeros(int(clean_ho.sum())), np.ones(len(z_syn))])
        s_sy_proto = np.concatenate([sp_ho[clean_ho], sp_syn])
        fused_syn = fus.predict_proba(np.stack([sp_syn, ae_syn], axis=1))[:, 1]
        s_sy_fus = np.concatenate([fused_ho[clean_ho], fused_syn])
        report["sintetico_livre_de_confound"] = {
            "n_clean": int(clean_ho.sum()), "n_synth": int(len(z_syn)),
            "modelo_proto": auroc_ap(y_sy, s_sy_proto),
            "modelo_fusao": auroc_ap(y_sy, s_sy_fus),
            "nota": f"erros injetados nas limpas de {ho_name} (mesma resolucao) -> sem confound geometrico",
        }
    else:
        report["sintetico_livre_de_confound"] = {
            "n_clean": int((ho["label"] == 0).sum()), "n_synth": 0,
            "nota": "sonda sintetica indisponivel (synthetic.enabled=false ou arquivo ausente)",
        }

    # ---- 4) auditoria same-resolution (erros reais 2076x2152, held-out) ----
    audit_pairs = [(va, sp_va, fused_va, aux_va)]
    if final_test:                                      # em DEV, ho==va -> evita duplicar
        audit_pairs.append((ho, sp_ho, fused_ho, aux_ho))
    sameres = []
    for z, sp, fu, aux in audit_pairs:
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
    res_clean_ho = ((res_ho[:, 0] == CLEAN_RES[0]) & (res_ho[:, 1] == CLEAN_RES[1])).astype(int)
    placebo_y = 1 - res_clean_ho  # 1 = resolucao nao-canonica (confound)
    falsi = {}
    if len(np.unique(placebo_y)) == 2:
        falsi["auroc_modelo_predizendo_resolucao"] = float(roc_auc_score(placebo_y, fused_ho))
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
    s_sh = lr_sh.predict_proba(normalize(ho["emb"]))[:, 1]
    falsi["auroc_label_shuffle_no_estrato"] = float(roc_auc_score(ho["label"], s_sh))
    report["falseabilidade"] = falsi

    # ---- 6) limiar por precisao-alvo (fixado na VAL) + metricas no held-out ----
    thr_results = {}
    for target in (0.90, 0.95, 0.99):
        thr, info = select_threshold_for_precision(fused_va, va["label"], target)
        pred = (fused_ho > thr).astype(int)
        tp = int(((pred == 1) & (ho["label"] == 1)).sum())
        fp = int(((pred == 1) & (ho["label"] == 0)).sum())
        fn = int(((pred == 0) & (ho["label"] == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        thr_results[f"precisao_alvo_{target}"] = {
            "threshold": thr, "val": info,
            "test_precision": prec, "test_recall": rec,
            "test_tp": tp, "test_fp": fp, "test_fn": fn,
        }
    report["limiar_por_precisao"] = thr_results
    report["precision_at_k"] = precision_at_k(ho["label"], fused_ho)
    report["ci95_global_auroc_fusao"] = grouped_bootstrap_ci(
        roc_auc_score, ho["label"], fused_ho, ho["group"])

    # ---- PONTO DE OPERACAO (objetivo do config) — metricas COMPLETAS em TREINO e HELD-OUT ----
    # O limiar e' SEMPRE escolhido na VAL (calibracao) e aplicado em ambos. Em modo FINAL o
    # held-out e' o teste; em DEV e' a propria val (in-sample — ver report['_modo']).
    if cfg.decision.objective == "precision":
        op_thr, op_val = select_threshold_for_precision(fused_va, va["label"], cfg.decision.target_precision)
    else:  # f1 (balanceado) — padrao
        op_thr, op_val = select_threshold_max_f1(fused_va, va["label"])

    def _op_metrics(y, fused, groups):
        pred = (fused > op_thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        return {
            "objetivo": cfg.decision.objective, "threshold": float(op_thr), "n": int(len(y)),
            "acuracia": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precisao": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) == 2 else float("nan"),
            "especificidade": float(tn / (tn + fp)) if tn + fp else 0.0,
            "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
            "auroc": float(roc_auc_score(y, fused)) if len(np.unique(y)) == 2 else float("nan"),
            "ap": float(average_precision_score(y, fused)) if len(np.unique(y)) == 2 else float("nan"),
            "brier": float(brier_score_loss(y, fused)) if len(np.unique(y)) == 2 else float("nan"),
            "ece": _expected_calibration_error(y, fused) if len(np.unique(y)) == 2 else float("nan"),
            "confusao": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "ci95_acuracia": grouped_bootstrap_ci(accuracy_score, y, pred, groups),
            "ci95_f1": grouped_bootstrap_ci(lambda yy, pp: f1_score(yy, (pp > 0.5).astype(int), zero_division=0),
                                            y, pred.astype(float), groups),
        }

    report["ponto_operacao"] = _op_metrics(ho["label"], fused_ho, ho["group"])         # HELD-OUT
    report["ponto_operacao_treino"] = _op_metrics(tr["label"], fused_tr, tr["group"])  # TREINO (in-sample)
    op = report["ponto_operacao"]
    tp, tn, fp, fn = op["confusao"]["TP"], op["confusao"]["TN"], op["confusao"]["FP"], op["confusao"]["FN"]

    # ---- ESTAGIO 2: clusterizacao por CATEGORIA (multi-cluster) — em HELD-OUT e TREINO ----
    cat_protos = cat_proto_ids = None
    clean_id = CATEGORY_TO_ID[CLEAN_CATEGORY]
    if multiclass:
        cat_tr = _cat_ids(tr)
        err_tr = cat_tr != clean_id
        cat_protos, cat_proto_ids = fit_category_prototypes(
            z_tr[err_tr], cat_tr[err_tr], k=cfg.decision.k_prototypes, seed=cfg.seed)
        err_class_ids = sorted({int(c) for c in cat_proto_ids}) if cat_protos is not None else []

        def _stage2(z_split, aux_split, cat_split):
            err = cat_split != clean_id
            if err.sum() == 0 or cat_protos is None:
                return None
            y_true = cat_split[err]
            y_proto = assign_category(z_split[err], cat_protos, cat_proto_ids)            # clusterizacao
            y_aux = np.array(err_class_ids)[aux_split[err][:, err_class_ids].argmax(axis=1)]  # aux head
            present = sorted(set(y_true.tolist()) | set(y_proto.tolist()) | set(y_aux.tolist()))

            def _m(yp):
                f1c = f1_score(y_true, yp, labels=present, average=None, zero_division=0)
                return {
                    "accuracy": float(accuracy_score(y_true, yp)),
                    "f1_macro": float(f1_score(y_true, yp, labels=present, average="macro", zero_division=0)),
                    "f1_por_classe": {ID_TO_CATEGORY[i]: float(f) for i, f in zip(present, f1c)},
                    "confusion_matrix": confusion_matrix(y_true, yp, labels=present).tolist(),
                }
            return {"n_erro": int(err.sum()), "classes": [ID_TO_CATEGORY[i] for i in present],
                    "por_prototipo": _m(y_proto), "por_aux_head": _m(y_aux),
                    "_y_true": y_true, "_y_proto": y_proto}

        s2_ho = _stage2(z_ho, aux_ho, _cat_ids(ho))
        s2_tr = _stage2(z_tr, aux_tr, cat_tr)
        if s2_ho:
            report["estagio2_categoria"] = {k: v for k, v in s2_ho.items() if not k.startswith("_")}
            report["estagio2_categoria"]["nota"] = ("categoria atribuida SO a imagens de ERRO. "
                "'por_prototipo' = clusterizacao no espaco z (protótipo mais próximo).")
            _confusion_plot_multiclass(rep_dir, s2_ho["_y_true"], s2_ho["_y_proto"],
                                       s2_ho["classes"], s2_ho["por_prototipo"]["f1_macro"], suffix="")
        if s2_tr:
            report["estagio2_categoria_treino"] = {k: v for k, v in s2_tr.items() if not k.startswith("_")}
            _confusion_plot_multiclass(rep_dir, s2_tr["_y_true"], s2_tr["_y_proto"],
                                       s2_tr["classes"], s2_tr["por_prototipo"]["f1_macro"], suffix="_treino")

    # ---- decision bundle (para inferencia self-contained) — SEMPRE calibrado na val ----
    d = z_tr.shape[1]
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
        multiclass=np.array([int(multiclass)]),
        categories=np.array(CATEGORIES if multiclass else ["clean", "error"]),
        cat_prototypes=(cat_protos if cat_protos is not None
                        else np.zeros((0, d), dtype=np.float32)),
        cat_proto_ids=(cat_proto_ids if cat_proto_ids is not None
                       else np.zeros((0,), dtype=int)),
    )
    _confusion_plot(rep_dir, tp, tn, fp, fn, report["ponto_operacao"], split_name=ho_name.upper(), suffix="")
    opt = report["ponto_operacao_treino"]; ct = opt["confusao"]
    _confusion_plot(rep_dir, ct["TP"], ct["TN"], ct["FP"], ct["FN"], opt,
                    split_name="TRAIN", suffix="_treino")

    # ---- plots ----
    _plots(rep_dir, ho, sp_ho, fused_ho, va, fused_va, y_sy, s_sy_fus, thr_results)

    out_name = "evaluation_report.json" if final_test else "evaluation_report_dev.json"
    with open(rep_dir / out_name, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def _confusion_plot(rep_dir, tp, tn, fp, fn, op, split_name="TEST", suffix=""):
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
    ax.set_title(f"Matriz de confusao ({split_name}, ponto balanceado)\n"
                 f"ACC={op['acuracia']:.2f}  Precisao={op['precisao']:.2f}  "
                 f"Recall={op['recall']:.2f}  F1={op['f1']:.2f}")
    fig.tight_layout(); fig.savefig(rep_dir / f"confusion_matrix{suffix}.png", dpi=120); plt.close(fig)


def _confusion_plot_multiclass(rep_dir, y_true, y_pred, names, f1_macro, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true.tolist()) | set(y_pred.tolist())))
    n = len(names)
    fig, ax = plt.subplots(figsize=(1.3 * n + 2, 1.3 * n + 1.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="black" if cm[i, j] < cm.max() / 2 else "white")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("categoria predita (protótipo)"); ax.set_ylabel("categoria real")
    ax.set_title(f"Estágio 2 — matriz de confusão por categoria (TEST)\nF1 macro = {f1_macro:.2f}")
    fig.tight_layout(); fig.savefig(rep_dir / f"confusion_matrix_categoria{suffix}.png", dpi=120); plt.close(fig)


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
    # sintetico livre de confound (pode estar ausente se synthetic.enabled=false)
    if y_sy is not None and s_sy_fus is not None:
        ax[2].hist(s_sy_fus[y_sy == 0], bins=20, alpha=0.6, label="limpo", color="tab:green")
        ax[2].hist(s_sy_fus[y_sy == 1], bins=20, alpha=0.6, label="erro sintetico", color="tab:red")
        ax[2].set_title("Sintetico livre de confound"); ax[2].set_xlabel("p(erro)"); ax[2].legend()
    else:
        ax[2].text(0.5, 0.5, "sonda sintetica\nindisponivel", ha="center", va="center")
        ax[2].set_axis_off()
    fig.tight_layout(); fig.savefig(rep_dir / "evaluation_plots.png", dpi=110)
    plt.close(fig)
