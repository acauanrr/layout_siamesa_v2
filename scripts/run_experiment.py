#!/usr/bin/env python
"""EXPERIMENTO COMPLETO E DEFINITIVO — um comando, do dado bruto ao resultado.

Roda TODO o pipeline no dataset corrigido (data/input -> data/processed = fonte da verdade) e
produz UM relatorio claro com as metricas que voce leva para a apresentacao:

    python scripts/run_experiment.py            # roda tudo e gera o relatorio
    python scripts/run_experiment.py --fresh    # reconstroi do zero (splits/processed/embeddings)

Etapas (idempotentes — pula o que ja existe, a menos que --fresh):
    1. split agrupado/estratificado (sem vazamento)        scripts/build_splits.py
    2. materializa data/processed/ (fonte da verdade)      scripts/export_processed.py
    3. embeddings DINOv2 (congelado, cacheados)            scripts/extract_features.py
    4. sondas sinteticas + reflow (anti-confound)          scripts/make_synthetic.py
    5. TREINO da cabeca siamesa                            scripts/train.py
    6. AVALIACAO DEV (validacao — iteracao honesta)        scripts/evaluate.py
    7. TESTE held-out (1x, numeros vinculantes)            scripts/evaluate.py --final-test
    8. RELATORIO consolidado -> artifacts/reports/EXPERIMENT_RESULTS.{md,json}

Saidas para a apresentacao:
    artifacts/reports/EXPERIMENT_RESULTS.md     <- tabela pronta + veredito honesto (LEIA ESTE)
    artifacts/reports/EXPERIMENT_RESULTS.json   <- metricas planas (p/ comparar com outros modelos)
    artifacts/reports/confusion_matrix.png      <- matriz de confusao (gate, teste)
    artifacts/reports/confusion_matrix_categoria.png  <- matriz por categoria (Estagio 2)

IMPORTANTE (comparacao justa): este dataset PODE ter um CONFOUND de resolucao (limpas
concentradas em poucas resolucoes de device). A regra trivial "resolucao nao-canonica => erro"
sozinha da AUROC = baseline_resolucao_trivial (~1.0 = confound forte; ~0.5 = quebrado). Quando
alta, a metrica GLOBAL e' majoritariamente trapaca. O relatorio mostra as metricas-padrao E as
metricas LIVRES DE CONFOUND (as honestas) E um VEREDITO automatico de "funciona ou nao".
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(step: str, argv: list[str], *, env_note: str = "") -> None:
    """Executa um passo do pipeline; mostra status limpo, aborta com erro claro."""
    print(f"\n\033[1m[{step}]\033[0m {' '.join(str(a) for a in argv[1:])} {env_note}")
    t0 = time.time()
    p = subprocess.run([PY, *argv], cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode != 0:
        print(f"  \033[31mFALHOU ({dt:.1f}s)\033[0m — saida:")
        tail = (p.stdout + "\n" + p.stderr).strip().splitlines()
        print("    " + "\n    ".join(tail[-25:]))
        sys.exit(f"\nExperimento abortado no passo: {step}")
    # mostra so as linhas informativas (sem warnings)
    lines = [l for l in p.stdout.splitlines()
             if l.strip() and not any(w in l for w in
             ("Warning", "warn(", "HF_TOKEN", "ConvergenceWarning", "n_iter", "FutureWarning"))]
    for l in lines[-6:]:
        print(f"    {l}")
    print(f"  \033[32mok\033[0m ({dt:.1f}s)")


def _exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


# ----------------------------- relatorio consolidado -----------------------------
def _g(d: dict, *keys, default=None):
    x = d
    for k in keys:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x


def _f(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _ci(pair):
    if not pair or pair[0] != pair[0]:  # None / NaN
        return ""
    return f"[{_f(pair[0],2)}–{_f(pair[1],2)}]"


# textos do grafico por idioma (os nomes das classes sao slugs do dataset -> nao traduzidos)
_CHART_I18N = {
    "pt": {"suptitle": "Métricas por classe de erro (held-out)  ·  melhor detectada: {bd}  ·  "
                       "melhor classificada: {bc}  ·  ⚠ n<5 = suporte insuficiente (não confiável)",
           "left": "Estágio 1 — DETECÇÃO por classe (o gate pega o erro?)",
           "right": "Estágio 2 — CLASSIFICAÇÃO por classe (acerta a categoria?)",
           "s_recall": "recall@op (detecta?)", "s_auroc": "AUROC vs limpo",
           "s_prec": "precisão", "s_rec": "recall", "s_f1": "F1", "ylab": "métrica"},
    "en": {"suptitle": "Per-class error metrics (held-out)  ·  best detected: {bd}  ·  "
                       "best classified: {bc}  ·  ⚠ n<5 = insufficient support (unreliable)",
           "left": "Stage 1 — DETECTION per class (does the gate catch the error?)",
           "right": "Stage 2 — CLASSIFICATION per class (correct category?)",
           "s_recall": "recall@op (detected?)", "s_auroc": "AUROC vs clean",
           "s_prec": "precision", "s_rec": "recall", "s_f1": "F1", "ylab": "metric"},
}


def _per_class_chart(por_categoria: dict, ranking: dict, rep_dir: Path, *,
                     lang: str = "pt", outfiles: list[Path] | None = None) -> list[str]:
    """Gráfico de barras por classe (detecção × classificação). Salva em cada caminho de
    `outfiles` (PNG raster ou PDF/SVG vetorial — inferido pela extensão). `lang` = 'pt'|'en'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    T = _CHART_I18N[lang]

    cats = sorted(por_categoria, key=lambda c: -(por_categoria[c].get("n_test") or 0))
    if not cats:
        return []
    n_sup = [por_categoria[c].get("n_test") or 0 for c in cats]
    val = lambda c, k: (por_categoria[c].get(k) if por_categoria[c].get(k) is not None else 0.0)
    det_recall = [val(c, "deteccao_recall") for c in cats]
    det_auroc = [val(c, "auroc_vs_limpo") for c in cats]
    cls_prec = [val(c, "classif_precisao") for c in cats]
    cls_rec = [val(c, "classif_recall") for c in cats]
    cls_f1 = [val(c, "classif_f1") for c in cats]

    x = np.arange(len(cats))
    labels = [f"{c}\n(n={n}{' ⚠' if n < 5 else ''})" for c, n in zip(cats, n_sup)]
    alpha = [1.0 if n >= 5 else 0.45 for n in n_sup]   # esmaece classes de suporte pequeno

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    def _bars(ax, series, title):
        m = len(series)
        w = 0.8 / m
        for j, (name, vals, color) in enumerate(series):
            xs = x + (j - (m - 1) / 2) * w
            bars = ax.bar(xs, vals, w, label=name, color=color)
            for b, v, a in zip(bars, vals, alpha):
                b.set_alpha(a)
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, alpha=max(a, 0.6))
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.08); ax.set_ylabel(T["ylab"]); ax.set_title(title, fontsize=12, weight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.grid(axis="y", alpha=0.25)

    _bars(axL, [(T["s_recall"], det_recall, "#1f77b4"), (T["s_auroc"], det_auroc, "#7fb1d8")], T["left"])
    _bars(axR, [(T["s_prec"], cls_prec, "#2e7d32"), (T["s_rec"], cls_rec, "#f0a030"),
                (T["s_f1"], cls_f1, "#66bb6a")], T["right"])

    fig.suptitle(T["suptitle"].format(bd=ranking.get("melhor_detectado"),
                                      bc=ranking.get("melhor_classificado")), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outs = outfiles or [rep_dir / "metricas_por_classe.png"]
    for out in outs:
        fig.savefig(out, dpi=140)   # .pdf/.svg -> vetorial; .png -> raster (dpi)
    plt.close(fig)
    return [str(o) for o in outs]


def build_report(emb_dir: Path, rep_dir: Path, cfg_path: Path) -> dict:
    """Le evaluation_report.json (held-out) + _dev e monta o relatorio de apresentacao."""
    final = json.loads((rep_dir / "evaluation_report.json").read_text())
    op = final["ponto_operacao"]
    glob = final["global_vs_baselines"]
    conf = final["confusao"] if "confusao" in final else op["confusao"]
    ctrl = final.get("primaria_subconjunto_controlado", {})
    synth = final.get("sintetico_livre_de_confound", {})
    fals = final.get("falseabilidade", {})
    cc = final.get("calibracao_comparacao", {})
    e2 = final.get("estagio2_categoria", {})

    n_test = final["n_test"]
    n_err = int(conf["TP"] + conf["FN"])
    n_clean = int(conf["TN"] + conf["FP"])

    # --- numeros-chave ---
    proto_auroc = _g(glob, "modelo_proto", "auroc")
    fus_auroc = _g(glob, "modelo_fusao", "auroc")
    res_trivial = _g(glob, "baseline_resolucao_trivial", "auroc")
    ctrl_proto = _g(ctrl, "modelo_proto", "auroc")
    ctrl_base = _g(ctrl, "baseline_confound", "auroc")
    ctrl_ci = _g(ctrl, "ci95_fusao_auroc")
    synth_proto = _g(synth, "modelo_proto", "auroc")
    synth_ap = _g(synth, "modelo_proto", "ap")
    fa_err = _g(fals, "auroc_modelo_predizendo_erro")
    fa_res = _g(fals, "auroc_modelo_predizendo_resolucao")
    # chave da taxonomia FINA conforme o método CANÔNICO do Estágio 2 (prototype|knn|aux)
    _canon = _g(e2, "oraculo", "metodo_canonico", default="prototype")
    _fk = {"prototype": "por_prototipo", "knn": "por_knn", "aux": "por_aux_head"}.get(_canon, "por_prototipo")
    e2_coarse = _g(e2, "oraculo", "grossa", "f1_macro")
    e2_coarse_ci = _g(e2, "oraculo", "grossa", "ci95_f1_macro")
    e2_fine = _g(e2, "oraculo", "fina", _fk, "f1_macro")
    e2_cond = _g(e2, "condicional_ao_gate", "grossa", "f1_macro")
    # contagem DINAMICA de classes do Estagio 2 (deriva do dataset, nao hardcode): o Estagio 2 so
    # categoriza ERROS -> grossa = super-classes de erro (ex.: 2), fina = classes de erro (ex.: 4).
    n_coarse = len(_g(e2, "oraculo", "grossa", "classes", default=[]) or [])
    n_fine = len(_g(e2, "oraculo", "fina", _fk, "classes", default=[]) or [])
    chance_fine = (1.0 / n_fine) if n_fine else 0.0

    # --- VEREDITO automatico (data-driven, honesto) ---
    verdict = []
    supera_ctrl = (ctrl_proto is not None and ctrl_base is not None and ctrl_proto > ctrl_base)
    sinal_synth = (synth_proto is not None and synth_proto >= 0.65)
    if supera_ctrl and sinal_synth:
        verdict.append("✅ REAL CONTENT SIGNAL: on the CONTROLLED subset (form-factor/orientation "
                       f"fixed) the model (prototype AUROC {_f(ctrl_proto)}) BEATS the confound baseline "
                       f"({_f(ctrl_base)}); and on the CONFOUND-FREE synthetic it reaches AUROC "
                       f"{_f(synth_proto)} (AP {_f(synth_ap)}). The model detects the ERROR, not just the device.")
    else:
        verdict.append("⚠️ WEAK/INCONCLUSIVE content signal in the confound-free regime "
                       f"(controlled prototype {_f(ctrl_proto)} vs confound {_f(ctrl_base)}; "
                       f"synthetic {_f(synth_proto)}).")
    if fa_err is not None and fa_res is not None:
        gap = fa_err - fa_res
        if abs(gap) < 0.05:
            verdict.append(f"⚠️ CONFOUND NOT BEATEN globally: the score predicts ERROR ({_f(fa_err)}) "
                           f"about as well as RESOLUTION ({_f(fa_res)}) (gap {_f(gap)}). The confound is "
                           "ATTENUATED, not eliminated — beating it needs more diverse CLEAN screens (data).")
        else:
            verdict.append(f"✅ FALSIFIABILITY: predicts ERROR ({_f(fa_err)}) better than RESOLUTION "
                           f"({_f(fa_res)}) (gap {_f(gap)}).")
    verdict.append(f"ℹ️ The GLOBAL metric is confounded: the trivial resolution rule alone gives AUROC "
                   f"{_f(res_trivial)} — so the model's global accuracy must NOT be compared naively with "
                   "models that exploit the confound. Lead with confound-free AUROC.")

    # --- MÉTRICAS POR CLASSE DE ERRO (pedido do coordenador) ---
    det = _g(final, "deteccao_por_categoria", "por_classe", default={}) or {}
    e2fine = _g(e2, "oraculo", "fina", _fk, default={}) or {}   # método canônico (prototype|knn|aux)
    prec_c = e2fine.get("precisao_por_classe", {})
    rec_c = e2fine.get("recall_por_classe", {})
    f1_c = e2fine.get("f1_por_classe", {})
    sup_c = e2fine.get("suporte_por_classe", {})
    cats = sorted(set(det) | set(f1_c))
    por_categoria = {}
    for c in cats:
        d = det.get(c, {})
        por_categoria[c] = {
            "n_test": d.get("n", sup_c.get(c)),
            # Estágio 1 — detecção
            "deteccao_recall": d.get("deteccao_recall_no_limiar"),
            "auroc_vs_limpo": d.get("auroc_vs_limpo_proto"),
            "auroc_vs_limpo_ci95": d.get("ci95_auroc_proto"),
            "p_erro_medio": d.get("p_erro_medio"),
            # Estágio 2 — classificação
            "classif_precisao": prec_c.get(c),
            "classif_recall": rec_c.get(c),
            "classif_f1": f1_c.get(c),
        }
    # rankings (ignora classes com suporte < 5 no destaque, mas reporta todas)
    robust = {c: v for c, v in por_categoria.items() if (v["n_test"] or 0) >= 5}
    best_det = max(robust, key=lambda c: (robust[c]["auroc_vs_limpo"] or 0), default=None)
    worst_det = min(robust, key=lambda c: (robust[c]["auroc_vs_limpo"] or 1), default=None)
    best_cls = max(robust, key=lambda c: (robust[c]["classif_f1"] or 0), default=None)

    flat = {
        "n_test": n_test, "n_clean": n_clean, "n_erro": n_err,
        "por_categoria": por_categoria,
        "ranking": {"melhor_detectado": best_det, "pior_detectado": worst_det,
                    "melhor_classificado": best_cls,
                    "nota": "ranking entre classes com suporte >= 5"},
        "ponto_operacao": {k: op.get(k) for k in
            ("acuracia", "precisao", "recall", "f1", "balanced_accuracy", "mcc",
             "especificidade", "fpr", "auroc", "ap", "brier", "ece", "threshold")},
        "confusao": conf,
        "ci95_acuracia": op.get("ci95_acuracia"), "ci95_f1": op.get("ci95_f1"),
        "ci95_precisao": op.get("ci95_precisao"), "ci95_especificidade": op.get("ci95_especificidade"),
        "auroc_gate_prototipo": proto_auroc, "auroc_gate_fusao": fus_auroc,
        "sintetico_livre_confound": {"auroc": synth_proto, "ap": synth_ap},
        "controlado": {"modelo_prototipo": ctrl_proto, "baseline_confound": ctrl_base, "ci95": ctrl_ci},
        "falseabilidade": {"prediz_erro": fa_err, "prediz_resolucao": fa_res},
        "baseline_resolucao_trivial_auroc": res_trivial,
        "estagio2": {"grossa_f1_macro": e2_coarse, "grossa_ci95": e2_coarse_ci,
                     "grossa_condicional_gate": e2_cond, "fina_f1_macro": e2_fine},
        "calibracao_comparacao": {m: {k: cc[m].get(k) for k in
            ("especificidade", "recall", "balanced_accuracy", "fpr")}
            for m in cc if not m.startswith("_")},
    }
    (rep_dir / "EXPERIMENT_RESULTS.json").write_text(json.dumps(flat, indent=2, ensure_ascii=False))
    # gráfico por classe: PT e EN, em PNG (raster) e PDF (vetorial p/ projetor/paper)
    _per_class_chart(por_categoria, flat["ranking"], rep_dir, lang="pt",
                     outfiles=[rep_dir / "metricas_por_classe.png", rep_dir / "metricas_por_classe.pdf"])
    _per_class_chart(por_categoria, flat["ranking"], rep_dir, lang="en",
                     outfiles=[rep_dir / "per_class_metrics_en.png", rep_dir / "per_class_metrics_en.pdf"])

    # --- tabela POR CLASSE DE ERRO ---
    def _pc(c, k, nd=3):
        v = por_categoria.get(c, {}).get(k)
        return _f(v, nd) if v is not None else "—"
    _rows = ""
    for c in cats:
        v = por_categoria[c]
        n = v["n_test"] if v["n_test"] is not None else "—"
        _rows += (f"| `{c}` | {n} | {_pc(c,'deteccao_recall')} | "
                  f"{_pc(c,'auroc_vs_limpo')} {_ci(v.get('auroc_vs_limpo_ci95'))} | "
                  f"{_pc(c,'classif_precisao')} | {_pc(c,'classif_recall')} | {_pc(c,'classif_f1')} |\n")
    percls_md = f"""## 5. PER-CLASS metrics (per error category — coordinator request)

Two distinct questions → two metrics (do not conflate):
- **Detection (Stage 1):** of all errors in this category, how many does the "is there an error?" gate catch.
- **Classification (Stage 2):** once it is an error, does the model assign the correct category.

| Category | n (test) | **Detection** recall@op | **Detection** AUROC vs clean [CI95] | **Classif.** precision | **Classif.** recall | **Classif.** F1 |
|---|---|---|---|---|---|---|
{_rows}
> **How to read:** *recall@op* = fraction of that category's errors flagged as ERROR at the operating
> threshold. *AUROC vs clean* = category-vs-clean separability (⚠️ **confounded** — each category has its
> own resolution profile; indicative only). *precision/recall/F1* = quality of the **category assignment**
> (Stage 2). **There is no per-class precision for the gate** (a false positive is a clean screen, not
> attributable to a category). ⚠️ Classes with **small n** (e.g. `disordered_layout`) have unstable
> metrics — always read with the **support**.

**Ranking (support ≥ 5 only):** best **detected** = **{best_det or '—'}** · worst
detected = **{worst_det or '—'}** · best **classified** = **{best_cls or '—'}**.

📊 **Slide-ready charts** (per-class bars, detection × classification; n<5 dimmed/⚠):
- 🇬🇧 EN: `per_class_metrics_en.png` · `per_class_metrics_en.pdf` (vector — for paper/projector)
- 🇧🇷 PT: `metricas_por_classe.png` · `metricas_por_classe.pdf` (vector)

![Per-class metrics](per_class_metrics_en.png)

"""

    # --- markdown de apresentacao ---
    md = f"""# Experiment results — UI layout-error detector (siamese head · frozen DINOv2)

> **Config:** `{cfg_path.name}` · **Dataset:** `data/processed_v3` (single source of truth, flat + labels.csv) ·
> **Held-out:** {n_test} images ({n_clean} clean + {n_err} errors), test **locked** and evaluated **once**.
> Selection/calibration **on validation only** (anti-leakage protocol).

## ⚠️ How to read these numbers (READ BEFORE COMPARING)

This dataset may carry a **resolution confound** (clean screens concentrated at few device
resolutions). The trivial rule *"non-canonical resolution ⇒ error"* alone gives **AUROC {_f(res_trivial)}**
— without looking at the layout (**≈1.0 = strong confound; ≈0.5 = broken**). When this is high, the
GLOBAL metric is mostly confound. For a **fair** comparison with other models:
- **DO NOT** lead with global accuracy/AUROC (a naive model "wins" by exploiting the device).
- **LEAD** with the **confound-free** metrics (§2) and check whether the competitor beats the
  **confound baseline** (§3).

## 1. Standard metrics (Stage 1 — "is there an error?" gate) — balanced operating point

> Production decision (prototype+aux fusion, **calibrated on the confound-free validation set**).

| Metric | Value | 95% CI |
|---|---|---|
| **Accuracy** | **{_f(op.get('acuracia'))}** | {_ci(op.get('ci95_acuracia'))} |
| **Precision** | **{_f(op.get('precisao'))}** | {_ci(op.get('ci95_precisao'))} |
| **Recall (sensitivity)** | **{_f(op.get('recall'))}** | — |
| **F1-score** | **{_f(op.get('f1'))}** | {_ci(op.get('ci95_f1'))} |
| **Specificity** | **{_f(op.get('especificidade'))}** | {_ci(op.get('ci95_especificidade'))} |
| **Balanced accuracy** | **{_f(op.get('balanced_accuracy'))}** | — |
| **MCC** | **{_f(op.get('mcc'))}** | — |
| **AUROC** (fusion / prototype) | **{_f(fus_auroc)} / {_f(proto_auroc)}** | — |
| **AP (PR-AUC)** | **{_f(op.get('ap'))}** | — |
| Brier / ECE (calibration) | {_f(op.get('brier'))} / {_f(op.get('ece'))} | — |

Confusion matrix (gate): **TP={conf['TP']} · TN={conf['TN']} · FP={conf['FP']} · FN={conf['FN']}**
(`artifacts/reports/confusion_matrix.png`).

> **For the comparison table**, prefer **AUROC/AP** (threshold-free) and the **prototype** (cleaner
> signal, AUROC {_f(proto_auroc)}). The fusion AUROC ({_f(fus_auroc)}) is lower **on purpose**: it was
> calibrated **not** to exploit the resolution confound.

## 2. CONFOUND-FREE metrics (the honest ones — lead with these)

| Evaluation | Model (prototype) | Confound baseline | Verdict |
|---|---|---|---|
| **Confound-free synthetic** (errors injected into clean screens, same resolution) | **AUROC {_f(synth_proto)}** · AP {_f(synth_ap)} | — | {'✅ real signal' if sinal_synth else '⚠️ weak'} |
| **Controlled subset** (form-factor/orientation fixed) | **AUROC {_f(ctrl_proto)}** {_ci(ctrl_ci)} | {_f(ctrl_base)} | {'✅ beats it' if supera_ctrl else '⚠️ does not beat'} |
| **Falsifiability** (predicts error vs predicts resolution) | error {_f(fa_err)} | resolution {_f(fa_res)} | {'✅ separates' if (fa_err and fa_res and fa_err-fa_res>0.05) else '⚠️ tracks resolution'} |

## 3. Confound baselines (the "cheating ceiling" — what to compare against)

| Classifier | AUROC |
|---|---|
| Trivial resolution-only rule | {_f(res_trivial)} |
| Gray-padding fraction only | {_f(_g(glob,'baseline_fracao_padding_cinza','auroc'))} |
| LogReg on raw DINOv2 | {_f(_g(glob,'baseline_logreg_dinov2_cru','auroc'))} |
| **Model (prototype)** | **{_f(proto_auroc)}** |
| One-class kNN (DINOv2) | {_f(_g(glob,'baseline_oneclass_knn_dinov2','auroc'))} |

## 4. Stage 2 — error category (only when Stage 1 = error)

| Taxonomy | macro-F1 | 95% CI | note |
|---|---|---|---|
| **Coarse ({n_coarse} super-classes)** ⭐ | **{_f(e2_coarse)}** | {_ci(e2_coarse_ci)} | primary (statistical power) |
| Coarse, gate-conditioned (production) | {_f(e2_cond)} | — | only errors flagged by Stage 1 |
| Fine ({n_fine} classes) | {_f(e2_fine)} | — | secondary/exploratory (structural ceiling) |

> ⚠️ The coarse macro-F1 is higher because it is a **{n_coarse}**-class task (aggregation of the {n_fine} fine
> classes), **not** because the model got better; the lower CI bound is near chance ({_f(chance_fine,2)}).
> Always report **with the CI**.

{percls_md}## 6. VERDICT — does the model work on this dataset?

{chr(10).join('- ' + v for v in verdict)}

---
*Generated by `scripts/run_experiment.py`. Flat metrics JSON:
`artifacts/reports/EXPERIMENT_RESULTS.json`. Methodology: `docs/DESIGN.md`,
results: `docs/RELATORIO_FINAL_PROCESSED_V3.md`.*
"""
    (rep_dir / "EXPERIMENT_RESULTS.md").write_text(md, encoding="utf-8")
    return flat


def _print_console(flat: dict) -> None:
    op = flat["ponto_operacao"]
    print("\n" + "=" * 72)
    print("\033[1m EXPERIMENT RESULT (honest held-out)\033[0m")
    print("=" * 72)
    print(f" Test: {flat['n_test']} imgs ({flat['n_clean']} clean + {flat['n_erro']} errors)")
    print("\n STANDARD METRICS (gate, balanced operating point):")
    print(f"   Accuracy {_f(op['acuracia'])}  Precision {_f(op['precisao'])}  "
          f"Recall {_f(op['recall'])}  F1 {_f(op['f1'])}")
    print(f"   Specif. {_f(op['especificidade'])}  bAcc {_f(op['balanced_accuracy'])}  "
          f"MCC {_f(op['mcc'])}  AUROC {_f(flat['auroc_gate_fusao'])}/{_f(flat['auroc_gate_prototipo'])} (fusion/prototype)")
    print(f"   Confusion: {flat['confusao']}")
    print("\n CONFOUND-FREE METRICS (the honest ones):")
    print(f"   Confound-free synthetic: AUROC {_f(flat['sintetico_livre_confound']['auroc'])} "
          f"AP {_f(flat['sintetico_livre_confound']['ap'])}")
    print(f"   Controlled: prototype {_f(flat['controlado']['modelo_prototipo'])} "
          f"vs confound {_f(flat['controlado']['baseline_confound'])}")
    print(f"   Falsifiability: predicts error {_f(flat['falseabilidade']['prediz_erro'])} "
          f"vs resolution {_f(flat['falseabilidade']['prediz_resolucao'])}")
    print(f"   (trivial resolution baseline: AUROC {_f(flat['baseline_resolucao_trivial_auroc'])})")
    print(f"\n STAGE 2 (category): coarse F1 {_f(flat['estagio2']['grossa_f1_macro'])} "
          f"{_ci(flat['estagio2']['grossa_ci95'])} · fine F1 {_f(flat['estagio2']['fina_f1_macro'])}")
    pc = flat.get("por_categoria", {})
    if pc:
        print("\n PER-CLASS METRICS (detection | classification):")
        print(f"   {'category':18s} {'n':>3s} {'det.rec':>9s} {'AUROC':>6s} | "
              f"{'prec':>5s} {'rec':>5s} {'F1':>5s}")
        for c, v in pc.items():
            print(f"   {c:18s} {str(v['n_test'] or '—'):>3s} {_f(v['deteccao_recall']):>9s} "
                  f"{_f(v['auroc_vs_limpo']):>6s} | {_f(v['classif_precisao']):>5s} "
                  f"{_f(v['classif_recall']):>5s} {_f(v['classif_f1']):>5s}")
        rk = flat.get("ranking", {})
        print(f"   → best detected: {rk.get('melhor_detectado')} · best classified: "
              f"{rk.get('melhor_classificado')} (support ≥ 5)")
    print("=" * 72)
    print(" 📄 Full report: artifacts/reports/EXPERIMENT_RESULTS.md")
    print("    Flat metrics: artifacts/reports/EXPERIMENT_RESULTS.json")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--fresh", action="store_true",
                    help="reconstroi tudo do zero (splits/processed/embeddings/sinteticos)")
    ap.add_argument("--input", type=Path, default=Path("data/input"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed_v3"))
    args = ap.parse_args()

    from siamese.config import Config
    cfg = Config.load(args.config)
    emb = Path(cfg.paths.emb_dir)
    rep = Path(cfg.paths.reports_dir)
    splits = Path(cfg.paths.splits_dir)
    fresh = args.fresh

    print("\033[1m" + "#" * 72 + "\033[0m")
    print(f"\033[1m# EXPERIMENTO COMPLETO — {args.config}\033[0m")
    print(f"\033[1m# dataset: {args.processed} | fresh={fresh}\033[0m")
    print("\033[1m" + "#" * 72 + "\033[0m")

    # dataset PLANO (ex.: data/processed_v3 com labels.csv) ja vem materializado e splitado por
    # scripts/rebuild_processed_v3.py -> os passos 1-2 (build_splits/export_processed, que montam a
    # arvore LEGADA data/processed a partir de data/input) nao se aplicam.
    flat_dataset = (args.processed / "labels.csv").exists()

    # 1. splits
    if flat_dataset:
        print(f"\n[1/8 split] dataset PLANO ({args.processed}/labels.csv) — split já embutido, pulando")
    elif fresh or not _exists(splits / "train.csv", splits / "val.csv", splits / "test.csv"):
        _run("1/8 split", ["scripts/build_splits.py", "--input", str(args.input),
                           "--out", str(splits), "--val-frac", str(cfg.val_frac),
                           "--test-frac", str(cfg.test_frac), "--seed", str(cfg.seed)])
    else:
        print("\n[1/8 split] já existe — pulando (use --fresh para refazer)")

    # 2. processed (fonte da verdade)
    if flat_dataset:
        print(f"[2/8 processed] dataset PLANO já materializado em {args.processed} — pulando")
    elif fresh or not _exists(args.processed / "manifest.csv"):
        _run("2/8 processed", ["scripts/export_processed.py", "--config", str(args.config),
                               "--out", str(args.processed)])
    else:
        print("[2/8 processed] já existe — pulando")

    # 3. embeddings (passa os flags do backbone do config)
    ps = ["--use-patch-stats"] if cfg.backbone.use_patch_stats else []
    if fresh or not _exists(emb / "train.npz", emb / "val.npz", emb / "test.npz"):
        _run("3/8 features", ["scripts/extract_features.py", "--processed", str(args.processed),
                              "--out", str(emb), "--preprocess", cfg.backbone.preprocess,
                              "--size", str(cfg.backbone.size), *ps])
    else:
        print("[3/8 features] já existe — pulando")

    # 4. sondas sinteticas + reflow
    if fresh or not _exists(emb / "val_synth.npz", emb / "train_reflow.npz"):
        _run("4/8 sintético+reflow", ["scripts/make_synthetic.py", "--config", str(args.config),
                                      "--processed", str(args.processed)])
    else:
        print("[4/8 sintético+reflow] já existe — pulando")

    # 5-7. treino + avaliacao (SEMPRE — sao rapidos e queremos resultado fresco e consistente)
    _run("5/8 treino", ["scripts/train.py", "--config", str(args.config)])
    _run("6/8 avaliação DEV (val)", ["scripts/evaluate.py", "--config", str(args.config)])
    _run("7/8 TESTE held-out (1×)", ["scripts/evaluate.py", "--config", str(args.config), "--final-test"])

    # 8. relatorio consolidado
    print("\n\033[1m[8/8 relatório]\033[0m consolidando métricas de apresentação ...")
    flat = build_report(emb, rep, args.config)
    _print_console(flat)


if __name__ == "__main__":
    main()
