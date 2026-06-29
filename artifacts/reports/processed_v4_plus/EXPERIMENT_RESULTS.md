# Experiment results — UI layout-error detector (siamese head · frozen DINOv2)

> **Config:** `plus_L_reg4.yaml` · **Dataset:** `data/processed_v3` (single source of truth, flat + labels.csv) ·
> **Held-out:** 194 images (127 clean + 67 errors), test **locked** and evaluated **once**.
> Selection/calibration **on validation only** (anti-leakage protocol).

## ⚠️ How to read these numbers (READ BEFORE COMPARING)

This dataset may carry a **resolution confound** (clean screens concentrated at few device
resolutions). The trivial rule *"non-canonical resolution ⇒ error"* alone gives **AUROC 0.661**
— without looking at the layout (**≈1.0 = strong confound; ≈0.5 = broken**). When this is high, the
GLOBAL metric is mostly confound. For a **fair** comparison with other models:
- **DO NOT** lead with global accuracy/AUROC (a naive model "wins" by exploiting the device).
- **LEAD** with the **confound-free** metrics (§2) and check whether the competitor beats the
  **confound baseline** (§3).

## 1. Standard metrics (Stage 1 — "is there an error?" gate) — balanced operating point

> Production decision (prototype+aux fusion, **calibrated on the confound-free validation set**).

| Metric | Value | 95% CI |
|---|---|---|
| **Accuracy** | **0.619** | [0.55–0.69] |
| **Precision** | **0.463** | [0.34–0.60] |
| **Recall (sensitivity)** | **0.657** | — |
| **F1-score** | **0.543** | [0.43–0.66] |
| **Specificity** | **0.598** | [0.51–0.69] |
| **Balanced accuracy** | **0.628** | — |
| **MCC** | **0.243** | — |
| **AUROC** (fusion / prototype) | **0.691 / 0.751** | — |
| **AP (PR-AUC)** | **0.590** | — |
| Brier / ECE (calibration) | 0.210 / 0.072 | — |

Confusion matrix (gate): **TP=44 · TN=76 · FP=51 · FN=23**
(`artifacts/reports/confusion_matrix.png`).

> **For the comparison table**, prefer **AUROC/AP** (threshold-free) and the **prototype** (cleaner
> signal, AUROC 0.751). The fusion AUROC (0.691) is lower **on purpose**: it was
> calibrated **not** to exploit the resolution confound.

## 2. CONFOUND-FREE metrics (the honest ones — lead with these)

| Evaluation | Model (prototype) | Confound baseline | Verdict |
|---|---|---|---|
| **Confound-free synthetic** (errors injected into clean screens, same resolution) | **AUROC 0.802** · AP 0.940 | — | ✅ real signal |
| **Controlled subset** (form-factor/orientation fixed) | **AUROC 0.655** [0.50–0.74] | 0.497 | ✅ beats it |
| **Falsifiability** (predicts error vs predicts resolution) | error 0.691 | resolution 0.496 | ✅ separates |

## 3. Confound baselines (the "cheating ceiling" — what to compare against)

| Classifier | AUROC |
|---|---|
| Trivial resolution-only rule | 0.661 |
| Gray-padding fraction only | 0.489 |
| LogReg on raw DINOv2 | 0.822 |
| **Model (prototype)** | **0.751** |
| One-class kNN (DINOv2) | 0.739 |

## 4. Stage 2 — error category (only when Stage 1 = error)

| Taxonomy | macro-F1 | 95% CI | note |
|---|---|---|---|
| **Coarse (2 super-classes)** ⭐ | **0.671** | [0.56–0.78] | primary (statistical power) |
| Coarse, gate-conditioned (production) | 0.664 | — | only errors flagged by Stage 1 |
| Fine (4 classes) | 0.401 | — | secondary/exploratory (structural ceiling) |

> ⚠️ The coarse macro-F1 is higher because it is a **2**-class task (aggregation of the 4 fine
> classes), **not** because the model got better; the lower CI bound is near chance (0.25).
> Always report **with the CI**.

## 5. PER-CLASS metrics (per error category — coordinator request)

Two distinct questions → two metrics (do not conflate):
- **Detection (Stage 1):** of all errors in this category, how many does the "is there an error?" gate catch.
- **Classification (Stage 2):** once it is an error, does the model assign the correct category.

| Category | n (test) | **Detection** recall@op | **Detection** AUROC vs clean [CI95] | **Classif.** precision | **Classif.** recall | **Classif.** F1 |
|---|---|---|---|---|---|---|
| `black_bars` | 22 | 0.773 | 0.892 [0.82–0.96] | 0.867 | 0.591 | 0.703 |
| `disordered_layout` | 10 | 0.600 | 0.638 [0.43–0.86] | 0.000 | 0.000 | 0.000 |
| `empty_space` | 14 | 0.714 | 0.717 [0.54–0.86] | 0.308 | 0.286 | 0.296 |
| `overlay` | 21 | 0.524 | 0.682 [0.57–0.80] | 0.500 | 0.762 | 0.604 |

> **How to read:** *recall@op* = fraction of that category's errors flagged as ERROR at the operating
> threshold. *AUROC vs clean* = category-vs-clean separability (⚠️ **confounded** — each category has its
> own resolution profile; indicative only). *precision/recall/F1* = quality of the **category assignment**
> (Stage 2). **There is no per-class precision for the gate** (a false positive is a clean screen, not
> attributable to a category). ⚠️ Classes with **small n** (e.g. `disordered_layout`) have unstable
> metrics — always read with the **support**.

**Ranking (support ≥ 5 only):** best **detected** = **black_bars** · worst
detected = **disordered_layout** · best **classified** = **black_bars**.

📊 **Slide-ready charts** (per-class bars, detection × classification; n<5 dimmed/⚠):
- 🇬🇧 EN: `per_class_metrics_en.png` · `per_class_metrics_en.pdf` (vector — for paper/projector)
- 🇧🇷 PT: `metricas_por_classe.png` · `metricas_por_classe.pdf` (vector)

![Per-class metrics](per_class_metrics_en.png)

## 6. VERDICT — does the model work on this dataset?

- ✅ REAL CONTENT SIGNAL: on the CONTROLLED subset (form-factor/orientation fixed) the model (prototype AUROC 0.655) BEATS the confound baseline (0.497); and on the CONFOUND-FREE synthetic it reaches AUROC 0.802 (AP 0.940). The model detects the ERROR, not just the device.
- ✅ FALSIFIABILITY: predicts ERROR (0.691) better than RESOLUTION (0.496) (gap 0.195).
- ℹ️ The GLOBAL metric is confounded: the trivial resolution rule alone gives AUROC 0.661 — so the model's global accuracy must NOT be compared naively with models that exploit the confound. Lead with confound-free AUROC.

---
*Generated by `scripts/run_experiment.py`. Flat metrics JSON:
`artifacts/reports/EXPERIMENT_RESULTS.json`. Methodology: `docs/DESIGN.md`,
results: `docs/RELATORIO_FINAL_PROCESSED_V3.md`.*
