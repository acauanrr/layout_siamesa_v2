# Experiment results — UI layout-error detector (siamese head · frozen DINOv2)

> **Config:** `default.yaml` · **Dataset:** `data/processed/` (single source of truth, corrected) ·
> **Held-out:** 130 images (41 clean + 89 errors), test **locked** and evaluated **once**.
> Selection/calibration **on validation only** (anti-leakage protocol).

## ⚠️ How to read these numbers (READ BEFORE COMPARING)

This dataset has a **resolution confound**: **every** clean screen is 2076×2152 (a single device).
The trivial rule *"resolution ≠ 2076×2152 ⇒ error"* alone gives **AUROC 0.994** — without
looking at the layout. **So the GLOBAL metric is ~98% confound.** For a **fair** comparison with other
models:
- **DO NOT** lead with global accuracy/AUROC (a naive model "wins" by exploiting the device).
- **LEAD** with the **confound-free** metrics (§2) and check whether the competitor beats the
  **confound baseline** (§3).

## 1. Standard metrics (Stage 1 — "is there an error?" gate) — balanced operating point

> Production decision (prototype+aux fusion, **calibrated on the confound-free validation set**).

| Metric | Value | 95% CI |
|---|---|---|
| **Accuracy** | **0.723** | [0.65–0.82] |
| **Precision** | **0.819** | [0.62–0.99] |
| **Recall (sensitivity)** | **0.764** | — |
| **F1-score** | **0.791** | [0.67–0.88] |
| **Specificity** | **0.634** | [0.54–0.82] |
| **Balanced accuracy** | **0.699** | — |
| **MCC** | **0.385** | — |
| **AUROC** (fusion / prototype) | **0.715 / 0.744** | — |
| **AP (PR-AUC)** | **0.801** | — |
| Brier / ECE (calibration) | 0.208 / 0.136 | — |

Confusion matrix (gate): **TP=68 · TN=26 · FP=15 · FN=21**
(`artifacts/reports/confusion_matrix.png`).

> **For the comparison table**, prefer **AUROC/AP** (threshold-free) and the **prototype** (cleaner
> signal, AUROC 0.744). The fusion AUROC (0.715) is lower **on purpose**: it was
> calibrated **not** to exploit the resolution confound.

## 2. CONFOUND-FREE metrics (the honest ones — lead with these)

| Evaluation | Model (prototype) | Confound baseline | Verdict |
|---|---|---|---|
| **Confound-free synthetic** (errors injected into clean screens, same resolution) | **AUROC 0.713** · AP 0.887 | — | ✅ real signal |
| **Controlled subset** (form-factor/orientation fixed) | **AUROC 0.693** [0.55–0.83] | 0.383 | ✅ beats it |
| **Falsifiability** (predicts error vs predicts resolution) | error 0.715 | resolution 0.705 | ⚠️ tracks resolution |

## 3. Confound baselines (the "cheating ceiling" — what to compare against)

| Classifier | AUROC |
|---|---|
| Trivial resolution-only rule | 0.994 |
| Gray-padding fraction only | 0.972 |
| LogReg on raw DINOv2 | 0.746 |
| **Model (prototype)** | **0.744** |
| One-class kNN (DINOv2) | 0.675 |

## 4. Stage 2 — error category (only when Stage 1 = error)

| Taxonomy | macro-F1 | 95% CI | note |
|---|---|---|---|
| **Coarse (3 super-classes)** ⭐ | **0.393** | [0.32–0.46] | primary (statistical power) |
| Coarse, gate-conditioned (production) | 0.395 | — | only errors flagged by Stage 1 |
| Fine (6 classes) | 0.209 | — | secondary/exploratory (structural ceiling) |

> ⚠️ The coarse macro-F1 is higher because it is a **3**-class task (aggregation of the 6 fine
> classes), **not** because the model got better; the lower CI bound is near chance (0.33).
> Always report **with the CI**.

## 5. PER-CLASS metrics (per error category — coordinator request)

Two distinct questions → two metrics (do not conflate):
- **Detection (Stage 1):** of all errors in this category, how many does the "is there an error?" gate catch.
- **Classification (Stage 2):** once it is an error, does the model assign the correct category.

| Category | n (test) | **Detection** recall@op | **Detection** AUROC vs clean [CI95] | **Classif.** precision | **Classif.** recall | **Classif.** F1 |
|---|---|---|---|---|---|---|
| `black_bars` | 28 | 0.821 | 0.791 [0.70–0.92] | 0.438 | 0.500 | 0.467 |
| `disordered_layout` | 13 | 0.769 | 0.711 [0.55–0.89] | 0.167 | 0.231 | 0.194 |
| `distortion` | 3 | 0.667 | 0.724 [0.48–0.91] | 0.000 | 0.000 | 0.000 |
| `empty_space` | 16 | 0.750 | 0.733 [0.60–0.88] | 0.333 | 0.188 | 0.240 |
| `orientation` | 2 | 1.000 | 0.854 [0.79–1.00] | 0.000 | 0.000 | 0.000 |
| `overlay` | 27 | 0.704 | 0.711 [0.61–0.85] | 0.333 | 0.370 | 0.351 |

> **How to read:** *recall@op* = fraction of that category's errors flagged as ERROR at the operating
> threshold. *AUROC vs clean* = category-vs-clean separability (⚠️ **confounded** — each category has its
> own resolution profile; indicative only). *precision/recall/F1* = quality of the **category assignment**
> (Stage 2). **There is no per-class precision for the gate** (a false positive is a clean screen, not
> attributable to a category). ⚠️ Classes with **small n** (`orientation`, `distortion`) have unstable
> metrics — always read with the **support**.

**Ranking (support ≥ 5 only):** best **detected** = **black_bars** · worst
detected = **overlay** · best **classified** = **black_bars**.

📊 **Slide-ready charts** (per-class bars, detection × classification; n<5 dimmed/⚠):
- 🇬🇧 EN: `per_class_metrics_en.png` · `per_class_metrics_en.pdf` (vector — for paper/projector)
- 🇧🇷 PT: `metricas_por_classe.png` · `metricas_por_classe.pdf` (vector)

![Per-class metrics](per_class_metrics_en.png)

## 6. VERDICT — does the model work on this dataset?

- ✅ REAL CONTENT SIGNAL: on the CONTROLLED subset (form-factor/orientation fixed) the model (prototype AUROC 0.693) BEATS the confound baseline (0.383); and on the CONFOUND-FREE synthetic it reaches AUROC 0.713 (AP 0.887). The model detects the ERROR, not just the device.
- ⚠️ CONFOUND NOT BEATEN globally: the score predicts ERROR (0.715) about as well as RESOLUTION (0.705) (gap 0.010). The confound is ATTENUATED, not eliminated — beating it needs more diverse CLEAN screens (data).
- ℹ️ The GLOBAL metric is confounded: the trivial resolution rule alone gives AUROC 0.994 — so the model's global accuracy must NOT be compared naively with models that exploit the confound. Lead with confound-free AUROC.

---
*Generated by `scripts/run_experiment.py`. Flat metrics JSON:
`artifacts/reports/EXPERIMENT_RESULTS.json`. Methodology: `docs/DESIGN.md`,
presentation: `docs/RELATORIO_APRESENTACAO.md`.*
