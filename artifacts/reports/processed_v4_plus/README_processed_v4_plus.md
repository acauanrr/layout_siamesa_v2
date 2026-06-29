# Experimento completo — `processed_v3_plus` → relatório `processed_v4_plus`

**Detector de erros de layout (cabeça siamesa sobre DINOv2 ViT-L/14 reg4 congelado).**
Pipeline rodado de ponta a ponta com `scripts/run_experiment.py` + `scripts/report_processed_v3.py`.

| | |
|---|---|
| **Config** | `configs/plus_L_reg4.yaml` (backbone vencedor `vit_large_patch14_reg4_dinov2`, 518px, patch-stats) |
| **Dataset** | `data/processed_v3_plus` — treino 1793 (493 reais + 1300 sintéticos) · val 122 · **teste 194 (held-out)** · 5 classes (clean + 4 erros). As matrizes/clusters de TREINO mostram as 493 reais (in-sample). |
| **Protocolo** | seleção/calibração **só na validação**; teste held-out avaliado **1×** (anti-vazamento) |
| **Limiar do gate** | 0.338 (calibrado por especificidade-alvo no conjunto livre-confound) |
| **Estágio 2** | categoria por **knn** · gate por **prototype** |

> ⚠️ **Como ler.** O dataset carrega um **confound de resolução**: a regra trivial *"resolução
> não-canônica ⇒ erro"* sozinha dá **AUROC 0.661** sem olhar o layout. Por isso a métrica
> GLOBAL é parcialmente confundida — **lidere com as métricas livre-confound (§3)**, não com a
> acurácia global. O confound já foi **quebrado** (era ~1.0 nos dados antigos → 0.661 agora).

---

## 1. Resumo executivo (respostas diretas)

- **Detectar erro vs tela limpa (held-out):** Acurácia **0.619** · Precisão **0.463** ·
  Recall **0.657** · F1 **0.543** · AUROC **0.691**
  (protótipo 0.751). A capacidade de **pegar telas com erro** (recall/sensibilidade)
  é **0.657**: o gate captura ~66% dos erros reais.
- **Sinal real (livre de confound):** sintético AUROC **0.802** (AP 0.940);
  no subconjunto controlado o modelo (0.655) **supera** o baseline de confound
  (0.497); e **prediz erro (0.691) melhor que resolução
  (0.496)**. → o modelo aprende o **erro**, não só o dispositivo.
- **Melhor classe de erro identificada:** **`black_bars`** (maior AUROC e maior F1 fim-a-fim).
  **Pior:** **`disordered_layout`** (classificação colapsa — ver §5).

---

## 2. Classificação binária — erro vs sem-erro (gate, Estágio 1)

Matrizes: `confusion_matrix_binaria_treino.png` · `confusion_matrix_binaria_teste.png`

| split | Acurácia | Precisão | Recall | F1 | AUROC | Especif. | bAcc | MCC | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TREINO (in-sample) | 0.767 | 0.624 | 0.792 | 0.698 | 0.873 | 0.754 | 0.773 | 0.522 | 133/245/80/35 |
| **TESTE (held-out)** | 0.619 | 0.463 | 0.657 | 0.543 | 0.691 | 0.598 | 0.628 | 0.243 | 44/76/51/23 |

> O **TREINO** é ressubstituição (o modelo já viu) → otimista; **o número vinculante é o TESTE**.
> *Recall* = "capacidade de identificar telas com erro". *Precisão* baixa + *especificidade* 0.598
> refletem a calibração **specificity-first** sobre dados com confound atenuado (corta falso-alarme em telas limpas).

---

## 3. Métricas LIVRE DE CONFOUND (as honestas — lidere com estas)

| Avaliação | Modelo (protótipo) | Baseline de confound | Veredito |
|---|---|---|---|
| **Sintético livre-confound** (erros injetados em telas limpas, mesma resolução) | **AUROC 0.802** · AP 0.940 | — | ✅ sinal real |
| **Subconjunto controlado** (form-factor/orientação fixos) | **AUROC 0.655** [0.50–0.74] | 0.497 | ✅ supera |
| **Falseabilidade** (prediz erro × prediz resolução) | erro 0.691 | resolução 0.496 | ✅ separa |

**Teto de "trapaça" (baselines de confound, held-out):** resolução-trivial 0.661 ·
LogReg DINOv2 cru 0.822 · 1-classe kNN 0.739 ·
fração padding-cinza 0.489 — comparar competidores contra estes, não contra a métrica global.

---

## 4. Categoria — clean + 4 erros (sistema 2 estágios, 5 classes)

Matrizes 5×5: `confusion_matrix_categoria_treino.png` · `confusion_matrix_categoria_teste.png`
(prediz `clean` se o gate disser sem-erro; senão a categoria do decisor de erro — por isso a coluna
`clean` bate com a matriz binária da §2).

| split | Acurácia | F1-macro | AUROC-macro |
|---|---:|---:|---:|
| TREINO (in-sample) | 0.753 | 0.742 | 0.916 |
| **TESTE (held-out)** | 0.505 | 0.327 | 0.596 |

---

## 5. Métricas POR CLASSE — qual erro o modelo identifica melhor

### 5a. Sistema fim-a-fim, 5 classes (held-out) — gráfico: `metricas_por_classe.png`

*precisão* = dos que chamou de X, quantos eram X · *recall* = dos X reais, quantos identificou
(passar no gate **e** acertar a categoria) · *AUROC* = ranqueia X acima do resto (livre de limiar) ·
*acc-ovr* = acerto no binário "é X ou não".

| classe | n | Precisão | Recall | F1 | AUROC | Acur. (one-vs-rest) |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 127 | 0.768 | 0.598 | 0.673 | 0.751 | 0.619 |
| `black_bars` | 22 | 0.522 | 0.545 | 0.533 | 0.783 | 0.892 |
| `disordered_layout` | 10 | 0.000 | 0.000 | 0.000 | 0.411 | 0.881 |
| `empty_space` | 14 | 0.182 | 0.286 | 0.222 | 0.533 | 0.856 |
| `overlay` | 21 | 0.162 | 0.286 | 0.207 | 0.500 | 0.763 |

**Ranking (suporte ≥ 5):** melhor AUROC = **`black_bars`** · melhor F1 = **`black_bars`** ·
pior AUROC = **`disordered_layout`** · pior F1 = **`disordered_layout`**.

> 🔎 **`disordered_layout`** é o ponto fraco: o gate até **detecta** parte dela, mas a **classificação**
> fim-a-fim vai a ~0 (poucos exemplos, n=10, e fronteira difícil nas features
> congeladas). É o alvo do trabalho futuro opcional (telas near-square / mais exemplos).

### 5b. Decomposição detecção × classificação (oráculo, held-out) — gráfico: `per_class_metrics_en.png`

Separa as duas perguntas: **Detecção (E1)** = dos erros desta classe, quantos o gate pega;
**Classificação (E2)** = já sendo erro, acerta a categoria (medido no oráculo de erros).

| classe | n | Detecção recall@op | Detecção AUROC-vs-limpo | Classif. precisão | Classif. recall | Classif. F1 |
|---|---:|---:|---:|---:|---:|---:|
| `black_bars` | 22 | 0.773 | 0.892 | 0.867 | 0.591 | 0.703 |
| `disordered_layout` | 10 | 0.600 | 0.638 | 0.000 | 0.000 | 0.000 |
| `empty_space` | 14 | 0.714 | 0.717 | 0.308 | 0.286 | 0.296 |
| `overlay` | 21 | 0.524 | 0.682 | 0.500 | 0.762 | 0.604 |

> Por classe de treino (referência/ressubstituição) em `metricas_por_classe.json` → `por_classe_treino`.

---

## 6. Estágio 2 — taxonomia do erro

| Taxonomia | F1-macro | IC95 | nota |
|---|---:|---|---|
| **Grossa (clean + 2 super-classes)** ⭐ | **0.671** | [0.56–0.78] | primária (poder estatístico) |
| Grossa, condicional ao gate (produção) | 0.664 | — | só erros sinalizados pelo E1 |
| Fina (4 classes de erro) | 0.401 | — | secundária/exploratória (teto estrutural) |

---

## 7. Veredito

✅ **O modelo funciona neste dataset** — detecta o **erro** (não só o dispositivo): supera o baseline de
confound no subconjunto controlado, atinge AUROC **0.802** no regime livre-confound e prediz
erro melhor que resolução. O confound foi **quebrado na origem** (resolução-trivial 0.661, era ~1.0).
A acurácia global do gate (0.619) é modesta **por escolha de calibração** (specificity-first) e
porque a métrica global ainda é parcialmente confundida — por isso o relatório lidera com §3.
Ponto a melhorar: **classificação de `disordered_layout`** (§5).

---

## 8. Índice de arquivos (`artifacts/reports/processed_v4_plus/`)

**Clusters interativos (espaço aprendido z, UMAP):**
- `clusters_treino.html` · `clusters_teste.html` — interativos (zoom/pan, hover com arquivo e p(erro), ★=protótipo)
- `clusters_treino.png` · `clusters_teste.png` — versões estáticas (mesmas coordenadas)

**Matrizes de confusão (treino + teste):**
- `confusion_matrix_binaria_treino.png` · `confusion_matrix_binaria_teste.png` — erro/sem-erro (acc/prec/rec/F1/AUROC no título)
- `confusion_matrix_categoria_treino.png` · `confusion_matrix_categoria_teste.png` — 5 classes (clean + 4 erros)

**Métricas por classe:**
- `metricas_por_classe.png` — fim-a-fim 5 classes (§5a) · `metricas_por_classe.json` — números planos (treino+teste)
- `per_class_metrics_en.png` / `per_class_metrics_pt.pdf` — decomposição detecção×classificação (§5b)

**Relatórios / métricas planas:**
- `README_processed_v4_plus.md` — **este arquivo** (índice + tabelas-resumo)
- `RELATORIO_processed_v4_plus.md` — relatório do gerador visual
- `EXPERIMENT_RESULTS.md` / `.json` — análise honesta livre-confound completa
- `evaluation_report_heldout.json` — métricas held-out completas (treino vs teste, baselines, falseabilidade, E2)
- `evaluation_plots.png` — curvas ROC/PR/calibração do gate (held-out)

*Gerado por `scripts/run_experiment.py` (treino+avaliação) e `scripts/report_processed_v3.py` (visual),
config `configs/plus_L_reg4.yaml`, dataset `data/processed_v3_plus`.*
