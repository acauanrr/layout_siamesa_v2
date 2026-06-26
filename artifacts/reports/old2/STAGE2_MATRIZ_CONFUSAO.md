# Estágio 2 — Matriz de confusão por categoria (taxonomia FINA, 6 grupos)

> **Held-out** (teste processado 1×). O Estágio 2 atribui a **categoria do erro** apenas a imagens
> de **erro**, pelo **protótipo de categoria mais próximo** no espaço SupCon (método canônico).
> Modo **ORÁCULO** = classifica os **89 erros reais** do teste (isola a qualidade do
> Estágio 2, independente do gate). Apêndice = **condicional ao gate** (produção).
>
> ⚠️ **Escala/fragilidade:** todas as métricas finas vêm de **um único held-out de 89 erros**;
> **4 das 6 classes têm n≤16** (raras: `distortion` n=3, `orientation` n=2). Tratar como **indicativo**, não definitivo.

## Resumo (oráculo, n=89)

- **Acurácia global (multiclasse):** **33.7%** — fração dos 89 erros com a categoria correta (acertos na diagonal: **30/89**).
- **F1-macro:** **0.21** [IC95 0.14–0.27] — intervalo largo (n pequeno): leia como ordem de grandeza.
- Taxonomia **grossa** (3 superclasses): acurácia **57.3%**, F1-macro **0.39** [IC95 0.32–0.46].
  > **Por que a grossa parece melhor?** O salto fina→grossa é, em boa parte, **agregação 6→3**: o mapa
  > funde justamente os vizinhos mais confundidos (`overlay`↔`disordered_layout`, `black_bars`↔`empty_space`).
  > É a **mesma representação** medida numa tarefa **mais fácil** (3 classes), não um modelo melhor — e os
  > dois IC95 (0.14–0.27 vs 0.32–0.46) **se sobrepõem**: o ganho é **sugestivo, não estatístico**.

## Métricas por grupo (precisão e acurácia em destaque)

| Grupo | n | acertos | **Precisão** | Recall (acerto) | F1 | **Acurácia¹** (1-vs-resto) |
|---|---|---|---|---|---|---|
| `black_bars` | 28 | 14 | **44%** | 50% | 0.47 | **64%** |
| `overlay` | 27 | 10 | **33%** | 37% | 0.35 | **58%** |
| `empty_space` | 16 | 3 | **33%** | 19% | 0.24 | **79%** |
| `disordered_layout` | 13 | 3 | **17%** | 23% | 0.19 | **72%** |
| `distortion` ⚠ | 3 | 0 | — *(0/0)* | 0% | 0.00 | **97%** |
| `orientation` ⚠ | 2 | 0 | — *(0/0)* | 0% | 0.00 | **98%** |

> ¹ **Acurácia (1-vs-resto)** = (acertos + rejeições corretas)/total. **Cuidado:** ela fica *alta*
> para classes **raras** (`distortion` n=3, `orientation` n=2) só porque há muitos verdadeiros-negativos — `orientation` tem
> acurácia 98% mas **recall 0%** (não acertou nenhum). Por isso, para "como o modelo está **acertando**
> cada grupo", a métrica honesta é o **recall (taxa de acerto)** + o **suporte (n)**. A **acurácia global
> multiclasse** (33.7%) é a média real (micro: precisão = recall = acurácia).
> **Precisão `—`** = o modelo **nunca previu** essa classe (`distortion`, `orientation`): precisão 0/0 é **indefinida**, não medida.

## Como o modelo está acertando (leitura das confusões)

- ✅ **`black_bars` é a única classe utilizável** — precisão **44%** / recall **50%** (n=28).
- ⚠️ **`empty_space` vaza para `black_bars`**: 6 de 16 (38%) viram `black_bars` — é mandado para `black_bars` **mais vezes** do que é acertado (recall só 19%).
- ⚠️ **`overlay` empata acerto e erro**: vai para `disordered_layout` (7) **tão frequentemente quanto acerta** (10) — ambos 37% da linha. Na figura, a **diagonal tem borda vermelha** para distinguir o acerto.
- ⛔ **Classes raras (n<5) não são interpretáveis** (`distortion` n=3, `orientation` n=2) — `orientation` nunca foi prevista (precisão indefinida); `distortion` é 0/3 (parece 100% de precisão por acaso de n).
- **Maiores confusões (verdadeiro → previsto):**
  - **overlay → black_bars**: 8 casos (30% da linha de `overlay`)
  - **overlay → disordered_layout**: 7 casos (26% da linha de `overlay`)
  - **black_bars → overlay**: 7 casos (25% da linha de `black_bars`)
  - **empty_space → black_bars**: 6 casos (38% da linha de `empty_space`)

## Matriz de confusão (linhas = verdadeiro · colunas = previsto)

| verdadeiro ↓ \ previsto → | black_bars | disordered_layout | distortion | empty_space | orientation | overlay | **n** |
|---|---|---|---|---|---|---|---|
| **black_bars** | **14** | 4 | 0 | 3 | 0 | 7 | 28 |
| **disordered_layout** | 4 | **3** | 0 | 1 | 0 | 5 | 13 |
| **distortion** | 0 | 1 | **0** | 0 | 0 | 2 | 3 |
| **empty_space** | 6 | 1 | 0 | **3** | 0 | 6 | 16 |
| **orientation** | 0 | 2 | 0 | 0 | **0** | 0 | 2 |
| **overlay** | 8 | 7 | 0 | 2 | 0 | **10** | 27 |

## Apêndice — condicional ao gate E1 (produção, n=68)

Só os erros que o Estágio 1 sinalizou (68/89). Acurácia global **32.4%**, F1-macro **0.19**.

| Grupo | n | acertos | **Precisão** | Recall | F1 | **Acurácia¹** |
|---|---|---|---|---|---|---|
| `black_bars` | 23 | 14 | **44%** | 61% | 0.51 | **60%** |
| `overlay` | 19 | 2 | **22%** | 11% | 0.14 | **65%** |
| `empty_space` | 12 | 3 | **33%** | 25% | 0.29 | **78%** |
| `disordered_layout` | 10 | 3 | **17%** | 30% | 0.21 | **68%** |
| `distortion` ⚠ | 2 | 0 | — *(0/0)* | 0% | 0.00 | **97%** |
| `orientation` ⚠ | 2 | 0 | — *(0/0)* | 0% | 0.00 | **97%** |


## Figuras geradas
- `confusion_matrix_stage2_fina.png` / `.pdf` (PT) · `_en` (EN) — matriz 6×6 (contagens + recall).
- `metricas_stage2_por_grupo.png` / `.pdf` (PT) · `stage2_per_group_metrics_en.*` (EN) — tabela por grupo.

*Métricas recalculadas da matriz crua e conferidas contra `evaluation_report.json` (assert).*
