# Estágio 2 — Matriz de confusão por categoria (taxonomia FINA, 6 grupos)

> **Held-out** (teste processado 1×). O Estágio 2 atribui a **categoria do erro** apenas a imagens
> de **erro**, pelo **protótipo de categoria mais próximo** no espaço SupCon (método canônico).
> Modo **ORÁCULO** = classifica os **89 erros reais** do teste (isola a qualidade do
> Estágio 2, independente do gate). Apêndice = **condicional ao gate** (produção).
>
> ⚠️ **Escala/fragilidade:** todas as métricas finas vêm de **um único held-out de 89 erros**;
> **4 das 6 classes têm n≤16** (raras: `distortion` n=3, `orientation` n=2). Tratar como **indicativo**, não definitivo.

## Resumo (oráculo, n=89)

- **Acurácia global (multiclasse):** **38.2%** — fração dos 89 erros com a categoria correta (acertos na diagonal: **34/89**).
- **F1-macro:** **0.36** [IC95 0.18–0.44] — intervalo largo (n pequeno): leia como ordem de grandeza.
- Taxonomia **grossa** (3 superclasses): acurácia **64.0%**, F1-macro **0.62** [IC95 0.38–0.76].
  > **Por que a grossa parece melhor?** O salto fina→grossa é, em boa parte, **agregação 6→3**: o mapa
  > funde justamente os vizinhos mais confundidos (`overlay`↔`disordered_layout`, `black_bars`↔`empty_space`).
  > É a **mesma representação** medida numa tarefa **mais fácil** (3 classes), não um modelo melhor — e os
  > dois IC95 (0.18–0.44 vs 0.38–0.76) **se sobrepõem**: o ganho é **sugestivo, não estatístico**.

## Métricas por grupo (precisão e acurácia em destaque)

| Grupo | n | acertos | **Precisão** | Recall (acerto) | F1 | **Acurácia¹** (1-vs-resto) |
|---|---|---|---|---|---|---|
| `black_bars` | 28 | 16 | **52%** | 57% | 0.54 | **70%** |
| `overlay` | 27 | 9 | **35%** | 33% | 0.34 | **61%** |
| `empty_space` | 16 | 4 | **31%** | 25% | 0.28 | **76%** |
| `disordered_layout` | 13 | 3 | **18%** | 23% | 0.20 | **73%** |
| `distortion` ⚠ | 3 | 2 | **100%** | 67% | 0.80 | **99%** |
| `orientation` ⚠ | 2 | 0 | — *(0/0)* | 0% | 0.00 | **98%** |

> ¹ **Acurácia (1-vs-resto)** = (acertos + rejeições corretas)/total. **Cuidado:** ela fica *alta*
> para classes **raras** (`distortion` n=3, `orientation` n=2) só porque há muitos verdadeiros-negativos — `orientation` tem
> acurácia 98% mas **recall 0%** (não acertou nenhum). Por isso, para "como o modelo está **acertando**
> cada grupo", a métrica honesta é o **recall (taxa de acerto)** + o **suporte (n)**. A **acurácia global
> multiclasse** (38.2%) é a média real (micro: precisão = recall = acurácia).
> **Precisão `—`** = o modelo **nunca previu** essa classe (`orientation`): precisão 0/0 é **indefinida**, não medida.

## Como o modelo está acertando (leitura das confusões)

- ✅ **`black_bars` é a única classe utilizável** — precisão **52%** / recall **57%** (n=28).
- ⚠️ **`empty_space` vaza para `black_bars`**: 7 de 16 (44%) viram `black_bars` — é mandado para `black_bars` **mais vezes** do que é acertado (recall só 25%).
- ⚠️ **`overlay` empata acerto e erro**: vai para `disordered_layout` (9) **tão frequentemente quanto acerta** (9) — ambos 33% da linha. Na figura, a **diagonal tem borda vermelha** para distinguir o acerto.
- ⛔ **Classes raras (n<5) não são interpretáveis** (`distortion` n=3, `orientation` n=2) — `orientation` nunca foi prevista (precisão indefinida); `distortion` é 2/3 (parece 100% de precisão por acaso de n).
- **Maiores confusões (verdadeiro → previsto):**
  - **overlay → disordered_layout**: 9 casos (33% da linha de `overlay`)
  - **empty_space → black_bars**: 7 casos (44% da linha de `empty_space`)
  - **black_bars → overlay**: 7 casos (25% da linha de `black_bars`)
  - **overlay → black_bars**: 6 casos (22% da linha de `overlay`)

## Matriz de confusão (linhas = verdadeiro · colunas = previsto)

| verdadeiro ↓ \ previsto → | black_bars | disordered_layout | distortion | empty_space | orientation | overlay | **n** |
|---|---|---|---|---|---|---|---|
| **black_bars** | **16** | 3 | 0 | 2 | 0 | 7 | 28 |
| **disordered_layout** | 2 | **3** | 0 | 3 | 0 | 5 | 13 |
| **distortion** | 0 | 0 | **2** | 0 | 0 | 1 | 3 |
| **empty_space** | 7 | 1 | 0 | **4** | 0 | 4 | 16 |
| **orientation** | 0 | 1 | 0 | 1 | **0** | 0 | 2 |
| **overlay** | 6 | 9 | 0 | 3 | 0 | **9** | 27 |

## Apêndice — condicional ao gate E1 (produção, n=78)

Só os erros que o Estágio 1 sinalizou (78/89). Acurácia global **37.2%**, F1-macro **0.39**.

| Grupo | n | acertos | **Precisão** | Recall | F1 | **Acurácia¹** |
|---|---|---|---|---|---|---|
| `black_bars` | 27 | 16 | **52%** | 59% | 0.55 | **67%** |
| `overlay` | 23 | 5 | **31%** | 22% | 0.26 | **63%** |
| `empty_space` | 13 | 4 | **31%** | 31% | 0.31 | **77%** |
| `disordered_layout` | 12 | 3 | **18%** | 25% | 0.21 | **71%** |
| `orientation` ⚠ | 2 | 0 | — *(0/0)* | 0% | 0.00 | **97%** |
| `distortion` ⚠ | 1 | 1 | **100%** | 100% | 1.00 | **100%** |


## Figuras geradas
- `confusion_matrix_stage2_fina.png` / `.pdf` (PT) · `_en` (EN) — matriz 6×6 (contagens + recall).
- `metricas_stage2_por_grupo.png` / `.pdf` (PT) · `stage2_per_group_metrics_en.*` (EN) — tabela por grupo.

*Métricas recalculadas da matriz crua e conferidas contra `evaluation_report.json` (assert).*
