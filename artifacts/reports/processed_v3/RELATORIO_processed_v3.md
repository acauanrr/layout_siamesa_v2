# Relatório visual — `processed_v3` (setup principal)

Gerado por `scripts/report_processed_v3.py` a partir do modelo congelado (treino+teste completos).
Teste = held-out (108 imagens); treino = in-sample (273), mostrado só como referência.

## 1. Clusters no espaço aprendido (z)
- `clusters_treino.png` / `clusters_treino.html` — treino por categoria (clean + 4 erros) + protótipos (★).
- `clusters_teste.png` / `clusters_teste.html` — teste por categoria, **treino em cinza ao fundo** + protótipos (★).
- Os `.html` são **interativos** (plotly): zoom/pan, ligar/desligar classes na legenda, e hover com o
  nome do arquivo (e `p(erro)` no teste). Mesmas coordenadas UMAP dos `.png`.

## 2. Erro vs sem-erro (gate) — matriz de confusão
- `confusion_matrix_binaria_treino.png` · `confusion_matrix_binaria_teste.png`

| split | ACC | Precisão | Recall | F1 | AUROC | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---|
| TREINO (in-sample) | 0.88 | 0.99 | 0.80 | 0.89 | 0.99 | 135/104/1/33 |
| **TESTE (held-out)** | **0.58** | **0.70** | 0.58 | 0.63 | **0.60** | 39/24/17/28 |

> O TREINO é ressubstituição (o modelo já viu) → quase perfeito; **o número que vale é o TESTE**.

## 3. Categoria (clean + 4 erros) — matriz de confusão 5×5
- `confusion_matrix_categoria_treino.png` · `confusion_matrix_categoria_teste.png`
- **Sistema fim-a-fim (2 estágios):** prediz `clean` se o gate disser sem-erro; senão atribui a
  categoria do protótipo de erro mais próximo. (Por isso a coluna `clean` aqui **bate** com a
  matriz binária da §2 — esta é aquela, refinada por categoria.)

| split | Acurácia | F1-macro | AUROC-macro |
|---|---:|---:|---:|
| TREINO (in-sample) | 0.88 | 0.88 | 0.99 |
| **TESTE (held-out)** | **0.38** | **0.29** | **0.60** |

## 4. Quão bem identifica CADA classe (teste held-out)
- `metricas_por_classe.png`

| classe | n | precisão | recall (acerto) | F1 | AUROC | acurácia one-vs-rest |
|---|---:|---:|---:|---:|---:|---:|
| `clean` | 41 | 0.46 | 0.59 | 0.52 | 0.61 | 0.58 |
| `black_bars` | 22 | 0.52 | 0.50 | 0.51 | 0.68 | 0.81 |
| `disordered_layout` | 10 | 0.00 | 0.00 | 0.00 | 0.52 | 0.78 |
| `empty_space` | 14 | 0.23 | 0.21 | 0.22 | 0.60 | 0.81 |
| `overlay` | 21 | 0.38 | 0.14 | 0.21 | 0.61 | 0.79 |

> **Leitura por classe:** *precisão* = dos que o sistema chamou de X, quantos eram X; *recall* = dos
> X reais, quantos o sistema identificou como X **fim-a-fim** (passar no gate **e** acertar a
> categoria) — é o "quão bom em identificar X"; *AUROC* = quão bem a proximidade ao protótipo de X
> ranqueia os X acima do resto (independe de limiar); *acurácia one-vs-rest* = acerto no problema
> binário "é X ou não" (alta e pouco informativa porque a maioria não é X — use recall/AUROC).

*Números planos: `metricas_por_classe.json`.*
