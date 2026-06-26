# Comparação empírica — k-NN vs protótipo · Triplet vs SupCon

Responde às perguntas da supervisão com **dados** (não opinião). Tudo selecionado na **validação
livre de confound**; o teste held-out foi tocado para reportar o modelo congelado sob regras de
decisão **pré-selecionadas na val** (sem snooping). Implementação atrás de flags
(`decision.gate_method`, `decision.stage2_method`, `train.loss`, `decision.knn_k`,
`train.triplet_margin`); o baseline congelado (`supcon` + `prototype`) permanece o default.
Reprodução: `python scripts/compare_methods.py` (val) + `tests/test_knn_triplet.py`.

---

## TL;DR (veredito)

| Pergunta | Resposta curta |
|---|---|
| **B) k-NN no lugar de centróides** | **Não melhora a detecção (gate)** — até reduz a especificidade; **melhora modestamente a categoria** (Estágio 2). |
| **D) Triplet no lugar de SupCon** | **SupCon vence claramente** no gate (livre-confound 0.79 vs 0.59); o Triplet só "ganha" no Estágio 2 de um modelo com gate fraco. |
| **A) Cabeça classificar 5 grupos** | Já faz (aux multi-classe + SupCon 5 classes); o 2-estágios é mantido para **controlar o falso-alarme** (limiar). |
| **C) "k do k-means = 5"** | Confusão de termos: `k_prototypes` = sub-clusters POR classe, não nº de classes. Com k-NN o k-means some. |

**Conclusão de engenharia:** a única mudança com ganho real e validado é **`stage2_method: knn`**
(categoria coarse 0.626→0.641 no teste, gate idêntico). k-NN no gate e Triplet **não** sobem a
acurácia/precisão de detecção — o gargalo é o **confound de dados** (um device de telas limpas), não
a regra de decisão nem a perda.

---

## 1. Validação (DEV, livre de confound — teste NÃO tocado)

`synthAUROC` = gate livre de confound na val (métrica primária, acaso 0.50). `spec/acc/prec/f1` =
ponto de operação na val (in-sample, comparativo relativo). `E2grossa` = F1-macro coarse (método
canônico); `fina(p/k)` = fina por protótipo / por k-NN.

| loss | gate | stage2 | synthAUROC | spec | acc | prec | E2grossa | fina(p/k) |
|---|---|---|---:|---:|---:|---:|---:|---|
| **supcon** | **prototype** | **prototype** | **0.794** | 0.769 | 0.676 | 0.812 | 0.664 | 0.412/0.374 |
| supcon | prototype | knn | 0.794 | 0.769 | 0.676 | 0.812 | **0.686** | 0.412/0.374 |
| supcon | knn | prototype | 0.775 | 0.731 | 0.662 | 0.788 | 0.664 | 0.412/0.374 |
| supcon | knn | knn | 0.775 | 0.731 | 0.662 | 0.788 | 0.686 | 0.412/0.374 |
| triplet (m=0.2) | prototype | prototype | 0.591 | 0.962 | 0.588 | 0.938 | 0.667 | 0.424/0.529 |
| triplet (m=0.5) | prototype | knn | 0.591 | 0.962 | 0.588 | 0.938 | **0.762** | 0.424/**0.529** |

- **k-NN no gate:** 0.775 < 0.794 → **pior** que o protótipo. Não ajuda a detecção.
- **Triplet:** synthAUROC 0.59 (≈ acaso) — o gate quase não separa clean de erro. A "precisão 0.94 /
  especificidade 0.96" é um **ponto de operação degenerado** (sinaliza pouquíssimo; alta precisão por
  marcar só o óbvio, mas recall baixíssimo) — não é qualidade real. Ambas as margens convergem ao mesmo
  ótimo ruim → instabilidade do batch-hard mining (exatamente o que a literatura/SupCon evitam).
- **k-NN no Estágio 2:** coarse **0.686 vs 0.664** (supcon) e **0.762 vs 0.667** (triplet) → **melhora a
  categoria**. (No triplet o ganho é maior, mas sobre um gate quebrado → irrelevante em produção.)

## 2. Teste held-out (modelo **supcon congelado** · 4 regras de decisão)

Mesmo modelo dos relatórios oficiais; só muda a regra de decisão (pré-selecionada na val).

| gate | stage2 | synthAUROC | acc | prec | **spec** | gAUROC | **E2grossa** | fina(p/k) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **prototype** | **prototype** (baseline) | 0.721 | 0.58 | 0.70 | **0.59** | 0.596 | **0.626** | 0.336/0.354 |
| prototype | **knn** | 0.721 | 0.58 | 0.70 | 0.59 | 0.596 | **0.641** ↑ | 0.336/0.354 |
| knn | prototype | 0.727 | 0.60 | 0.69 | **0.54** ↓ | 0.588 | 0.626 | 0.336/0.354 |
| knn | knn | 0.727 | 0.60 | 0.69 | 0.54 | 0.588 | 0.641 | 0.336/0.354 |

- **Gate k-NN:** synthAUROC +0.006 (ruído), mas **especificidade −0.05** (0.59→0.54) e gAUROC −0.008 →
  **empate/leve piora**. Não é melhoria.
- **Estágio 2 k-NN:** coarse **0.626→0.641** (+0.015), consistente com a val → **ganho real, modesto**,
  com gate idêntico (Pareto).

## 3. Por que o k-NN não conserta a especificidade (a hipótese do roteiro)

O roteiro previu que o k-NN subiria a especificidade modelando o manifold limpo **multimodal**
(telas de apps diferentes). **Os dados não confirmam:** a especificidade *caiu* com o gate k-NN.
Razão: as telas limpas de **teste são do mesmo device/resolução (2076×2152)** das de treino — o
"gap de domínio" é de **app/conteúdo**, não de device. O cluster limpo já é bem capturado por
poucos protótipos; a flexibilidade extra do k-NN sobretudo **encolhe a margem** (um erro perto de
*uma* limpa de treino recebe score baixo) → mais falso-negativo de margem / menos especificidade.
O baixo desempenho real continua sendo **limite de dados** (um único device de telas limpas), não
da regra de decisão.

## 4. Recomendações

1. **Detecção (gate):** manter **`supcon` + `gate_method: prototype`** — é o melhor na métrica honesta.
2. **Categoria (Estágio 2):** **`stage2_method: knn`** dá ganho pequeno e consistente (coarse
   0.626→0.641; fina 0.336→0.354) sem custo no gate. **ADOTADO como default** (jun/2026) — os
   relatórios oficiais já refletem o k-NN; o protótipo fica como diagnóstico/ablação.
3. **Triplet:** **não adotar** — pior e instável; fica disponível (`train.loss: triplet`) só para o
   comparativo acadêmico (responde à pergunta D com dado: SupCon ≥ Triplet aqui).
4. **A alavanca real** para subir acurácia/precisão de detecção continua sendo **coletar telas limpas
   diversas** (outros devices/resoluções), não a regra de decisão nem a perda — ver
   [`RELATORIO_FINAL_PROCESSED_V3.md`](RELATORIO_FINAL_PROCESSED_V3.md) e [`DESIGN.md`](DESIGN.md).

## Flags (default = melhor configuração validada)
```yaml
decision:
  gate_method: prototype  # DEFAULT (melhor). 'knn' disponível p/ ablação (pior especificidade)
  stage2_method: knn      # DEFAULT ADOTADO (ganho modesto). 'prototype' p/ reverter / diagnóstico
  knn_k: 5
train:
  loss: supcon            # DEFAULT (melhor). 'triplet' disponível p/ ablação (instável)
  triplet_margin: 0.5     # só usado com loss=triplet
```
