# Relatório final — detector de erro de layout em `processed_v3`

> **Setup principal** (o que vai para os demais times). Modelo: cabeça siamesa sobre DINOv2 ViT-S/14
> congelado, decisão em 2 estágios. Config congelada: `configs/default.yaml`. Dataset: `data/processed_v3`
> (plano + `labels.csv`). Teste held-out avaliado **1×** após congelar a config. Números reproduzidos
> de ponta a ponta (`scripts/run_experiment.py`), determinísticos.

---

## 0. Veredito (TL;DR)

1. **Não há vazamento.** Splits agrupados por ticket: **0 grupos** compartilhados entre train/val/test;
   o teste fica fisicamente trancado (`siamese.protocol`) durante toda a seleção.
2. **Há confound de resolução nos DADOS, mas o modelo NÃO o explora.** Provado por 6 ângulos
   independentes (§2). O **AUROC 1.000** que assustava é o *baseline trivial de resolução* (um
   diagnóstico que medimos de propósito) — **não é o modelo**. O modelo dá AUROC 0.60–0.72.
3. **Desempenho honesto e moderado:**
   - **Detecção erro/não-erro** (medida justa, livre de confound): **AUROC 0.72** (acaso 0.50),
     acurácia balanceada **0.68**. Nos erros reais, no ponto de operação: **acurácia 0.58,
     precisão 0.70**.
   - **Tipo de erro** (k-NN de categoria): grosso (2 super-classes) **acurácia 0.64 / F1 0.64**;
     fino (4 classes) **acurácia 0.43 / F1 0.35** — confiável só em `black_bars`.
4. **Melhor métrica para medir aprendizado:** **AUROC livre de confound** (a sonda sintética de
   resolução constante). É a única que não dá para fraudar com o atalho do device, é independente
   de base-rate e de limiar, e já é o critério de early-stop. **Evite** liderar com acurácia/AUROC
   global no real (confundidos) e com **AP nessa sonda** (acaso 0.80 — superestima).

---

## 1. O dataset e o confound (a causa de tudo)

| split | clean (real) | erro (real) | sintético (treino) | clean @2076×2152 | erro @2076×2152 |
|---|---:|---:|---:|---:|---:|
| train | 105 | 168 | 419 | **105/105** | 2/168 |
| val   | 26  | 42  | —   | **26/26**  | 2/42 |
| test  | 41  | 67  | —   | **41/41**  | **0/67** |

**100% das telas limpas reais são exatamente 2076×2152** (um único device). Os erros vêm de muitas
resoluções/devices/form-factors. No teste **nenhum** erro está na resolução canônica. Consequência:
a regra trivial *"resolução ≠ 2076×2152 ⇒ erro"* separa o teste **perfeitamente**:

| baseline trivial (só metadados, zero conteúdo) | AUROC | acurácia |
|---|---:|---:|
| "resolução ≠ 2076×2152 ⇒ erro" | **1.000** | **1.000** |
| fração de padding cinza | **1.000** | — |

> ⚠️ É **este** o "AUROC 1.000 (confound total)". Ele mede o quanto os **dados** são fraudáveis —
> é o **teto de trapaça**, calculado de propósito para sabermos contra o que comparar. **Não é o
> modelo.** O confound vai além da resolução: a tela limpa é uma fonte homogênea
> (`form_factor=unknown`, `orientation=unknown`, `kind=screenshot`), enquanto o erro varia em tudo.

---

## 2. O modelo trapaceia? **Não.** (6 provas independentes)

| # | Teste | Resultado | Leitura |
|---|---|---|---|
| 1 | Baseline trivial de resolução | AUROC **1.000** | teto de trapaça (diagnóstico, não o modelo) |
| 2 | Modelo no **mesmo** teste real | AUROC **0.607 / 0.596** (proto/fusão) | muito **abaixo** de 1.0 → não usa o atalho |
| 3 | Se trapaceasse, acc/prec/rec/espec → ~1.0 | observado **0.58 / 0.70 / 0.58 / 0.59** | todas longe de 1.0 → refuta trapaça |
| 4 | AUROC livre-confound vs AUROC real | **0.721 > 0.607** | um trapaceiro seria o **inverso** (alto no real, ~acaso sem confound) |
| 5 | Correlação score × geometria (real) | corr(fused, padding)=**0.05**, corr(fused, aspect)=**−0.19**, corr(proto, padding)=**0.04** | o score **não segue** a resolução |
| 6 | Sonda de **resolução constante** (clean e erro ambos 2076×2152) | separa a **AUROC 0.721 / AP 0.903** | a separação **não pode** vir da resolução → é **conteúdo** |

**Conclusão:** o atalho existe nos dados, mas o modelo aprendeu **conteúdo de erro**, não o device.
As defesas que garantem isso (todas ativas na config): erros sintéticos injetados nas telas limpas
**na mesma resolução** (quebra o atalho pelo lado do erro), negativos *reflow*/benignos com
resolução variada (quebra pelo lado limpo), e estatísticas de patch com **máscara de padding**.

---

## 3. Pergunta 1 — "A tela tem erro ou não?" (Estágio 1)

**Sim, o modelo distingue, com sinal real porém moderado.** Duas leituras (ambas honestas):

| Regime de avaliação | Acurácia | Precisão | Recall | Especificidade | AUROC |
|---|---:|---:|---:|---:|---:|
| **Livre de confound** (clean real vs erro mesma-resolução) — *medida justa* | **0.68** ᵇ | — | 0.78 | 0.585 | **0.719** |
| **Erros reais**, ponto de operação — *rendimento no mundo real* | **0.583** | **0.696** | 0.582 | 0.585 | 0.596 |
| (Treino, ressubstituição — **não é resultado**) | 0.875 | 0.993 | 0.804 | 0.990 | 0.991 |

ᵇ acurácia **balanceada** (neutra à proporção de classes). IC95 (teste real, bootstrap por ticket):
acurácia **[0.49–0.67]**, precisão **[0.46–0.95]**, F1 **[0.49–0.76]**, especificidade **[0.25–0.74]**.
Matriz de confusão (teste real, n=108): **TP=39 · TN=24 · FP=17 · FN=28**.

**Por que duas acurácias (0.68 vs 0.58)?** A taxa de falso-alarme em tela limpa real é a **mesma**
(especificidade 0.585); a diferença vem do **lado do erro**: erros sintéticos são mais nítidos
(recall 0.78) que os erros reais, mais sutis (recall 0.58). Ou seja: **0.72 / 0.68 é o teto de
capacidade do modelo**; **0.58 / 0.70 é o que ele rende nos erros reais que temos**.

**Se o coordenador priorizar precisão alta** (menos falso-alarme), o trade-off real é (limiar fixado
na validação):

| alvo de precisão | precisão (teste) | recall (teste) | erros pegos |
|---|---:|---:|---:|
| 0.90 | 0.75 | 0.224 | 15 de 67 |
| 0.95 | 0.667 | 0.149 | 10 de 67 |

> O teste é pequeno (41 limpas) → não dá para **certificar** precisão ≥ 0.90 (limite inferior do
> IC95 não alcança). Honestamente: o modelo entrega precisão ~0.70 com recall ~0.58, ou troca
> recall por precisão na curva acima.

---

## 4. Pergunta 2 — "Que tipo de erro é?" (Estágio 2, só em telas de erro)

O Estágio 2 atribui categoria **apenas** a telas já classificadas como erro (**k-NN de categoria**
no espaço aprendido — adotado por dar ganho modesto sobre o protótipo, ver
[`COMPARACAO_KNN_TRIPLET.md`](COMPARACAO_KNN_TRIPLET.md)). Duas taxonomias:

| Taxonomia | Acurácia | F1-macro | IC95 (F1) | leitura |
|---|---:|---:|---|---|
| **Grossa — 2 super-classes** (`dead_region`, `displaced_content`) ⭐ | **0.642** | **0.641** | [0.52–0.75] | **primária** (tem poder estatístico) |
| Grossa, condicionada ao gate (produção) | 0.667 | 0.643 | — | só erros que o Estágio 1 pegou |
| Fina — 4 classes | 0.433 | 0.354 | [0.25–0.43] | exploratória (acaso 0.25) |

**Por classe (taxonomia fina, oráculo):**

| classe | precisão | recall | F1 | detecção (recall@op, Est.1) | suporte |
|---|---:|---:|---:|---:|---:|
| `black_bars` | **0.786** | 0.500 | **0.611** | 0.727 | 22 |
| `overlay` | 0.424 | 0.667 | 0.519 | 0.476 | 21 |
| `empty_space` | 0.286 | 0.286 | 0.286 | 0.571 | 14 |
| `disordered_layout` | 0.000 | 0.000 | 0.000 | 0.500 | 10 |

**Leitura honesta:**
- **Funciona bem só em `black_bars`** (precisão 0.79, melhor detectada e melhor classificada).
- `overlay` é mediano; `empty_space` e `disordered_layout` são fracos (este último, com n=10,
  fica em zero — sem poder estatístico e visualmente confundível com os demais).
- **Reporte a taxonomia grossa como primária** (acc 0.64) e a fina como exploratória. A fina sobe
  pouco acima do acaso (0.25) fora de `black_bars`.

---

## 5. Qual métrica usar para medir o aprendizado?

**Primária: AUROC na sonda livre de confound** (`val_synth_gate` — erros injetados nas telas limpas,
**mesma resolução**). Motivos:

1. **Imune ao atalho:** a resolução é constante na sonda, então é impossível pontuar bem lendo o
   device — sobe **só** se o modelo aprender conteúdo.
2. **Independente de base-rate:** acaso = 0.50 sempre (não infla com a proporção de erros).
3. **Independente de limiar:** não depende da calibração, que aqui transfere mal (só 26–41 telas
   limpas → especificidade-alvo 0.80 vira 0.585 no teste).
4. **Já é o critério de early-stop** → o treino otimiza exatamente o que medimos.

**Evite como métrica-guia:**

| métrica | por que enganosa aqui |
|---|---|
| Acurácia/AUROC **global no real** | confundidos: o teto de trapaça é 1.000 |
| **AP na sonda sintética** | a sonda é 80% positiva → **acaso da AP = 0.80**; AP 0.90 é só +0.10 (já o AUROC 0.72 é +0.22 sobre o acaso) |
| Métricas de **treino** | ressubstituição (AUROC 0.99) — não é generalização |
| Acurácia **crua** na sonda | base-rate 80% → "chutar tudo erro" já dá 0.80; use **acurácia balanceada** |

> Para a tabela comparativa entre times, lidere com **AUROC livre de confound** e exija que o
> concorrente **supere o baseline de confound** (§1). Um time que reportar "AUROC 0.95 global" pode
> estar apenas lendo a resolução.

---

## 6. Limitações estruturais (honestidade) e o que elevaria o teto

- **A tela limpa é uma fonte única** (1 device, 1 resolução, 1 form-factor). Por isso (a) não dá
  para medir detecção em erros reais **sem** confound (clean e erro nunca compartilham resolução —
  0 erros canônicos no teste), e (b) a especificidade/precisão no mundo real ficam instáveis (poucas
  limpas). Isto é **limite de dados**, não bug do modelo.
- **O que sobe o teto (única alavanca real):** coletar telas **limpas diversas** — outros devices,
  resoluções, fold/unfold, portrait/landscape, fotos — pareadas com a diversidade dos erros. Aí o
  confound some e a métrica real passa a valer.
- **Gap treino→teste** (AUROC 0.99 → 0.60): esperado — o treino também é confundido (fácil de
  ajustar) e é ressubstituição; o número que vale é o livre-de-confound, reproduzido por
  grouped-CV em rodadas anteriores (AUROC OOF ~0.70).

---

## 7. Reprodução

```bash
.venv/bin/python scripts/run_experiment.py --config configs/default.yaml --processed data/processed_v3
```

Saídas: `artifacts/reports/EXPERIMENT_RESULTS.{md,json}` (tabela + veredito automático),
`artifacts/reports/evaluation_report.json` (métricas completas do teste held-out),
`artifacts/reports/confusion_matrix*.png`, `metricas_por_classe.png`. Todos os números deste
relatório foram reproduzidos nesta sessão (determinístico, early-stop no epoch 97).
