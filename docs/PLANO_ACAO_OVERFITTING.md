# Diagnostico de overfitting e plano de acao

## Resumo executivo

Ha evidencia de overfitting no modelo atual, mas o problema principal nao e apenas
"modelo grande demais". A causa raiz mais forte e um confound de dados: todas as telas
`clean` reais estao na mesma resolucao/dispositivo (`2076x2152`), enquanto os erros reais
sao heterogeneos em resolucao, origem e form factor.

Por isso, metricas de treino quase perfeitas e metricas globais altas nao devem ser
apresentadas como evidencia de generalizacao. O resultado honesto deve priorizar:

- subconjunto controlado;
- sonda sintetica livre de confound;
- comparacao contra baseline trivial de resolucao/padding;
- metricas com intervalo de confianca.

## Evidencia observada

Artefatos principais:

- `artifacts/reports/confusion_matrix_treino.png`
- `artifacts/reports/confusion_matrix.png`
- `artifacts/reports/evaluation_report.json`
- `artifacts/reports/EXPERIMENT_RESULTS.md`
- `artifacts/reports/nested_cv_report.json`

Metricas no ponto de operacao atual:

| Cenario | Acc | Precision | Recall | F1 | Balanced Acc | MCC | Especificidade | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Treino | 0.970 | 0.969 | 0.987 | 0.978 | 0.960 | 0.930 | 0.933 | 0.991 |
| Teste held-out | 0.685 | 0.722 | 0.876 | 0.792 | 0.572 | 0.179 | 0.268 | 0.681 |

Matriz de confusao:

| Cenario | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Treino | 98 | 7 | 3 | 222 |
| Teste held-out | 11 | 30 | 11 | 78 |

Conclusao: o gap treino -> teste e grande, principalmente em especificidade/FPR. O modelo
aprende muito bem o conjunto de treino, mas generaliza mal para telas limpas held-out,
marcando muitas telas sem erro como erro.

## Causa raiz

1. Confound de resolucao/dispositivo:
   - `clean`: 172/172 imagens reais em `2076x2152`.
   - erros reais: muitas resolucoes e form factors.
   - baseline trivial de resolucao chega a AUROC perto de `0.99`.

2. Poucos negativos diversos:
   - o modelo quase nao ve telas limpas de outros devices, fotos, landscape, laptop/tent,
     ou apps/form factors diferentes.

3. Split pequeno para calibracao:
   - ha apenas 26 telas limpas reais em validacao e 41 em teste.
   - isso torna limiar, especificidade e precisao instaveis.

4. Estagio 2 com classes raras e rotulos sobrepostos:
   - `orientation` e `distortion` tem suporte muito baixo.
   - categorias como `black_bars`, `empty_space`, `overlay` e `disordered_layout` podem
     coocorrer ou ser visualmente proximas.

5. Historico de avaliacao otimista ja identificado:
   - relatorios atuais indicam que metricas legadas altas vinham de vazamento de
     near-duplicates limpas e snooping no teste. O protocolo atual corrige isso, mas as
     metricas antigas nao devem ser usadas como resultado.

## Verificacao de vazamento/protocolo

Verificacoes executadas:

```bash
.venv/bin/python -m pytest tests/test_protocol_guard.py tests/test_split_isolation.py -q
```

Resultado: `11 passed`.

O protocolo atual tem protecoes importantes:

- teste trancado programaticamente por `siamese.protocol`;
- grid search ranqueado por validacao, nao por teste;
- splits agrupados por ticket e por sessao/near-duplicate das telas limpas;
- `data/processed/` como fonte unica de verdade.

Portanto, no estado atual, a leitura mais correta e:

- houve overfitting e avaliacao otimista no historico do projeto;
- o protocolo atual ja mitiga vazamento/snooping;
- ainda existe baixa generalizacao por causa do confound e da falta de diversidade de
  exemplos limpos.

## Plano de acao

### Fase 1 - Corrigir dados antes de mexer em arquitetura

Objetivo: quebrar o atalho de resolucao/dispositivo.

1. Coletar telas `clean` pareadas com a diversidade dos erros:
   - outros devices e resolucoes;
   - fold/unfold;
   - portrait/landscape;
   - laptop/tent;
   - fotos de tela sem erro;
   - apps/telas diferentes sem erro.

2. Definir uma meta minima de cobertura:
   - pelo menos 30-50 imagens `clean` por estrato importante;
   - negativos limpos nos mesmos form factors dos erros;
   - manter teste held-out separado desde a coleta.

3. Reprocessar o dataset:
   - `scripts/build_splits.py`;
   - `scripts/export_processed.py`;
   - `scripts/extract_features.py`;
   - `scripts/make_synthetic.py`.

### Fase 2 - Reavaliar com protocolo honesto

Objetivo: medir generalizacao real, sem atalho.

1. Manter teste bloqueado ate congelar configuracao.
2. Selecionar hiperparametros apenas por validacao ou nested CV.
3. Reportar sempre:
   - AUROC/AP no subconjunto controlado;
   - AUROC/AP sintetico livre de confound;
   - baseline de resolucao/padding;
   - falseabilidade: score prediz erro vs prediz resolucao;
   - IC95 por bootstrap agrupado;
   - especificidade/FPR, nao apenas F1.

### Fase 3 - Reduzir overfitting operacional

Objetivo: estabilizar decisao e reduzir falso positivo.

1. Usar calibracao livre de confound como padrao.
2. Testar calibracao OOF por grouped k-fold para escolher limiar.
3. Escolher limiar por especificidade minima ou precisao minima com IC95, nao por F1 puro.
4. Monitorar `balanced_accuracy`, `MCC`, `specificity` e `ECE`.
5. Manter reflow-clean e hard negatives, mas medir trade-off em validacao.

### Fase 4 - Melhorar geracao sintetica e negativos benignos

Objetivo: ensinar invariancias corretas.

1. Aumentar sinteticos de erro baseados em casos reais.
2. Criar negativos benignos:
   - mudanca de resolucao sem erro;
   - compressao/blur leve;
   - foto sem erro;
   - variacoes de layout legitimas.
3. Evitar sinteticos que contradigam a taxonomia, especialmente em `orientation` e
   `distortion`.

### Fase 5 - Revisar Estagio 2

Objetivo: reduzir promessa excessiva de classificacao fina.

1. Tratar a taxonomia grossa como primaria.
2. Reportar a taxonomia fina como exploratoria.
3. Migrar para multi-label quando houver coocorrencia de erros.
4. Coletar mais exemplos para `orientation` e `distortion`.
5. Avaliar um classificador simples por categoria como baseline obrigatorio.

## Criterios de aceite

O problema pode ser considerado mitigado quando:

1. O modelo superar claramente o baseline de resolucao no regime controlado.
2. A falseabilidade mostrar `AUROC(prediz erro)` maior que `AUROC(prediz resolucao)` por
   margem material, nao gap proximo de zero.
3. O FPR em telas limpas held-out cair para uma faixa operacional aceitavel.
4. A precisao alvo tiver limite inferior de IC95 compativel com a meta.
5. O gap treino -> teste diminuir em balanced accuracy, MCC e especificidade.
6. As metricas forem reproduzidas por grouped CV ou por um novo teste held-out coletado
   depois da correcao de dados.

---

## Implementacao realizada (jun/2026) — restricao: SEM coletar novas telas

> Restricao do projeto: nao e' possivel coletar novas telas agora. A **Fase 1** (coletar telas
> limpas diversas) foi reinterpretada para o que e' possivel com o dataset ATUAL: **quebrar o
> atalho de resolucao pelo LADO LIMPO**, usando so as imagens limpas existentes (negativos
> benignos + reflow). As demais fases (2-5) foram implementadas. Toda a iteracao/selecao foi
> feita na VAL; o **teste held-out foi tocado UMA unica vez** apos congelar a config (protocolo).

### O que mudou (config congelada em `configs/default.yaml`)

| Campo | Antes (legado) | Depois | Por que (fase) |
|---|---|---|---|
| `synthetic.benign_augment` | false | **true** | negativos LIMPOS com resolucao/qualidade variada (F1/F4) |
| `train.temperature` | 0.05 | **0.1** | re-validado no grid da val — legado nunca re-validado (F2) |
| `train.aux_weight` | 0.6 | **0.3** | idem; menos peso no aux = cluster limpo menos overfitado (F2) |
| `head.proj_dim` | 128 | **64** | grid + estabilidade multi-seed (regra 1-SE): cabeca menor generaliza (F2/F3) |
| `decision.objective` | f1 | **specificity** | headline specificity-first, nao F1 puro — corta o FPR (F3.3) |

Codigo: `src/siamese/evaluate.py` passou a reportar **IC95 de precisao e especificidade** no
ponto de operacao + IC95 da precisao nos limiares por alvo (criterio 4). Novo
`scripts/stage2_baseline.py` = baseline obrigatorio do Estagio 2 (F5.5). Selecao honesta:
`grid_search.py` (val_synth_gate) + `multiseed_stability.py` (1-SE) + `nested_cv.py` (OOF).

### Resultado HELD-OUT (teste tocado 1x; antes -> depois)

| Metrica | Antes | Depois |
|---|---:|---:|
| Especificidade | 0.268 | **0.634** |
| FPR | 0.732 | **0.366** |
| MCC | 0.179 | **0.385** |
| Balanced accuracy | 0.572 | **0.699** |
| Precisao | 0.722 | **0.819** (IC95 0.62-0.99) |
| Recall | 0.876 | 0.764 |
| F1 | 0.792 | 0.791 |
| Acuracia | 0.685 | 0.723 |
| AUROC (fusao / proto) | 0.681 | 0.715 / 0.744 |
| ECE | 0.158 | 0.136 |
| Matriz de confusao | TP78 TN11 **FP30** FN11 | TP68 TN26 **FP15** FN21 |
| Gap treino->teste (bAcc / MCC / espec) | 0.39 / 0.75 / 0.67 | **0.17 / 0.30 / 0.37** |
| FP-rate em reflow (limpo re-arranjado) | 0.415 | **0.244** |

Sinal honesto de conteudo (mantido): **controlado** AUROC 0.693 vs confound 0.383 (margem +0.31);
**sintetico livre de confound** AUROC 0.713 / AP 0.887. Os falso-alarmes em telas limpas caíram
pela metade (FP 30 -> 15) e o gap treino->teste tambem, sem perder F1.

### Reproducao por grouped nested-CV (criterio 6)

OOF (5 folds x 3 seeds, agrupado por sessao/ticket; teste NAO tocado): AUROC **0.700 +- 0.032**
(IC95 0.44-0.82), AP **0.803 +- 0.024**, MCC **0.382 +- 0.035**. Bate com o held-out (AUROC 0.715,
AP ~0.80, MCC 0.385) -> o numero do teste **nao e' sorte de split**.

### Estagio 2 — baseline obrigatorio (Fase 5.5)

`scripts/stage2_baseline.py` (features DINOv2 CRUAS, val): LogReg F1-macro fina **0.19** / grossa
**0.29**; centroide **0.18** / **0.34**. O decisor por **prototipo aprendido** supera (fina ~0.26-0.32,
grossa ~0.39) -> a cabeca siamesa agrega valor na categorizacao (modesto, sobretudo na fina).
Grossa = primaria; fina = exploratoria; **multi-label** fica como trabalho futuro p/ erros que
coocorrem. (proj_dim=64 favorece o gate; custa um pouco no Estagio 2 fino — trade aceito porque o
plano prioriza o gate e rebaixa a classificacao fina.)

### Veredito por criterio de aceite

1. **Supera o baseline de resolucao no controlado** — ✅ proto 0.693 vs confound 0.383 (+0.31).
2. **Falseabilidade com margem material** — ⚠️ **parcial (estrutural)**: no teste prediz-erro
   0.715 vs prediz-resolucao 0.705 (margem ~0.01). A metrica e' DEGENERADA neste dataset (so ~2
   erros na resolucao canonica -> "prediz erro" ~= "prediz resolucao" por construcao). O sinal
   REAL de falseabilidade e' o **controlado (+0.31)** e o **sintetico (0.713, sem diferenca de
   resolucao)** — ambos ✅. Fechar o gap exige erros em resolucao canonica (novas telas).
3. **FPR aceitavel em limpas held-out** — ✅ FPR 0.732 -> **0.366** (especificidade 0.27 -> 0.63);
   nos pontos de alta-precisao (alvo 0.90/0.95) o FP cai p/ 11/9 de 41 limpas.
4. **IC95 da precisao alvo** — ⚠️ **parcial (limite de dados)**: precisao-ponto **0.82** (IC95
   0.62-0.99); por alvo a precisao fica 0.81-0.85, mas o **limite inferior do IC95 (~0.60-0.64) nao
   alcanca 0.90/0.95** — o teste (130 imgs, 41 limpas) e' pequeno demais p/ certificar >=0.90 a 95%.
   A maquinaria de IC agora existe e e' reportada (antes nao era).
5. **Gap treino->teste menor** — ✅ bAcc 0.39->0.17, MCC 0.75->0.30, especificidade 0.67->0.37.
6. **Reproduzido por grouped CV** — ✅ OOF AUROC 0.700+-0.032 ~= teste 0.715.

**Resumo:** 4/6 criterios plenamente atingidos (1, 3, 5, 6) e 2 parcialmente (2, 4) — estes dois
limitados ESTRUTURALMENTE pelo dataset (resolucao confundida + n pequeno), nao pelo modelo. Subir
o teto deles exige exatamente o que a Fase 1 original pedia (telas limpas pareadas em mais
resolucoes), fora do escopo "sem novas telas". Dentro da restricao, o **overfitting operacional
foi materialmente reduzido**: FPR ~metade, gap treino->teste ~metade, MCC ~2x, sem perder F1, e a
generalizacao foi reproduzida por grouped CV.
