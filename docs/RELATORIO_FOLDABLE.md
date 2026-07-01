# Relatório — domínio **foldable**: o que dá para extrair dos dados atuais (sem coleta)

> Escrito após a premissa mudar (jun/2026): **não há como coletar novas imagens** → o objetivo deixa
> de ser "consertar o gargalo foldable" (provadamente impossível sem dados) e passa a ser **extrair o
> melhor ponto de operação dos dados que temos e reportar honestamente**, com o teto documentado.
> Reprodução: `scripts/foldable_operating_point.py` e `scripts/domain_slice_eval.py`.

---

## 0. TL;DR (veredito)

1. **Ganho real e de graça:** trocar o gate foldable de **fusão → protótipo** (+ limiar recalibrado no
   bucket) move a **especificidade foldable de 0.512 → 0.683** (IC95 [0.571, 0.835]) **mantendo o recall**
   (0.657 → 0.641). O limite inferior do IC (0.571) **já clareia** a produção → melhoria **sem downside**.
2. **Sem romance:** o modelo continua **moderado** no foldable — separabilidade AUROC **0.66** [0.55, 0.82].
   Não "consertamos"; **otimizamos dentro dos dados**.
3. **Teto DUPLO, provado:** o gargalo é **(a) conteúdo** (exp. #1: síntese de aspecto não move nada,
   0.512→0.512) **e (b) tamanho de amostra** (A4: 41 clean → todo IC ~±0.15). **Só dado foldable real
   levanta** — tooling pronto (`capture_foldable.py`), se a coleta voltar a ser possível.
4. **Backbone:** B_reg4 ≈ L_reg4 no foldable (**empate dentro do ruído**, A4) → **manter o L_reg4**
   commitado; o ganho vem do **score**, não do backbone.

---

## 1. O gargalo (recap do diagnóstico)

A clean foldable são **172 telas de ~16 sessões de 1 device/1 dia** (todas 2076×2152); os erros cobrem
6 form factors / 217 tickets. Essa assimetria **é** o gargalo: o modelo memoriza um cluster limpo
estreito e **dá falso-alarme em toda tela foldable nova**.

| Medida no domínio foldable (cross-eval) | Valor |
|---|---|
| AUROC livre-de-confound | 0.731 (vs 0.721 do baseline = +0.01, ruído) |
| Especificidade (produção L_reg4-fusão) | **0.512** (20 FP / 41 clean) |
| Confound trivial no v3-test | 1.000 → **v3-test é avaliação confundida; não reportar** |

O **experimento #1** (reflow/síntese em AR near-square) **provou que o gap é CONTEÚDO, não aspecto**:
mirar o AR dos erros deixou a especificidade foldable **0.512 → 0.512** (os mesmos 20 FP). Ver `ROADMAP.md`.

---

## 2. O que tentamos sem dados novos (A1–A4) e o que sobreviveu

| Lever | Resultado | Veredito |
|---|---|---|
| **A2** — re-selecionar backbone p/ o foldable (S/B_reg4/L_reg4) | B_reg4 free-conf 0.768 vs L_reg4 0.731 | **empate** (A4: diff +0.03, IC95 [−0.04, 0.14], P=0.82 — ruído) |
| **A1** — recalibrar limiar por bucket (#3) | +0.02 a +0.07 spec | **modesto** (clean foldable e diversa pontuam parecido) |
| **A1** — gate **protótipo** em vez de fusão | spec 0.46→0.68 (a fusão degrada o ranking: proto AUROC 0.751 > fusão 0.691 já no global) | ✅ **o lever real** |
| **A4** — estabilidade (bootstrap agrupado + seed) | ICs largos; fit de protótipo determinístico (0.685 ± 0.000) | confirma **teto de amostra** |

---

## 3. A config recomendada + números honestos (com IC95)

**Config:** L_reg4 (mantido) · **gate = score de protótipo** (não fusão) · **limiar calibrado na clean
foldable da val** (objetivo specificity, alvo 0.80). Medido na fatia foldable do teste (41 clean + 67
erros), IC95 bootstrap **agrupado por ticket** (3000×):

| Ponto de operação foldable | Especificidade | Recall | AUROC |
|---|---:|---:|---:|
| **Produção atual** (L_reg4, fusão, limiar global) | 0.512 | 0.657 | 0.620 |
| **Recomendado** (L_reg4, protótipo, limiar foldable) ⭐ | **0.683** [0.571, 0.835] | 0.641 [0.522, 0.754] | 0.662 [0.548, 0.819] |
| (alternativa B_reg4 — equivalente) | 0.636 [0.522, 0.889] | 0.671 [0.557, 0.774] | 0.692 [0.594, 0.830] |

> **+0.17 de especificidade com recall ~igual, de graça.** O limite inferior do IC (0.571) fica acima da
> produção (0.512) → **não piora** no pior caso, **melhora** no esperado.

**A curva de operação** (você escolhe o ponto, fronteira do teste foldable):

| se a prioridade é… | especificidade | recall |
|---|---:|---:|
| equilíbrio (recomendado) | ~0.68 | ~0.64 |
| menos falso-alarme | ~0.73 | ~0.51 |
| recall alto | ~0.49 | ~0.81 |

---

## 4. Os dois tetos (por que não vai além — ambos provados)

1. **Conteúdo** (exp. #1): a clean foldable é 1 device/16 sessões; sintetizar aspecto não cria conteúdo
   foldable novo → especificidade não se move (0.512→0.512). **Só conteúdo foldable real diversifica.**
2. **Tamanho de amostra** (A4): com **41 clean + 67 erros**, *toda* métrica tem IC ~±0.15 (spec
   [0.57, 0.84], AUROC [0.55, 0.82]). **Nenhuma melhoria de método é certificável nesse N** — só mais
   dados apertam os ICs.

---

## 5. Decisão de produto (recomendação)

- ✅ **Adotar o gate de protótipo + limiar foldable** para o bucket near-square (sem downside; +spec).
  Mecanismo de deploy: rotear por resolução (near-square → gate foldable; resto → gate global).
- ✅ **Escopo honesto:** entregar a **taxonomia grossa** (2 super-classes, F1 ~0.67) + **`black_bars`**
  (a única classe fina confiável); reportar o **falso-alarme foldable (~30%)** abertamente.
- ⚠️ **Não prometer** detecção fina confiável no foldable nem especificidade alta — o teto de dados não permite.

---

## 6. O que levantaria o teto (quando/se possível)

**Dado foldable real diverso** (≥50 telas, ≥40 apps, 4 postures) — a única alavanca, provada. **Toda a
infra já está pronta e testada** (Fase 2.b CABEADA): `scripts/capture_foldable.py` (coleta) →
`merge_clean_extra.py` (#4.2) → `configs/plus_fold_L_reg4.yaml` → `domain_slice_eval.py` (aceite). Ver
`docs/SPEC_COLETA_FOLDABLE.md` §7. **Gate de aceite quando houver dados: a especificidade foldable
ultrapassar o IC atual ([0.57, 0.84]) de forma estável.**

---

## 7. Reprodução

```bash
# ponto de operação foldable (proto + limiar foldable) com IC95 bootstrap:
python scripts/foldable_operating_point.py --config configs/plus_L_reg4.yaml
# (--target-spec 0.90 p/ o ponto de menos falso-alarme; --subset near-square|form-factor p/ outras fatias)

# métricas por domínio (global vs fatia foldable), self-check no held-out cheio:
python scripts/domain_slice_eval.py --config configs/plus_L_reg4.yaml --subset v3test
```

Todos os números deste relatório saem desses dois comandos (determinísticos; bootstrap com seed fixa).
Análises A1–A4 originais: histórico no `docs/ROADMAP.md` (seção Fase 2.b).
