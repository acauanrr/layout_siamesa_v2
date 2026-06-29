# Design — Rede Siamesa para detecção de erro de layout em UI

> Backbone: **DINOv2 ViT-S/14** (congelado). Objetivo: dada uma imagem/print de tela, decidir em
> **dois estágios** — (E1) **tem erro de layout?** e (E2) **de que tipo?** — sem cair no confound dos dados.

Este documento (1) critica a formulação siamesa clássica para *este* problema, (2) reporta o achado
que muda tudo — o **confound de resolução** —, (3) justifica o design implementado, e (4) mostra os
resultados honestos medidos em `data/processed_v3`. Números atuais em
[`RELATORIO_FINAL_PROCESSED_V3.md`](RELATORIO_FINAL_PROCESSED_V3.md); diagrama em [`pipeline.mmd`](pipeline.mmd).

> **🔄 Atualização (jun/2026 — Fases 2–4, ver [`ROADMAP.md`](ROADMAP.md)):** ao design abaixo somaram-se
> duas alavancas, **sem mudar o núcleo**: (1) **ingestão de telas limpas públicas multi-resolução**
> (`fetch_clean_extra.py` + `merge_clean_extra.py`) que QUEBRA o confound na origem — cada limpa nova
> gera também erros sintéticos casados na mesma resolução; (2) **backbone maior com registers**
> (`vit_large_patch14_reg4_dinov2`; `backbone.py` agora pula `num_prefix_tokens` nas stats de patch).
> Resultado: AUROC livre-de-confound 0.72 → **0.80**, gap treino→teste 0.40 → **0.18** (estável
> multi-seed). O caminho **binário legado** (`multiclass=false`) segue funcional e marcado `LEGADO`
> (mantido como fallback documentado, não removido).

---

## 1. Resumo executivo (leia primeiro)

**O dado tem um confound quase perfeito.** As **172** telas `clean` são **todas** do mesmo device e
resolução (**2076×2152**, uma única sessão de captura). Os **277** erros reais são heterogêneos
(muitas resoluções, fotos de câmera, fold/unfold/laptop/tent). Consequência medida no teste:

| Classificador | AUROC (teste) |
|---|---|
| **Regra trivial "resolução ≠ 2076×2152 ⇒ erro"** | **1.000** |
| Fração de padding cinza | 1.000 |
| LogReg sobre DINOv2 cru | 0.72 |
| **Modelo (protótipo)** | **0.61** |

Ou seja: **qualquer métrica global é ~98% trapaça** — basta olhar a resolução. Um modelo que "acerta
95%" no teste global provavelmente detecta *device*, não *erro*.

**Consequência de design:** "tem erro de layout?" só pode ser aprendido/medido honestamente se
quebrarmos o confound. Fazemos isso **injetando erros sintéticos nas próprias telas limpas** (mesma
resolução), criando pares onde *só o conteúdo do erro muda*. Treinado assim, o modelo detecta erro de
**conteúdo** de forma **modesta** (held-out: AUROC **0.72** livre de confound), e — verificado — **não
explora o atalho**: se explorasse, todas as métricas reais seriam ~1.0; são ~0.58 (as 6 provas estão
no relatório de resultados).

**Recomendação nº 1 (de dados, não de arquitetura):** para generalizar é preciso coletar telas
`clean` que cubram a diversidade dos erros — outros devices/resoluções, fotos, landscape, laptop/tent.
Nenhuma arquitetura supera essa lacuna de dados.

---

## 2. Crítica à formulação siamesa clássica

A descrição clássica ("parear o alvo `x₁` com uma referência-de-sucesso `x₂` e perguntar se diferem")
é **mal-posta aqui**, porque pressupõe **uma** referência boa canônica. Mas a classe `clean` é
**visualmente diversa**: telas de onboarding de apps diferentes (cores, ilustrações, idiomas). Duas
telas **limpas** de apps diferentes são legitimamente **dissimilares** — treinar "alvo vs uma
referência boa" ensinaria a confundir **"tela diferente"** com **"tela errada"** ⇒ falso-positivo
estrutural em qualquer tela nova.

**Veredito:** mantém-se a *estrutura* siamesa (ramos de pesos compartilhados + comparação no espaço de
embeddings), mas muda **o referente da comparação**: em vez de *uma* imagem de referência, comparamos
contra **múltiplos protótipos do manifold `clean`** (a ideia de clustering do pedido, que está
**certa**). Tecnicamente é uma **rede siamesa one-class / de metric learning**. O treino usa
**Supervised Contrastive** (estável com poucos dados, sem exigir referência única); o vetor de fusão
pareado `[z₁,z₂,|z₁−z₂|,z₁⊙z₂]` (`SiamesePairHead`) e a Contrastive Loss (`losses.py`) ficam
disponíveis como alternativas.

---

## 3. Os dados (`data/processed_v3`)

Dataset **plano** + `labels.csv` (reconstruído por `scripts/rebuild_processed_v3.py`: dedup
717→**449 reais únicos**, re-split **agrupado por ticket** com **0 vazamento**, sintéticos
regenerados). Auditoria: `scripts/audit_dataset.py`.

| split | clean (real) | erro (real) | sintético | clean @2076×2152 | erro @2076×2152 |
|---|---:|---:|---:|---:|---:|
| train | 105 | 168 | 419 (+420 reflow) | 105/105 | 2/168 |
| val | 26 | 42 | sonda | 26/26 | 2/42 |
| test | 41 | 67 | sonda | 41/41 | **0/67** |

| Atributo | clean | erro | Confound? |
|---|---|---|---|
| Resolução 2076×2152 | 172/172 | ~6/277 | **Crítico** — separa quase perfeito |
| Form factor | único (`unknown`) | fold/unfold/laptop/tent | **Forte** |
| Foto de câmera | 0 | dezenas | **Forte** |
| Orientação landscape | 0 | algumas | Forte |

O split é **agrupado por ticket** (`IKSWW-\d+`) e por **sessão/near-dup** das limpas — imagens do mesmo
bug/sessão nunca cruzam train/test (0 grupos cruzando; travado por `tests/test_split_isolation.py`).
**Taxonomia (jun/2026):** reduzida de 6→**4 erros** — `distortion` e `orientation` removidas (sem
suporte no v3). Classes: `clean` · `black_bars` · `disordered_layout` · `empty_space` · `overlay`.

---

## 4. Design implementado

```
imagem ─► padding CINZA até quadrado + resize 518×518 (+ máscara de patch) ─► DINOv2 ViT-S/14 ❄ CONGELADO
                                              │  CLS 384 + mean/std dos patch tokens de CONTEÚDO (37×37)
                                              ▼  = vetor base 1152-d  (cacheado em .npz)
                          cabeça de PROJEÇÃO g(·) (pesos compartilhados, ~314k treináveis):
                          LayerNorm → Linear(1152,256) → GELU → Drop(0.3) → Linear(256,64) → L2-norm
                                              ▼
                          z ∈ S⁶³  ──► (A) score de protótipo: 1 − cos(z, protótipo-limpo-mais-próximo)
                              │          (B) cabeça auxiliar: Linear(64→5) → P(erro) = 1 − softmax[clean]
                              ▼
                       FUSÃO calibrada (LogReg de 2 features) ─► p(erro) ─► limiar
           perda: L = SupCon(z, τ=0.1) + 0.3 · CE(aux, 5 classes)
```

**Por que cada escolha:**

- **DINOv2 congelado.** Com ~450 imagens, fine-tunar 22M params overfittaria os confounds. Toda a
  capacidade treinável fica na cabeça (~314k); embeddings são fixos ⇒ **cacheados** e o treino roda em segundos.
- **Pré-processamento `pad` (padrão).** 518 é a única resolução do checkpoint (37×37 patches). `pad` =
  padding até quadrado **preservando o aspecto**, preenchido com **cinza neutro = média ImageNet** (vira
  ~0 após Normalize; distinto de preto e do fundo). **Preserva a geometria do erro** (uma faixa preta
  não é espremida). O risco — a *área* de cinza correlacionar com aspect-ratio — é neutralizado
  calculando `mean/std` dos patches **só na região de conteúdo** (`content_patch_mask`, `geometry.py`).
  Auditoria: a fração de padding cinza dá AUROC 1.000 *no DINO cru*, mas o **masking** impede que vaze
  para as patch-stats (768 das 1152 dims). O CLS (384 dims) ainda vê a borda — por isso a verificação
  empírica de que o modelo não a explora (corr(score, padding)≈0.05).
- **CLS + estatísticas de patch (1152-d).** black-region/empty-space são anomalias de **homogeneidade
  espacial**: o desvio-padrão dos patches cai em regiões grandes uniformes (`use_patch_stats: true`).
- **Cabeça de projeção compartilhada = a parte "siamesa".** A mesma `g` para qualquer imagem; comparar
  duas = comparar `z₁,z₂`. Treinada com **SupCon** (limpo vira cluster compacto, erro cai fora).
- **Cabeça auxiliar (B).** Detector direto que **não** depende do banco de referências — evita o
  detector-de-novidade puro (que dispararia em qualquer app novo). Multi-classe (5): o gate é `1−P(clean)`.
- **Injeção de erros sintéticos** (`synthetic.py`) — **a alavanca anti-confound** (§7).

### 4.1 Regra de decisão = clustering do pedido (corrigido)
- **k protótipos** do cluster `clean` via k-means sobre os `z` de treino (`fit_prototypes`, k=3);
  `score_proto(x) = 1 − cos(z_x, protótipo mais próximo)`.
- **Fusão calibrada** de `[score_proto, aux_err]` por LogReg (**não** `max`/OR, que derruba precisão).
- **Limiar** fixado na **validação livre de confound** — padrão **specificity-first** (alvo 0.80);
  alternativas `f1` e `precision` (alvo 0.90/0.95).

---

## 5. Resultados (teste held-out · `processed_v3`)

Config congelada após seleção íntegra na val + estabilidade multi-seed/1-SE; teste = **108 imagens**
(41 limpas + 67 erros), processado **uma única vez** (`evaluate.py --final-test`). Gate = **protótipo**;
categoria (Estágio 2) = **k-NN** (`stage2_method: knn`, ver §8).

### Estágio 1 — "tem erro?"
| Avaliação (TESTE) | Modelo | Baseline de confound | Leitura |
|---|---|---|---|
| **Sintético livre de confound** (41 limpas vs 164 erro, mesma resolução) | **AUROC 0.72** (AP 0.90) | — | ✅ sinal real (medida justa) |
| **Controlado** (form-factor/orientação fixos) | **0.60** | 0.32 | ✅ supera o confound |
| **Global** | 0.60 | resolução trivial **1.000** · padding **1.000** | ↓ de propósito (não trapaceia) |

Ponto de operação (`specificity`, calibração livre de confound): acc **0.58** · precisão **0.70** ·
recall 0.58 · especificidade 0.585 · F1 0.63 · MCC 0.16 (TP39/TN24/FP17/FN28; IC95 acc 0.49–0.67).
⚠️ A **AP 0.90** da sonda **engana** (80% positivos → acaso 0.80); o sinal real é o **AUROC 0.72**
(acaso 0.50). A acurácia honesta livre-de-confound é a **balanceada 0.68**.

### Estágio 2 — categoria (n=67 erros, decisor **k-NN**)
Taxonomia **grossa primária** (2 super-classes): F1-macro **0.64** · acc 0.64 [IC95 0.52–0.75].
Taxonomia **fina** (4 classes): F1-macro **0.35** · acc 0.43. `black_bars` é a classe forte
(precisão 0.79, F1 0.61); `disordered_layout`/`empty_space` são fracas. Artefatos visuais e métricas
por classe: `artifacts/reports/processed_v3/` (`scripts/report_processed_v3.py`).

> **Veredito honesto:** globalmente o modelo **não vence** o confound (0.60 vs 1.000) — *de propósito*,
> pois não o explora. Há sinal de layout **real porém modesto** no sintético (0.72) e no controlado
> (0.60 vs 0.32). Uma alegação de **alta precisão NÃO se sustenta** (teste pequeno, IC95 largo). O teto
> é dado pelo **confound de um único device**: subir depende de **novas telas limpas pareadas**.

---

## 6. Protocolo de avaliação honesta

`evaluate.py` **não** trata a métrica global como headline; reporta lado a lado:
1. **Primária = sintético livre de confound** + subconjunto **controlado**.
2. **Baselines de confound** sempre (resolução trivial 1.000; padding 1.000; DINO cru 0.72; kNN
   one-class 0.64). O modelo só "vale" se superar o confound no regime controlado/sintético.
3. **Falseabilidade:** o score prediz *resolução* tão bem quanto *erro*? (degenerada no teste real —
   ~0 erros em resolução canônica; o sinal real é o sintético, onde a resolução é constante).
4. **Auditoria same-resolution** (os poucos erros 2076×2152), separando ticket independente de
   quase-duplicatas de sessão.
5. **IC bootstrap agrupado por ticket** em toda métrica; **precision@K**; **MCC, Brier, ECE,
   especificidade, FPR**.

**Trava do teste (anti-snooping):** `siamese.protocol.guard_path` (acionado em
`features.load_embeddings`) levanta `TestSetAccessError` em qualquer leitura de `test*` sem a flag
`--final-test`. Grid/ablação/visualização ficam **fisicamente impedidos** de tocar o teste; a seleção
ranqueia só por **métricas de validação** (`val_synth_gate`). Travado por `tests/test_protocol_guard.py`.

> Métricas de **treino** são **ressubstituição** (in-sample; F1≈1.0 é artefato de protótipos ajustados
> no próprio treino) — diagnóstico de overfitting, **não** resultado.

---

## 7. Técnicas anti-confound

### 7.1 Injeção de erros sintéticos (`synthetic.py`) — a alavanca
Para cada tela limpa 2076×2152, geramos versões com black-region/empty-space/overlay/disorder/cropped
**na mesma resolução/device**. O par (limpa, corrompida) difere **só** pelo conteúdo do erro ⇒ todos os
confounds geométricos são constantes ⇒ o modelo é forçado a aprender o **erro**. Os 4 tipos com
correspondente real (`black_region→black_bars`, `empty_space`, `overlay`, `disorder→disordered_layout`)
são rotulados e alimentam o Estágio 2; `cropped` só o Estágio 1. **Prova** (ablação na val): "só real"
prediz resolução ≈ tão bem quanto erro (aprende o confound); "só sintético" derruba isso e detecta
conteúdo. Produção = **real + sintético**; para um detector **agnóstico de device**,
`train.use_real_errors: false` (`configs/robust_synthonly.yaml`).

### 7.2 Reflow + benign — anti-confound pelo lado limpo (`reflow.py`)
Variantes **LIMPAS** (label 0) de layout legítimo — `scroll_shift`, `two_pane`, `ar_relayout`,
`band_jitter` — injetadas nas limpas de treino (`train_reflow.npz`). São **hard-negatives** que
expandem o cluster limpo e atacam (i) o falso-positivo estrutural (§2) e (ii) o confound: `ar_relayout`
tira a limpa de 2076×2152, então `clean` deixa de ser monorresolução. **Não-colisão** (invariante):
reflow **MOVE/REESCALA**, bug **MATA/TRUNCA** — validado por `scripts/audit_reflow.py` (AUROC(limpo vs
reflow)≈0.5). `benign_augment` (round-trip de resolução + jitter foto-métrico, **on**) remove o atalho
secundário de nitidez (vários erros reais são fotos). É **trade-off**: atenua o rastreamento do
confound ao custo de alguma especificidade.

### 7.3 Calibração livre de confound (`decision.calibrate_on=confound_free`)
A fusão+limiar são fixados em **limpas-val reais (0) + `val_synth` (1) + `val_reflow` (0)** — muitos
negativos limpos + positivos livres de confound → limiar **estável**. Corrige o ponto de operação que,
calibrado em ~26 limpas reais, inundava o teste de falso-positivo. **Reconciliação anti-snooping:** a
regra é sobre o **TESTE**; `val_*` vêm de `processed/VAL` e já sustentam o early-stop — nunca se usa
`test_*`. Reportam-se os dois pontos (`real_val` vs `confound_free`) lado a lado.

---

## 8. Estágio 2 — categoria (limitação estrutural)
Decisor canônico = **k-NN de categoria** (média top-k da similaridade aos erros de treino por classe;
adotado por dar ganho modesto sobre o protótipo — coarse 0.626→0.641, ver
[`COMPARACAO_KNN_TRIPLET.md`](COMPARACAO_KNN_TRIPLET.md)); protótipo e cabeça aux ficam como
**diagnóstico**. Avaliado **condicional ao gate E1** (= produção) e oráculo. **Taxonomia grossa
primária** (2 super-classes, `manifest`): `dead_region` (black_bars+empty_space) · `displaced_content`
(overlay+disordered). A fina de 4 é **secundária/exploratória**.

Há um **teto de separabilidade nas features DINOv2**: um classificador direto (LogReg/kNN) sobre os
embeddings crus das categorias atinge F1-macro baixo; a cabeça aprendida fica no/levemente acima desse
teto. Causas estruturais: (i) categorias **semanticamente próximas** (regiões mortas vs deslocadas);
(ii) rótulo **single-label** para erros que **coocorrem**; (iii) classes com **n pequeno**. O F1
0.35(fino)→0.64(grosso) é em boa parte efeito de **agregar 4→2 classes**, não de melhor modelo — e o
IC95 grosso [0.52–0.75] tem limite inferior perto do acaso (0.5). **Alavancas:** mais dados por
categoria, rótulo **multi-rótulo**. Reportar sempre com o IC.

---

## 9. Localização — onde está o erro (`localize.py`, apoio)
Saída adicional de explicabilidade: mapa de anomalia por patch (37×37) sobreposto na tela.
- **PatchCore (padrão):** banco de memória dos patches **limpos**; cada patch da consulta recebe a
  distância ao patch limpo mais próximo. Tela limpa fica **fria**; conteúdo estranho (fotos, overlays)
  fica quente. **Limitação:** não localiza bem a "faixa preta" (preto também ocorre em limpas).
- **Geométrico (`--geometric`):** CV clássica (`geometric.py`) — localiza barras pretas/vazios com
  precisão, mas **NÃO classifica** (AUROC≈0.50): barra preta grande **não** é sinal de erro (vídeos têm
  letterbox legítimo). É **EVIDÊNCIA/localização**, usada junto com o modelo, nunca como decisor.

> Localizar erro de layout por-patch é **intrinsecamente limitado**: parte do sinal é
> relacional/contextual (alinhamento, sobreposição), não local. O heatmap é **aid de atenção**, não
> classificador.

## 10. O que NÃO fazer (decisões registradas)
- ❌ Input 224/392 no DINOv2 (o checkpoint só aceita 518).
- ⚠️ Padding **só** cinza neutro **e** mascarando os patches de padding — senão vaza aspect-ratio.
  Nunca padding **preto** (imita "black region").
- ❌ Fundir os ramos por `max`/OR (derruba precisão) — usar fusão calibrada.
- ❌ Fine-tune/LoRA do backbone com ~450 imagens.
- ❌ Reportar métrica **global** como headline (é ~98% confound), ou **AP** da sonda como headline
  (acaso 0.80) — liderar com **AUROC livre de confound**.
- ❌ Calibrar limiar/fusão em qualquer derivado do **teste** (`test_synth`/`test_reflow`/`test.npz`);
  calibrar na **validação** livre de confound é legítimo.
- ❌ Prometer "precisão ≥ 0.95" como número único sem IC.

## 11. Limitação central e próximos passos
Com `clean` vindo de **um único device**, **nenhuma arquitetura demonstra "alta precisão independente
de confound"** de forma estatisticamente sólida. Em ordem de impacto:
1. **Coletar telas limpas diversas** (outros devices/resoluções, fotos, landscape/laptop/tent) — a
   única alavanca que torna a métrica global significativa.
2. Rótulo **multi-rótulo** para erros que coocorrem (melhora o Estágio 2).
3. Enriquecer os erros sintéticos com base em telas reais (cobrir fotos/glare).
4. Avaliação por k-fold agrupado + calibração de limiar OOF para reduzir o IC do teste (n pequeno).
