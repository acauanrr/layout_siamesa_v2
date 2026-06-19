# Design — Rede Siamesa para detecção binária de erro de layout em UI

> Backbone: **DINOv2 ViT-S/14** (congelado) · Objetivo: dada uma imagem/print de tela,
> dizer com **alta precisão** se ela **tem erro** de layout ou **não tem**.

Este documento (1) avalia criticamente a descrição de rede siamesa trazida no pedido,
(2) reporta os achados sobre os dados — que mudam tudo —, (3) propõe e justifica o design
melhorado, e (4) mostra os resultados reais medidos neste dataset.

---

## 1. Resumo executivo (leia isto primeiro)

**O dado tem um confound quase perfeito.** As 172 imagens *sem-erro* são **todas** do
mesmo dispositivo e da mesma resolução (**2076×2152**, telas de onboarding limpas de uma
única sessão de captura). As 188 imagens *com-erro* são heterogêneas: 74 resoluções
diferentes, 47 fotos de câmera, vários form factors (fold/unfold/laptop/tent). Medimos:

| Classificador | AUROC (test) | Precisão | Recall |
|---|---|---|---|
| **Regra trivial "resolução ≠ 2076×2152 ⇒ erro"** | **0.982** | **1.000** | 0.964 |
| LogReg só com (resolução, aspecto, é-foto) | 0.911 | — | — |
| LogReg sobre DINOv2 cru (CLS 384-d) | 0.849 | — | — |

Ou seja: **qualquer métrica global "de erro" neste dataset é ~98% trapaça** — basta olhar
a resolução. Um modelo que "acerta 95%" no test global provavelmente está detectando
*dispositivo/resolução*, **não** erro de layout.

**A consequência de design:** o objetivo real ("tem erro de layout?") só pode ser
aprendido e medido de forma honesta se quebrarmos esse confound. Fazemos isso com
**injeção de erros sintéticos nas próprias imagens limpas** (mesma resolução/device),
criando pares onde *só o conteúdo do erro muda*. O modelo treinado assim detecta erro de
**conteúdo** de forma **modesta** (held-out: AUROC **0.70** / AP 0.87 no sintético livre de
confound; ver §5), enquanto um modelo treinado só com erros reais aprende a trapacear pela
resolução (ver §7).

**Recomendação nº 1 (de dados, não de arquitetura):** para um detector que generalize, é
preciso coletar imagens **sem-erro** que cubram a mesma diversidade das com-erro — telas
limpas de **outros devices, outras resoluções, fotos de telas sem erro, landscape, laptop,
tent**. Nenhuma arquitetura supera essa lacuna de dados.

---

## 2. Avaliação crítica da descrição de rede siamesa do pedido

A descrição trazida é o **pipeline siamês clássico genérico**, e tem dois problemas para
**este** problema:

**2.1. É orientada a texto/grafos, não a imagem.** Cita SBERT, Transformers de texto,
GATv2 (redes em grafos), "log de execução bem-sucedida", "código estruturalmente
correto". Nada disso se aplica a uma imagem única de UI. Aqui o sinal é **visual** e o
extrator correto é o **DINOv2** (já decidido).

**2.2. (o ponto central) A formulação "parear o alvo `x₁` com uma referência-de-sucesso
`x₂` e perguntar se diferem" é mal-posta aqui.** Ela pressupõe que existe **uma**
referência boa canônica. Mas a classe *sem-erro* é **visualmente diversa**: telas de
onboarding de apps diferentes, com cores, ilustrações e idiomas distintos. Duas telas
**limpas** de apps diferentes são legitimamente **dissimilares** — mesmo ambas corretas.
Treinar "alvo vs uma referência boa" ensinaria o modelo a confundir **"tela diferente"**
com **"tela errada"** ⇒ falso-positivo estrutural em qualquer app/tela novos.

**Veredito:** a *estrutura* siamesa (ramos de pesos compartilhados + comparação no espaço
de embeddings) é mantida e é correta. O que muda é **o referente da comparação**: em vez
de comparar contra *uma* imagem de referência, comparamos contra **múltiplos protótipos do
manifold "sem-erro"** (a ideia de clustering levantada no pedido, que está **certa** e foi
promovida a regra de decisão). Tecnicamente isto é uma **rede siamesa one-class / de
metric learning** — satisfaz o pedido literal ("rede siamesa, binária, alta precisão") sem
cair na armadilha da referência única.

As partes da descrição que **aproveitamos**: vetor de fusão `[z₁, z₂, |z₁−z₂|, z₁⊙z₂]`
(implementado em `SiamesePairHead`), cabeçalho sigmoid + BCE, e a alternativa de
**Contrastive Loss** (implementada em `losses.py`). O treino principal usa **Supervised
Contrastive** (mais estável com poucos dados e sem exigir referência única).

---

## 3. Achados sobre os dados (auditoria de confounds)

`python scripts/build_splits.py` imprime a auditoria completa. Resumo (541 imagens; split
agrupado por **ticket de erro** + **sessão/near-dup das telas limpas**, com **0 vazamento de
grupo** — ver §10.4 sobre a correção do vazamento de quase-duplicatas limpas):

| Atributo | sem-erro | com-erro | Confound? |
|---|---|---|---|
| Resolução 2076×2152 | 172 / 172 | 8 / 188 | **Crítico** — separa quase perfeito |
| Foto (câmera) | 0 | 47 | **Forte** — foto ⇒ sempre erro |
| Form factor no nome | 0 (todas "Screenshot_") | fold/unfold/laptop/tent | **Forte** |
| Orientação landscape | 0 | 30 | Forte |
| Caixa vermelha desenhada (`_boundBox`) | 0 | 2 | Anotação espúria |
| `_competitor` (UI de concorrente) | 0 | 16 | Possível ruído de rótulo |

Pontos sutis e importantes:
- As **8** imagens com-erro que **são** 2076×2152 são o único sinal real anti-confound.
  Dessas, **7 são `Screenshot_*` da mesma sessão** das sem-erro (quase-duplicatas de
  domínio) e **apenas 1 é um ticket independente** (`IKSWW-130772`). O poder estatístico
  para "alta precisão independente de confound" é, honestamente, **baixíssimo**.
- O split é **agrupado por ticket** (`IKSWW[-_]\d+`): imagens do mesmo bug nunca cruzam
  train/test. Verificado: 0 grupos cruzando splits.

---

## 4. Design proposto (o que implementamos)

```
imagem ─► [resize anamórfico 518×518] ─► DINOv2 ViT-S/14 (CONGELADO)
                                              │  CLS 384  +  mean/std dos patch tokens 37×37
                                              ▼  = vetor base 1152-d  (cacheado em disco)
                              ┌──────────  cabeça de PROJEÇÃO g(·)  (pesos compartilhados)
                              │            LayerNorm→Linear(1152,256)→GELU→Drop→Linear(256,128)→L2-norm
                              ▼
                          z ∈ S¹²⁷  ──►  (A) score de protótipo: 1 − cos(z, protótipo-limpo-mais-próximo)
                              │          (B) cabeçalho auxiliar: Linear(128→1) = logit de erro
                              ▼
                       FUSÃO calibrada (LogReg de 2 features) ─► p(erro) ─► limiar p/ precisão-alvo
```

**Por que cada escolha:**

- **DINOv2 congelado.** Com ~360 imagens, fazer fine-tune dos 22M de parâmetros
  overfittaria os confounds imediatamente. Toda a capacidade treinável fica na cabeça
  (~330k params). Embeddings são fixos ⇒ **cacheados** (`artifacts/embeddings/*.npz`) e o
  treino roda em segundos.
- **Pré-processamento 518×518 — dois modos (`backbone.preprocess`), `pad` é o padrão.**
  518 é a **única** resolução aceita pelo checkpoint (1369 patch tokens = 37×37); nunca
  usar center-crop (descartaria topo/laterais, onde moram black-region/cropped).
  - `resize`: resize **anamórfico** direto (espreme o aspecto; não injeta bordas).
  - `pad` (**padrão, validado empiricamente**): padding até quadrado **preservando o
    aspecto**, preenchido com o **cinza neutro = média do ImageNet** (vira ~0 após a
    normalização → influência mínima; é distinto de preto e do fundo das telas), e só então
    resize. **Preserva a geometria real do erro** (uma faixa preta não é espremida).
  - O risco do `pad` — a *área* de cinza correlacionar com o aspect-ratio (logo com form
    factor) — é neutralizado calculando o `mean/std` dos patch tokens **apenas na região de
    conteúdo** (`content_patch_mask`, ver `geometry.py`).
  - **Medição (PRÉ-correção, no test):** `pad` ≥ `resize` em todas as métricas honestas —
    subconjunto controlado **0.865 → 0.910**, recall, detecção sintética 0.881 → 0.882.
    (Valores absolutos predam a correção de vazamento; `compare_preprocess.py` agora roda em
    DEV/val. O padrão qualitativo — `pad` ajuda — se mantém. Held-out honesto na §5.)
- **CLS + estatísticas de patch (1152-d).** black-region/empty-space são anomalias de
  **homogeneidade espacial**: o desvio-padrão dos patch tokens cai em regiões grandes
  uniformes. O ganho de adicionar patch stats (**AUROC 0.71 → 0.88**, pré-correção) motivou o
  `use_patch_stats: true` (padrão); re-medir na val após a Fase 0.
- **Cabeça de projeção compartilhada = a parte "siamesa".** A mesma `g` é aplicada a
  qualquer imagem; comparar duas imagens = comparar `z₁, z₂`. Treinada com **Supervised
  Contrastive** (`supcon_loss`) para que limpo forme cluster compacto e erro caia fora.
- **Cabeçalho auxiliar (B).** Detector binário direto que **não** depende do banco de
  referências — evita o detector-de-novidade puro (que dispararia em qualquer app novo).
- **Injeção de erros sintéticos** (`synthetic.py`) — **a alavanca anti-confound**. Para
  cada imagem limpa 2076×2152 geramos versões com black-region/empty-space/overlay/
  disorder/cropped, **na mesma resolução/device**. O par (limpa, corrompida) difere **só**
  pelo conteúdo do erro ⇒ todos os confounds geométricos são constantes ⇒ o modelo é
  forçado a aprender o **erro**, não o dispositivo. (Preview: `artifacts/reports/synthetic_preview.png`.)

### 4.1. Regra de decisão = a ideia de clustering do pedido (corrigida)

- **k protótipos** do cluster *limpo* via k-means sobre os `z` das imagens sem-erro de
  treino (`fit_prototypes`; `k=1` basta aqui, pois o limpo é unimodal; `k>1` para o caso
  multimodal). `score_proto(x) = 1 − cos(z_x, protótipo mais próximo)`.
- **Fusão calibrada** de `[score_proto, aux_logit]` por uma LogReg ajustada na validação
  (**não** `max`/OR, que maximiza recall e derruba precisão).
- **Limiar** fixado na **validação real** para a **precisão-alvo** (0.90/0.95/0.99),
  reportando o recall honesto resultante.

---

## 5. Resultados reais (produção: patch-stats, `pad`, real+sintético)

**Avaliação HELD-OUT honesta** (config `proj_dim=128` **congelada** após seleção íntegra na val +
estabilidade multi-seed/1-SE; teste = **130 imagens** = 41 limpas + 89 erros; processado **uma única
vez** via `scripts/evaluate.py --final-test`; jun/2026). Os números legados aqui (F1 0.86, P@5 1.0,
alta-precisão 1.0@recall 0.5) foram **REVOGADOS** — mediam split com vazamento + config escolhida por
snooping no teste (§10.4). O held-out real é modesto:

### Estágio 1 — detecção vs. baselines de confound

| Avaliação (TEST) | Modelo | Baseline de confound | |
|---|---|---|---|
| Global AUROC | 0.725 (IC95 0.67–0.84) | **resolução trivial 0.994** · padding 0.972 · DINO-cru 0.746 | ❌ não supera |
| Falseabilidade | prediz ERRO 0.725 | prediz RESOLUÇÃO 0.721 | ❌ rastreia resolução |
| **Controlado** (n=71, 30 err) | **0.671** (IC95 **0.576**–0.835) | confound 0.383 | ✅ supera |
| **Sintético livre de confound** (41 vs 164) | **0.695** · AP 0.868 | — | ✅ sinal real, modesto |

Ponto de operação (limiar de F1 fixado na val): acc 0.70 · F1 0.815 · **bAcc 0.544 · MCC 0.171 ·
especificidade 0.12 · FPR 0.88** (TP86/TN5/FP36/FN3; acc IC95 0.50–0.93). O limiar, calibrado em
**26 limpas** de val, é **instável** — inunda o teste de falso-positivo. Limiar por precisão-alvo
(val→test): alvo 0.90 → precisão 0.805 / recall 0.371 (fp=8); alvo 0.95/0.99 → 0.781 / 0.281 (fp=7).
**precision@K**: P@5 0.6 · P@10 0.6 · P@20 0.75.

### Estágio 2 — categoria (n=89 erros)
F1-macro **0.388** (protótipo) / 0.379 (aux head), dominado por ruído de amostra minúscula
(`distortion` 0.80 em n=3; `orientation` 0.00 em n=2; `black_bars` 0.52; `overlay` 0.39).

> **Veredito honesto:** globalmente o modelo **NÃO supera** o confound de resolução (0.73 vs 0.99) e
> a falseabilidade confirma que rastreia resolução ≈ tão bem quanto erro. Existe sinal de layout
> **real porém modesto** no regime **controlado** (0.67, IC exclui 0.5) e **sintético** (0.70). Uma
> alegação de **alta precisão NÃO se sustenta** (fp de um dígito, IC largo — critério #7 da
> auditoria). O teto é dado pelo **confound de um único device**: subir depende de **novas telas
> limpas pareadas** (Fase 1), não de mais tuning.

---

## 6. Protocolo de avaliação honesta (por que confiar nos números)

`evaluate.py` reporta, e **não** trata a métrica global como headline:

1. **Métrica primária = subconjunto controlado** (unfold-portrait-screenshot) + **detecção
   sintética** livre de confound.
2. **Baselines de confound** sempre lado a lado (held-out: resolução trivial **0.994**; padding
   0.972; DINOv2 cru 0.746; kNN one-class 0.675). O modelo só "vale" se superar o confound no
   regime controlado/sintético — e **globalmente não supera** (ver §5).
3. **Testes de falseabilidade:** (a) o score do modelo prediz a *resolução* tão bem quanto
   o *erro*? (b) embaralhar rótulos dentro do estrato.
4. **Auditoria same-resolution** (os 8 erros 2076×2152), separando o ticket independente
   das quase-duplicatas de sessão.
5. **IC bootstrap agrupado por ticket** em toda métrica; **precision@K**.

---

## 7. Ablação — a prova de que sintético quebra o confound

> **⚠️ Números absolutos abaixo são PRÉ-correção (split com vazamento + medidos no test).** O
> `ablation.py` agora roda em modo **DEV (val)** e o teste é trancado; o held-out honesto está na
> §5. A ablação continua válida **qualitativamente** (o padrão importa, não os valores): "só real"
> prediz resolução ≈ tão bem quanto erro (aprende o confound); "só sintético" derruba isso. Re-rode
> `python scripts/ablation.py` p/ os números na val.

| Treino (pré-correção) | synt AUROC | synt AP | global AUROC | controlado AUROC | →prediz **resolução** | →prediz **erro** |
|---|---|---|---|---|---|---|
| **real + sintético** | 0.881 | 0.966 | 0.890 | 0.865 | 0.912 | 0.890 |
| **só sintético** | 0.861 | 0.961 | 0.668 | 0.679 | **0.657** | 0.668 |
| **só real** | 0.752 | 0.924 | 0.933 | 0.955 | **0.919** | 0.933 |

Leitura:
- **só real** parece ótimo no global (0.933), mas prediz **resolução** (0.919) tão bem
  quanto erro ⇒ **aprendeu o confound**; e tem a **pior** detecção de conteúdo (0.752).
- **só sintético** detecta conteúdo (0.861) e **não** rastreia resolução (0.657) — é o
  detector genuíno; cai no global (0.668) porque se recusa a trapacear e porque não modela
  tipos que o sintético não cobre (fotos, glare).
- **real + sintético** equilibra (melhor sintético + global decente), mas ainda herda algum
  rastreamento de resolução dos erros reais.

Escolha de produção: **real + sintético** (mais forte e com P@K=1.0). Para um detector
**robusto a confound / agnóstico de device**, use **`só sintético`** (treine com
`train.use_real_errors: false`).

---

## 7b. Mapa de calor — onde está o erro (`localize.py`)

Saída adicional de explicabilidade: além do `p(erro)` por imagem, um **mapa de anomalia
por patch** (37×37) sobreposto na tela. `python scripts/localize.py`.

- **PatchCore (padrão)** — banco de memória dos patch tokens das telas **limpas**; cada
  patch da consulta recebe a distância ao patch limpo mais próximo. Escala **absoluta
  calibrada** (distância "normal" entre patches limpos) → tela limpa fica **fria** (score
  0.19) e regiões incomuns ficam quentes. Bom para **conteúdo estranho** (fotos, overlays,
  arte fora do padrão). **Limitação honesta:** *não* localiza bem a "faixa preta", porque
  preto também ocorre em telas limpas (não é "novidade").
- **Geométrico (`--geometric`)** — visão computacional clássica (`geometric.py`): detecta
  regiões grandes lisas (baixa variância local) e quase-pretas (luminância ~0), com critério
  de **banda de borda** para excluir o fundo da tela e barras de **sistema** finas. Roda na
  imagem **original** (vê a barra real, não o padding). **Localiza a faixa preta com
  precisão** (onde o PatchCore falha — ver overlay de IKSWW-105455). **Mas mediu-se que NÃO
  classifica: AUROC ≈ 0.50.** Motivo (achado honesto e instrutivo): *barra preta grande não
  é sinal de erro* — players de vídeo têm **letterbox preto legítimo** (ex.: a tela LIMPA
  `Screenshot_20260613_103606.png` tem barras laterais idênticas às do erro 105455), e telas
  de onboarding limpas são legitimamente vazias. O erro é **semântico/contextual**, não o
  pixel preto em si. → ferramenta de **EVIDÊNCIA/localização**, usada **junto** com o modelo
  (o modelo decide "é erro"; o geométrico mostra "aqui estão as barras"), nunca como decisor.
- **Supervisionado por sintéticos (`--supervised`, experimental)** — classificador por-patch
  treinado nos erros injetados. **Ruidoso** e dispara em conteúdo normal, porque os erros
  "overlay"/"disorder" produzem patches que *parecem normais* (conteúdo real deslocado) →
  fronteira de decisão ambígua. Mantido como opção, não recomendado.

> Realidade honesta: localizar erro de layout por-patch é **intrinsecamente limitado** aqui
> — parte do sinal é **relacional/contextual** (alinhamento, sobreposição), não local; e até
> o mais "visual" dos erros (faixa preta) é **ambíguo** (idêntico a letterbox de vídeo). O
> heatmap é um **aid de atenção/evidência**, não um segmentador/classificador de defeito.

## 8. O que **não** fazer (decisões registradas)

- ❌ Não usar input 224/392 no DINOv2 (o checkpoint só aceita 518).
- ⚠️ Padding **só** com cinza neutro (média ImageNet) **e** mascarando os patches de padding
  nas estatísticas — senão a área de cinza vaza o aspect-ratio (testado: com máscara, `pad`
  supera `resize`; sem máscara, reintroduz confound). Nunca padding **preto** (imita o erro
  "black region").
- ❌ Não calibrar o limiar em dados sintéticos (data-snooping); limiar na validação real.
- ❌ Não fundir os ramos por `max`/OR (derruba precisão); usar fusão calibrada.
- ❌ Não fazer fine-tune/LoRA do backbone com ~360 imagens.
- ❌ Não reportar métrica **global** como headline (é ~98% confound).
- ❌ Não usar clustering **não-supervisionado** como decisor (alinha a app/form-factor).
- ❌ Não prometer "precisão ≥ 0.95" como número único sem IC.

## 9. Próximos passos (em ordem de impacto)

1. **Coletar telas limpas diversas** (outros devices/resoluções, fotos sem erro, landscape/
   laptop/tent). É a única alavanca que torna a métrica global significativa.
2. Rotular os 16 `_competitor` (são UI limpa de concorrente?) e tratar ruído de rótulo.
3. Inpaint das 2 imagens `_boundBox` (caixa vermelha) antes da extração.
4. Enriquecer os erros sintéticos com base em telas reais (cobrir fotos/glare).
5. Avaliação por k-fold agrupado (CV) + calibração de limiar OOF para reduzir o IC do test (130
   imagens; o ponto de operação calibrado em 26 limpas de val é instável — ver §5).

---

## 10. Extensão multi-cluster, reorganização de dados e limpeza (jun/2026)

Quatro mudanças foram entregues sobre o detector binário descrito acima.

### 10.1 Reorganização de dados (C)
A fonte de erros migrou da pasta flat `with_errors/` (binária) para
**`errors_dataset/<categoria>/`**, com 6 categorias reais (`black bars`, `disordered
layout`, `distortion`, `empty space`, `orientation`, `overlay`). `manifest.scan_dataset`
ganhou um parâmetro **`source`** (exclusivo: `errors_dataset` padrão | `with_errors`
legado — nunca os dois, pois ≥40 nomes coincidem), um **mapa pasta→slug** canônico e uma
coluna **`category`** que flui do CSV → `.npz` (`features.py`) → treino/decisão. O split
continua **agrupado por ticket** (0 vazamento; só 3 tickets cruzam categorias) mas passou
a ser **estratificado por categoria**, garantindo que toda classe — inclusive as raras
(`orientation`=7, `distortion`=13) — apareça em train/val/test. Fracoes `val/test = 0.15/0.24`
(**test ampliado** p/ >=40 telas limpas de referencia — estimativa robusta de falso-alarme:
**41 limpas / 90 erros / 131 no test**; train 328, val 82; seed 42). O caminho binário legado é
100% reprodutível (`--source with_errors`, `train.multiclass: false`).

### 10.2 Limpeza de marcações vermelhas (D)
`scripts/audit_red_marks.py` detecta (numpy+PIL+scipy) marcações humanas vermelhas
(retângulos/círculos/setas) por máscara de vermelho forte + componentes conexos + heurística
de forma (contorno oco / traço fino). Das 407 imagens, 59 foram sinalizadas; a decisão de
exclusão foi tomada por **inspeção visual** (folhas de contato rotuladas) para evitar apagar
conteúdo vermelho legítimo (apps de compras, vídeos, banners). **35 imagens** foram excluídas
(30 com `_boundbox` no nome — anotação explícita, em vermelho OU azul — + 5 formas desenhadas
confirmadas); os ~24 falsos-positivos da heurística (preço riscado, jaqueta, vestido) foram
**preservados** e listados em `artifacts/reports/red_marks_review.csv`. Manifesto do que foi
apagado: `red_marks_deleted.csv`. `errors_dataset/`: 407 → **372**.

**Deduplicação por conteúdo (hash):** auditoria por md5 achou **3 imagens de conteúdo idêntico**
— `IKSWW-173861` (2 imagens de "black region" catalogadas TAMBÉM em `overlay`, uma com nome
trocado) e uma cópia literal `IKSWW-93466_..._(1)`. Removidas (mantendo a cópia na categoria
correta, confirmada visualmente): `errors_dataset/`: 372 → **369**; dataset real **= 541 únicas**,
**0 duplicatas**. **0 vazamento de grupo entre splits** (verificado) — porém o teste "ticket não
cruza split" **não detectava** o vazamento de **quase-duplicatas das telas limpas** (capturas
sequenciais da mesma sessão/dispositivo, similaridade DINO ≈ 0.99). Corrigido na **Fase 0**:
limpas reagrupadas por **sessão (timestamp) + near-duplicate perceptual (dHash)** antes do split
(172 arquivos → 15 grupos atômicos), travado por `tests/test_split_isolation.py`. Ver §10.4.

### 10.3 Multi-cluster em DOIS ESTÁGIOS (B)
A "ida além do binário" foi feita preservando o gate de alta precisão:

- **Estágio 1 — gate "tem erro?"**: inalterado conceitualmente (distância ao protótipo
  limpo + cabeça auxiliar + fusão calibrada + limiar). No multi-classe, `p(erro)=1−softmax[clean]`.
- **Estágio 2 — categoria**: a cabeça `g()` é treinada com **SupCon multi-classe** (a
  `supcon_loss` já agrupa por igualdade de rótulo — sem mudança); a cabeça auxiliar virou
  `Linear(proj_dim, N+1)` + cross-entropy; os **batches são balanceados por classe**
  (≥2/classe, oversample das raras). A decisão de categoria usa **protótipos por categoria**
  (`fit_category_prototypes`, k-means dentro de cada classe) + `argmax` de similaridade.
  Os erros **sintéticos** alimentam os dois estágios: anti-confound (todos) no Estágio 1 e,
  rotulados, os 4 tipos com correspondente real (`black_region→black_bars`,
  `empty_space`, `overlay`, `disorder→disordered_layout`) no Estágio 2 (`cropped` não tem
  classe real → só Estágio 1; `distortion`/`orientation` não têm gerador).

### 10.4 Grid search e o "problema do limiar" (A)
`scripts/grid_search.py` varre o produto cartesiano de uma grade (chaves dotted), isola
artefatos por ponto e **re-extrai embeddings só quando o eixo toca `backbone.*`/`synthetic.*`**.
**❌ Afirmação anterior estava ERRADA (corrigido na Fase 0 — auditoria jun/2026).** O texto
aqui dizia que a seleção de combos era "livre de data-snooping" porque usava o **AUROC sintético
do gate** (`sintetico_livre_de_confound`). Mas essa métrica era calculada em `evaluate.py`
**a partir de `test_synth.npz` + as imagens limpas de TESTE** — ou seja, **a seleção enxergava o
teste** (snooping real, problema #2 da auditoria). A "melhor config" antes anunciada aqui
(`temperature=0.05, aux_weight=0.6, proj_dim=256, k_prototypes=3`, synth-AUROC 0.81, F1 0.85…)
fica **REVOGADA** — foi escolhida sobre derivados do teste.

**Protocolo corrigido:**
- `scripts/grid_search.py` **não importa `evaluate` nem lê nenhum `test*`**: ranqueia
  **exclusivamente por métricas de VALIDAÇÃO** devolvidas por `train_head()` — por padrão
  `val_synth_gate` (gate sintético livre de confound medido na **val**: protótipos vêm do
  *train*, sonda = limpas de val + `val_synth.npz`). Independente de limiar e de confound.
- O **TESTE é trancado programaticamente** (`siamese.protocol.guard_path`, acionado em
  `siamese.features.load_embeddings`): qualquer leitura de `test.npz`/`test_synth.npz` sem a
  trava liberada levanta `TestSetAccessError`. Grid/ablação/visualização ficam **fisicamente
  impedidos** de tocar o teste.
- O melhor combo, **depois de congelado**, é avaliado no TESTE **uma única vez** via
  `python scripts/evaluate.py --config <vencedora> --final-test` (único ponto que chama
  `allow_test_access()`). Em modo DEV (sem a flag) o `evaluate.py` reporta sobre a **val**.
- Testes de integridade em `tests/test_protocol_guard.py` (grid shield) e
  `tests/test_split_isolation.py` travam essas invariantes.

> Métricas de TREINO continuam reportadas como **ressubstituição** (in-sample, p/ diagnóstico de
> overfitting — F1 de treino ≈ 1.0 é artefato de protótipos ajustados no próprio treino, **não
> reportar como resultado**). Os números **vinculantes** só saem do `--final-test`, uma vez,
> após seleção íntegra — e uma alegação de alta precisão exige o **limite inferior do IC95%**
> alcançando a meta.

**Estabilizacao da selecao (IMPLEMENTADO):** o early-stop do `train.py` agora usa o **sintetico de
VALIDACAO** (`val_synth.npz` — erros injetados nas limpas de val, mesma resolucao -> livre de
confound), combinado com o macro-F1 de categoria, em vez do gate confundido por resolucao. Isso
elevou o synth-AUROC do gate de **0.70 -> 0.81** e tornou o ranking do grid muito mais **estavel**
(os melhores combos agora ficam agrupados em 0.78-0.81, sem os colapsos para ~0.50 de antes), sem
data-snooping no TEST. `epochs=500`, `patience=80` (teto alto; o early-stop decide a parada).
Ainda assim, com base pequena/confundida, **re-rode o grid ao mudar os dados**.

### 10.5 Limitação central do Estágio 2 (achado honesto)
A clusterização por categoria tem um **teto de separabilidade nas próprias features DINOv2**:
um classificador direto (LogReg / kNN-5) sobre os embeddings crus das 6 categorias atinge só
**F1-macro ≈ 0.10–0.21** (kNN-5: acc 0.375). Nossa cabeça (F1-macro 0.24) já está **no/levemente
acima** desse teto. As causas são estruturais, não de implementação: (i) categorias
**semanticamente sobrepostas** (black bars vs empty space vs cropped são todas "regiões";
overlay vs disordered são "elementos deslocados"); (ii) **rótulo single-label** para erros que
**coocorrem** na mesma tela; (iii) classes **raras** (`orientation`=1, `distortion`=2 no test)
sem poder estatístico. O Estágio 2 é forte nas categorias visuais frequentes (`black_bars` F1
0.67, `overlay` 0.44, `empty_space` 0.35) e satura nas demais. **Alavancas** (análogas à §1):
mais dados por categoria rara, rótulo **multi-rótulo**, ou taxonomia mais grossa (ex.: agrupar
distortion+disordered). A explicabilidade (heatmaps) segue **adiada** (§7b), conforme alinhado.

### 10.6 Fluxo de dados: `data/processed/` como FONTE DA VERDADE

Para garantir que **o modelo e as outras equipes usem exatamente o mesmo dataset**, o fluxo foi
invertido: o modelo **não** lê mais de `data/input/`; ele lê do dataset **materializado e
categorizado em `data/processed/`** (o mesmo que é compartilhado).

```
data/input/  (entrada bruta; novas imagens chegam aqui)
  └─ build_splits.py      → data/splits/*.csv      (atribui split: agrupado+estratificado)
  └─ export_processed.py  → data/processed/        (MATERIALIZA reais por categoria + sintéticos + manifest/card)
data/processed/  (FONTE DA VERDADE — o que o modelo usa e o que se compartilha)
  └─ extract_features.py  → artifacts/embeddings/  (VARRE a arvore processed/; paths apontam p/ processed/)
  └─ make_synthetic.py    → val/test_synth.npz     (sonda livre de confound, de processed/{val,test}/real/clean)
  └─ train.py / evaluate.py / visualize.py
```

- `extract_features.py` **varre a árvore** `processed/<split>/<fonte>/<categoria>/` (estrutura =
  verdade; metadados de confound re-derivados do nome do arquivo via `manifest._parse_meta`).
  Logo, **correções manuais** em `processed/` (mover de categoria, remover, editar pixels) são
  honradas sem precisar de manifesto — basta re-rodar `extract_features` → `train` → `evaluate`.
- **Não** re-rode `export_processed` após correções manuais: ele reconstrói `processed/` a partir
  de `input/` (com `_reset_dir`) e sobrescreveria os ajustes. `export_processed` é só para
  **ingestão** (novas imagens em `input/`).
- O `train_synth.npz` vem de `extract_features` embedando `processed/train/synthetic/` (FONTE
  única); `make_synthetic` cobre apenas a sonda val/test. Verificado: após o refactor, **100% dos
  embeddings apontam para `data/processed/`** e as métricas ficam idênticas (mesmo dado, nova fonte).
- O caminho **legado** (`extract_split` lendo `splits/*.csv`→`input/`) permanece em `features.py`
  para reprodutibilidade binária, mas não é o fluxo padrão.
