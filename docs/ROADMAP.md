# Roadmap de Ação — `layout_siamesa_v2`

> Plano de implementação para fechar a disparidade treino→teste, quebrar o confound de
> resolução na origem (dados) e deixar o projeto limpo e objetivo. Escrito após auditoria
> profunda do pipeline (jun/2026). Cada item é executável e tem critério de aceite.
>
> **Tese central, agora provada com números:** o gargalo **não é o modelo nem a regra de
> decisão** — é a **pobreza do manifold limpo**. A correção primária é **dados** (telas
> limpas diversas, multi-resolução, multi-form-factor). Métodos e limpeza são alavancas
> secundárias.

---

## 0. Diagnóstico (o que a investigação encontrou)

### 0.1 Os números (baseline congelado, `data/processed_v3`)

| Métrica | TREINO (resub.) | TESTE (held-out) | Gap |
|---|---|---|---|
| Gate AUROC | 0.991 | **0.596 / 0.607** | **−0.40** |
| Gate acurácia | 0.875 | **0.583** | −0.29 |
| Gate bAcc / precisão / recall / especif. | 0.897 / 0.993 / 0.804 / 0.990 | 0.584 / 0.696 / 0.582 / 0.585 | −0.31 / −0.30 / −0.22 / −0.41 |
| 5-classes fim-a-fim (acc / F1-macro) | 0.875 / 0.882 | **0.380 / 0.286** | −0.50 / −0.60 |
| **AUROC livre-de-confound** (teto honesto) | — | **0.721** (AP 0.903) | — |
| Estágio 2 grossa (2 super-cls): acc / F1 | — | 0.642 / 0.641 | — |
| Estágio 2 fina (4 cls, oráculo): acc / F1 | — | 0.433 / 0.354 | — |

Teste = 108 imgs (41 limpas + 67 erros). Treino = resubstituição (in-sample, n=273).
Regra canônica: gate `prototype` + Estágio 2 `knn`.

### 0.2 A causa-raiz, quantificada

| | LIMPAS (real) | ERROS (real) |
|---|---|---|
| imagens / grupos | 172 / **16 sessões** | 277 / 217 tickets |
| campanha de captura | **1 dia** (`Screenshot_20260613_*`) | múltiplas |
| resoluções distintas | **1** (2076×2152, AR 0.96) | 108 |
| form factors | **só `unknown`** | unfold·fold·laptop·tent·desktop |
| orientações | **só `unknown`** | portrait·landscape |

- **Toda** tela limpa real é 2076×2152 (um device, near-square → provável **dobrável
  desdobrado**). Só 4 erros compartilham essa resolução → a regra trivial
  "resolução ≠ 2076×2152 ⇒ erro" dá **AUROC 1.000** (teto-de-trapaça, *diagnóstico*, não o
  modelo).
- O conceito de "limpo" do modelo é construído de **~16 sessões quase idênticas de um único
  dia/device**, enquanto "erro" cobre **6 form factors e 217 tickets**. Essa assimetria **é**
  o gap 0.90→0.57: o modelo memoriza o cluster limpo estreito (treino) e **toda tela de app
  nova cai fora dele** (teste) → falso-positivo → acc 0.58.

### 0.3 Dois problemas distintos, **mesma alavanca**

1. **Validade da avaliação** — no teste real, resolução separa limpo/erro quase perfeitamente;
   qualquer "acurácia no teste real" é confundida. Métrica honesta hoje = AUROC
   livre-de-confound (0.72) + subconjunto controlado (0.60 vs 0.32).
2. **Generalização** — o manifold limpo é pequeno/homogêneo demais para generalizar.

Ambos se resolvem com **telas limpas reais diversas** (multi-resolução, multi-form-factor,
muitos apps). Propriedade-chave do pipeline que torna isso barato:
**`extract_synthetic`/`extract_reflow_clean` geram a partir de qualquer `clean_rows`** — então
**cada limpa nova vira automaticamente** (a) um negativo real, (b) 4 erros sintéticos *casados
na mesma resolução/domínio* e (c) 4 negativos de reflow em resoluções variadas. Diversificar o
limpo diversifica os dois lados **sem criar novo confound**.

### 0.4 O que **não** é o problema (já investigado/descartado)

- **Regra de decisão:** k-NN vs protótipo já comparado — k-NN não ajuda o gate, ganho modesto
  no Estágio 2 (já adotado). Não é a alavanca (`docs/COMPARACAO_KNN_TRIPLET.md`).
- **Perda:** SupCon > Triplet (Triplet colapsa). Não trocar.
- **Vazamento:** o teste é fisicamente trancado no chokepoint `features.load_embeddings`
  (`protocol.guard_path`); calibração sempre na val. Sem vazamento clássico.
- **Conclusão:** o modelo está **perto do teto dos dados atuais**. Subir o teto = dados.

> ⚠️ Nuance de honestidade: as provas de "não-trapaça" são mais fracas do que relatado em
> alguns docs — o teste de falseabilidade global é degenerado (gap 0.000) e o `label_shuffle`
> deu 0.718 (esperado ~0.5). A evidência **honesta** é a sonda sintética (0.72) + subconjunto
> controlado (0.60 vs 0.32) + correlações (corr(score,padding)=0.05). Liderar por essas.

---

## 1. Princípios de execução

1. **Dados primeiro.** É a única alavanca que sobe o teto. Tudo mais é secundário.
2. **Selecionar sempre na val, nunca no teste.** O protocolo held-out é sagrado.
3. **Falhar alto, nunca silencioso.** Fallbacks silenciosos hoje podem trocar o regime que
   gerou os números do headline (ver Fase 1.2).
4. **Aditivo e reversível.** Cada fase deixa o baseline reproduzível; tag antes de mexer.
5. **Métrica-guia = AUROC livre-de-confound** estável em multi-seed/CV agrupado.

---

## FASE 1 — Higiene + baseline congelado (rápido, baixo risco)

Objetivo: travar o baseline atual, eliminar fallbacks silenciosos e inconsistências, deixar o
repo "clean e objetivo" **antes** de qualquer mudança de dados/método.

### 1.1 Congelar baseline
- [ ] `git tag baseline-v3-knn e1a34dc` (árvore já limpa) + exportar
  `evaluation_report.json` atual para `docs/baselines/`.
- [ ] Registrar números da §0.1 como "antes" (já estão neste doc).

### 1.2 Eliminar fallbacks silenciosos (correção de robustez — eval agent)
- [ ] `evaluate.py:216` — troca silenciosa de calibração `confound_free → real_val` quando
  falta `val_synth.npz`. **Fazer falhar alto** (ou WARN no topo do relatório), pois muda o
  ponto de operação do headline (regime FPR 0.88).
- [ ] `train.py:289-295` — early-stop cai silenciosamente para `val_ap` (confundido) se
  `val_synth_gate` for NaN; o aviso some porque `run_experiment._run` só ecoa as últimas 6
  linhas (`run_experiment.py:59`). **Propagar o aviso ao relatório final.**
- [ ] `report_processed_v3.py` — `fused()` **hardcoda gate=prototype** (`:312-315`) e
  `multiclass=True` (`:315`); diverge da produção se `gate_method=knn` ou modelo binário.
  **Ler `gate_method`/`multiclass` do bundle** e reconstruir o gate igual ao `evaluate.py`.
- [ ] `evaluate.py:60-68` (`_resolutions`) — falha de leitura PIL vira `(-1,-1)` →
  silenciosamente tratada como resolução não-canônica, corrompendo baselines de confound.
  **Contar e reportar falhas; abortar se houver.**
- [ ] `evaluate.py:285-291` — máscara do subconjunto controlado (`unfold & portrait &
  screenshot`) pode esvaziar silenciosamente e virar veredito "não bate o confound".
  **Logar tamanho da máscara; avisar se < N mínimo.** (Hoje as limpas têm form_factor
  `unknown` → a máscara já é frágil; melhora com a Fase 2.)

### 1.3 Corrigir inconsistências de taxonomia/caminho (tech-debt agent)
- [ ] "6 classes"→"4" e "3 super-classes"→"2 (+clean)" em: `config.py:115,117`,
  `evaluate.py:532,533,613`, `scripts/stage2_baseline.py:7`, `scripts/visualize.py:40`,
  `scripts/export_processed.py:9,223`. Reconciliar `EXPERIMENT_RESULTS.md:63` (2) vs
  `artifacts/indt/.../EXPERIMENT_RESULTS.md:63` (3) regenerando ambos.
- [ ] Comentários "FONTE DA VERDADE = data/processed/" → `data/processed_v3` em
  `features.py:3,17`, `synth_features.py:68`, `make_synthetic.py:3,5`,
  `extract_features.py:2,4,14`, `run_experiment.py:4,12`, `grid_search.py:71`.
- [ ] `train.py:99,181` + `default.yaml:75` — atualizar justificativa de
  `max_oversample_per_class` (cita "orientation/distortion" removidas).
- [ ] `config.py` — alinhar **defaults das dataclasses ao vencedor validado** (hoje
  `use_patch_stats=False`, `preprocess="resize"`, `proj_dim=128`, `temperature` legados;
  `Config()` sem YAML roda o regime errado). E `load()` `test_frac=0.15` ≠ dataclass `0.24`
  (`config.py:124,139`).

### 1.4 Configs de ablação confiáveis (tech-debt HIGH)
- [ ] `configs/ablation_noreflow.yaml` — está **confundido**: além de `reflow_clean`, difere em
  `proj_dim`/`temperature`/`aux_weight`/`objective`/`benign_augment`. **Regenerar do
  `default.yaml` virando só `reflow_clean: false`.** Idem `robust_synthonly.yaml`.
- [ ] `scripts/ablation.py:10` — exemplo aponta para `configs/patchstats.yaml` inexistente →
  `configs/default.yaml`.
- [ ] Adicionar `gate_method`/`knn_k`/`stage2_method`/`triplet_margin` explicitamente nos 3
  configs não-default (ou documentar os defaults).

### 1.5 Código morto / testes
- [ ] Mover para `archive/` (ou apagar): `scripts/build_dataset_indt.py` (superado por
  `rebuild_dataset_indt.py`, 0 refs) e `scripts/visualize_test.py` (0 refs, subsumido por
  `report_processed_v3.py`).
- [ ] `scripts/visualize.py:40,43` — corrigir comentário "6 categorias" e remover chaves mortas
  `distortion`/`orientation` da paleta.
- [ ] Renomear `src/siamese/geometric.py` → `region_detector.py` (desambiguar de `geometry.py`).
- [ ] **Adicionar teste para `src/siamese/synthetic.py`** (os geradores anti-confound
  sustentam o headline e não têm teste) + smoke test de `src/siamese/model.py`.
- [ ] `tests/test_split_isolation.py:52-55` — só cobre `data/processed` legado e pula no v3;
  adicionar checagem de isolamento de split no formato plano.

**Aceite Fase 1:** `pytest` verde; `run_experiment.py --processed data/processed_v3` reproduz a
§0.1 byte-equivalente; nenhum caminho de fallback silencioso; artefatos/docs concordam na
taxonomia; `git status` limpo.

---

## FASE 2 — Coleta de dados limpos diversos (a alavanca principal)

Objetivo: trocar "16 sessões de 1 dia/device" por um manifold limpo **rico em apps, form
factors, orientações e resoluções**, cobrindo as resoluções dos erros. Meta inicial: **≥ 300–500
limpas novas em ≥ 50 grupos distintos** (hoje: 172 em 16).

### 2.1 Especificação dos dados-alvo (derivada da distribuição dos ERROS)

Para resolução/aspecto deixarem de prever o rótulo, as **limpas** precisam cobrir o espaço dos
**erros**:

| Eixo | Alvo (espelhar os erros) |
|---|---|
| Form factor | unfold · fold · laptop · tent · desktop |
| Orientação | portrait **e** landscape |
| Resoluções-chave | 2232×2484, 2484×2232, 1080×2520, 2520×1080, 1272×2772, 2200×2480 **+** manter algumas 2076×2152 |
| Conteúdo | muitos apps/telas distintos (home, settings, listas, formulários, mídia…) |
| Qualidade | screenshots **e** algumas fotos (11 erros reais são foto → cobrir o atalho de nitidez) |

> 🔎 **Refinamento importante:** o pedido foi "tablet near 2076×2152", mas a evidência
> (near-square AR 0.96 + form factors unfold/fold/tent/laptop) aponta para **dispositivo
> dobrável** (ex.: Galaxy Z Fold / Pixel Fold) em várias posturas — **não** um tablet comum.
> Priorizar telas de **dobrável desdobrado** (near-square) e as outras posturas.

### 2.2 Captura no mesmo device (padrão-ouro) — ADIADA (sem acesso agora)

A menor domain-gap viria de capturar limpas nos próprios dobráveis que geram os erros.
**Decisão (jun/2026): sem acesso aos devices → caminho via download (§2.3).** Manter como opção
futura de maior impacto; se surgir acesso, capturar ≥ 30–50 apps × posturas × orientações e
**popular `form_factor`/`orientation`** (hoje `unknown`, o que cega o subconjunto controlado).

### 2.3 Download de datasets públicos (CAMINHO PRIMÁRIO escolhido)

Pesquisa verificada contra fontes primárias (jun/2026). Use telas **normais** como `clean`
(rótulo 0); os erros sintéticos são gerados **sobre** elas (pares casados → sem novo confound de
conteúdo). **Licença importa:** Apache/MIT/CC-BY = livres; WebUI = só-pesquisa; App Store = só
treino interno.

**Recomendação ranqueada:**

| # | Fonte | Licença | Imagens / resoluções (verificadas) | Download |
|---|---|---|---|---|
| **1** | **ScreenSpot-v2 + GroundUI-18K + ScreenSpot-Pro** | Apache-2.0 / MIT | ~16k reais, **multi-res**: phone (1080×2155, 2400×1080), **tablet iPad 2360×1640**, desktop (2560×1440→3000×1687), 2 orientações | `load_dataset("HongxinLi/ScreenSpot_v2")` · `load_dataset("agent-studio/GroundUI-18K")` · `load_dataset("likaixin/ScreenSpot-Pro")` |
| **2** | **WebUI** `biglab/webui-70k-elements` (filtrar iPad-Pro) | **só-pesquisa** | ~10k **2048×2732** (≈ seu 2076 de largura) + iPhone + 4 desktop; WebP q50 | `load_dataset("biglab/webui-70k-elements")` → `.filter(key_name=="iPad-Pro")` |
| **3** | **App Store / Play marketing** (scrape) | copyright → **só treino interno** | iPad nativo **2048×2732**, iPhone, Play tablet; escala ilimitada | `app-store-scraper` + reescrever token URL `/WxHbb.jpg`→`/2048x2732bb.jpg` |
| extra | **AMEX** (CC-BY, 104k Android **1440×2960**) · **WebSight v0.2** (CC-BY, ~2560-wide) · **Common Screens** (CC-BY, 55M web 1920×1080) | livres | buckets extras de resolução | `huggingface-cli download` / `load_dataset` / `aws s3 --no-sign-request` |

**DISPENSADOS (não servem):** RICO/Enrico/CLAY/Screen2Words/UIBert (só **540×960** ou só
metadados — re-imporiam confound de baixa-res); AITW/AITZ/AndroidControl (TFRecord 270–720px);
GUI-World (vídeo, sem licença); Apple Screen-Recognition/Ferret-UI (nunca liberados).

**⚠️ Regras metodológicas (senão o download NÃO quebra o confound):**
- **NÃO** redimensionar tudo para 2076×2152 — re-impõe a uniformidade que criou o confound.
  Manter resoluções **nativas diversas**; dedup por p-hash.
- Confound só morre quando `P(res|clean) ≈ P(res|erro)` → a **mesma** distribuição de resolução
  nos **dois** lados (o pipeline já garante: o sintético é gerado das próprias limpas).
- **Nunca** preencher padding com **preto** (colide com a classe `black_bars`); randomizar
  cor/posição.
- Reencodar **todas** as fontes no mesmo formato/qualidade (senão WebP-vs-PNG vira novo atalho).
- **Portão de aceite (provar que quebrou):** a sonda só-de-resolução `(w,h,aspect,scale)→rótulo`
  deve cair de **AUROC ~1.0 → ~0.5** (já existe como `baseline_resolucao_trivial` em
  `evaluate.py`); rodá-la depois de ingerir os dados novos.

### 2.4 Diversidade de resolução sintética (reforço barato, já parcialmente feito)

`reflow.ar_relayout` + `benign_augment` já tiram a limpa de 2076×2152. Reforçar:
- [ ] Adicionar operador de reflow que **renderiza explicitamente nas resoluções dos erros**
  (§2.1) — não só aspecto aleatório `U(0.5,2.0)`.
- [ ] (Ablação) medir o ganho isolado disso vs. dados reais — para saber quanto do gap é
  resolução (sintetizável) vs. conteúdo (só dados reais resolvem).

### 2.5 Ingestão no pipeline (engenharia)

- [ ] Estender a build do dataset para aceitar uma **fonte de limpas externas**
  (`data/clean_extra/<app>/<...>.png` + um `labels` próprio), já que `rebuild_processed_v3.py`
  hoje rotula via `data/splits/all.csv` (nome→ticket) — limpas baixadas não estão lá.
  Tudo entra como `category=clean, label=0, source=real`, agrupado por app.
- [ ] **Onde colocar (duas finalidades distintas):**
  - **TREINO** (generalização): novas limpas em `train/real` → expandem o cluster limpo →
    derrubam o falso-positivo no teste.
  - **VAL/TESTE** (de-confound da avaliação): novas limpas **nas resoluções dos erros** em
    `val/real`+`test/real` → cria o **held-out onde limpo e erro compartilham resolução**
    (critério de aceite da auditoria) → acurácia real honesta pela 1ª vez.
- [ ] Re-rodar: `extract_features` → `make_synthetic` (regenera sondas+reflow do pool
  expandido) → `run_experiment`. Verificar **0 vazamento de grupo** entre splits (auto-check
  do rebuild já faz; manter).
- [ ] Re-validar hiperparâmetros (head/temperature) na **nova val** — o ótimo pode mudar com
  mais dados (head pode crescer um pouco sem overfit).

**Aceite Fase 2:** manifold limpo com ≥ 50 grupos, ≥ 4 form factors, ≥ 5 resoluções incluindo
as dos erros; **gap treino→teste do AUROC do gate < 0.15**; AUROC livre-de-confound **> 0.75**
estável em multi-seed; especificidade no held-out sobe de forma estável (CI95 acima do alvo).

---

## FASE 3 — Upgrade de métodos (secundário, barato, paralelizável)

O backbone congelado é o **teto de separabilidade** das features (por isso a fina trava em
F1≈0.35). Trocar é barato: re-extrair embeddings + retreinar a cabeça (segundos). Selecionar
**sempre na val livre-de-confound**.

- [ ] **Backbone maior + registers** (disponíveis no `timm` 1.0.27; GPU 17 GB comporta L):
  `vit_base_patch14_reg4_dinov2` (768→2304-d) e `vit_large_patch14_reg4_dinov2`
  (1024→3072-d). Registers melhoram **tokens de patch** (que este projeto usa em mean/std) →
  ganho esperado em erros espaciais e na categoria fina. Benchmark via `grid_search.py`.
- [ ] **Resolução de entrada**: grid `size` 518 → 686/784 (mais patch tokens → detalhe
  espacial p/ black_bars/empty_space). Trade-off memória/tempo.
- [ ] **`preprocess` pad vs resize**: `resize` anamórfico mata o sinal de aspecto/padding (bom
  anti-confound), `pad` preserva geometria (melhor p/ erro espacial, 0.71→0.88). Ablar com o
  novo dataset — o trade-off pode mudar.
- [ ] **(Depois da Fase 2) Fine-tune leve** (LoRA / último bloco do DINOv2): só faz sentido com
  o manifold expandido (hoje ~360 imgs → overfit imediato). Manter cabeça pequena; val-gated.

**Aceite Fase 3:** o melhor backbone/preproc **bate o atual** na AUROC livre-de-confound da val
**e** no F1 da categoria, sem aumentar o gap treino→teste. Congelar o vencedor no `default.yaml`
com nota de proveniência (como já se faz).

---

## FASE 4 — Avaliação honesta e relatório final

- [ ] **Held-out de resolução casada** (da Fase 2.5): reportar acurácia/precisão/recall reais
  onde limpo e erro compartilham resolução — a regra trivial morre, o número fica honesto.
- [ ] Liderar o headline por: AUROC livre-de-confound + subconjunto controlado +
  métricas estratificadas por resolução. **Nunca** por acc/AUROC global confundido nem por
  AP da sonda (acaso 0.80).
- [ ] **Estabilidade**: multi-seed (`scripts/multiseed_stability.py`) + CV agrupado
  (`scripts/nested_cv.py`) com calibração de limiar out-of-fold.
- [ ] `disordered_layout` sair do zero (n=10 hoje, sem poder estatístico) **ou** sair da
  promessa fina (coletar mais ou fundir na grossa).
- [ ] Atualizar `docs/RELATORIO_FINAL_PROCESSED_V3.md`, `DESIGN.md`, `README.md` com os números
  finais; UMAP do relatório **fit só no treino** (hoje é transdutivo em treino+teste — eval/
  tech-debt agents).

**Critérios de aceite finais (da auditoria, `docs/AUDITORIA...md` §4):**
1. AUROC livre-de-confound **> 0.75** estável (multi-seed + CV agrupado).
2. **Gap treino→teste de AUROC < 0.15**.
3. Bate **DINOv2+LogReg (0.716)** e **one-class kNN (0.644)** no subconjunto controlado.
4. Especificidade com CI95-inferior **acima** do alvo operacional.
5. `disordered_layout` fora de recall/F1 = 0 (ou removido da promessa fina).
6. Existe um held-out onde limpo e erro **compartilham resolução/form-factor**.

---

## FASE 5 — Projeto clean (higiene final)

- [ ] Decisão deliberada sobre o **caminho binário legado** (decisão/manifest/features marcados
  "LEGADO"): manter documentado ou remover. Não é código morto, mas é peso.
- [ ] `scripts/stage2_baseline.py` e `scripts/package_dataset.py` (0 refs): wire no pipeline ou
  arquivar.
- [ ] `docs/AUDITORIA...md §6`: `artifacts/audit/processed_v3.json` está obsoleto (não citar);
  `scripts/audit_dataset.py` default → `data/processed_v3/labels.csv`.
- [ ] Mensagens de commit descritivas (hoje "atualizar", "ajystes base").
- [ ] README com o "antes→depois" e o comando único.

---

## Ordem de execução sugerida (ROI)

```
Fase 1 (higiene)  →  Fase 2 (DADOS, maior ROI)  →  Fase 3 (métodos, em paralelo)  →  Fase 4 (eval honesta)  →  Fase 5 (clean)
   1–2 dias              o grosso do esforço            re-extração barata             relatório               polimento
```

- **Quick wins** primeiro (Fase 1) para travar o baseline e parar de medir errado.
- **Fase 2 é onde está o ganho real** — sem dados diversos, nenhum método fecha o gap.
- **Fase 3 roda em paralelo** à coleta (benchmark de backbone não depende de novos dados; o
  fine-tune depende).

## Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Dados baixados criam **novo confound** (domínio-baixado = limpo) | Erros sintéticos gerados **sobre** as próprias limpas baixadas (pares casados); priorizar captura no mesmo device; sempre remapear resolução p/ os alvos |
| Limpas de baixa qualidade → ruído de rótulo | Curar; preferir screenshots nativos; dedup por sha256+p-hash (já existe) |
| Backbone maior → overfit da cabeça | Cabeça pequena; seleção só na val; regra 1-SE |
| Coletar muito do mesmo app | Agrupar por app/sessão; estratificar; limitar nº por grupo |

## Resultados — Fase 2, 1ª iteração (jun/2026)

Pool de **360 limpas** públicas (ScreenSpot-v2 + GroundUI, multi-resolução) fundido em
`data/processed_v3_plus` (clean treino 105→**325**; val 26→80; teste 41→127; sintéticos
regenerados das limpas expandidas). Pipeline: `fetch_clean_extra.py` → `merge_clean_extra.py`
→ `run_experiment.py --config configs/default_plus.yaml`. **Tudo melhorou:**

| Métrica (teste) | Antes (processed_v3) | **Agora (plus, 1ª iter)** | Meta |
|---|---|---|---|
| **Confound (regra trivial resolução)** | 1.000 | **0.661** (sonda LR: 0.99→0.63) | → 0.50 |
| **AUROC livre-de-confound** (headline) | 0.721 | **0.768** | > 0.80 (mín. 0.75 ✅) |
| **Gap treino→teste** (AUROC gate) | 0.40 | **0.24** | < 0.15 |
| Gate AUROC (protótipo) | 0.607 | **0.719** | — |
| Especificidade | 0.585 | **0.685** | estável, CI95 acima do alvo |
| Balanced accuracy | 0.584 | **0.619** | — |
| Estágio 2 grossa F1 | 0.641 | **0.775** | — |
| Estágio 2 fina F1 | 0.354 | **0.421** | > 0.45 |
| `black_bars` detecção AUROC | 0.728 | **0.898** | — |
| **Falseabilidade** (erro vs resolução) | 0.000 (degenerada) | **+0.197** (passa) | passa ✅ |
| Repo | fallbacks silenciosos / inconsistências | **clean, testes verdes (54)** | ✅ |

**Veredito:** a tese do roadmap está PROVADA — telas limpas diversas quebram o confound
(1.0→0.66) **e** sobem as métricas honestas, **reduzindo a disparidade treino→teste pela metade**
(0.40→0.24; o modelo deixou de memorizar o cluster limpo: treino caiu 0.99→0.88, teste subiu
0.60→0.64). **Próximos passos p/ fechar o gap (<0.15) e AUROC>0.80:** (a) mais volume + cobertura
**near-square** (bucket dominante dos erros, hoje fraco no pool — fonte pública escassa; usar
WebUI iPad 2048×2732 / scrape App Store, ou sintetizar near-square); (b) **Fase 3** (backbone
`reg4`/large) sobre o dataset já de-confoundado.

### Resultados — Fase 3 (backbone, jun/2026)

Backbone congelado = teto das features. Comparados 3 (seleção na VAL livre-de-confound,
`scripts/compare_backbones.py`) sobre `processed_v3_plus`; **vencedor = `vit_large_patch14_reg4_dinov2`**
(registers melhoram os patch tokens usados em mean/std). Corrigido `backbone.py` p/ pular
`num_prefix_tokens` (CLS + 4 registers) nas stats de patch. Final-test (held-out) do vencedor:

| Métrica (teste) | Baseline (S, v3) | Plus (S) | **Plus + L_reg4** | Meta |
|---|---|---|---|---|
| **AUROC livre-de-confound** | 0.721 | 0.768 | **0.802** | > 0.80 ✅ |
| **Gap treino→teste** (AUROC) | 0.40 | 0.24 | **0.182** | < 0.15 (quase) |
| Gate AUROC | 0.607 | 0.636 | **0.691** | — |
| `val_synth_gate` (seleção) | 0.81 | 0.81 | **0.857** | — |

Seleção na val: L_reg4 `val_synth_gate` 0.857 / `val_cat_f1` 0.576 > B_reg4 0.847/0.382 > S
0.811/0.453. **As duas alavancas COMPÕEM:** dados quebram o confound e sobem a base (0.72→0.77);
o backbone reg4/large sobe o teto (0.77→**0.80**) e fecha o gap (0.24→**0.18**). Estágio-2 fina no
teste ~estável (knn 0.40) — limitada pelo suporte por classe (`disordered_layout` n=10) e clean
cross-domain; mais dados + **near-square** (Fase 2.b) é o que falta p/ gap<0.15.

### Resultados — Fase 4 (consolidação + estabilidade, jun/2026)

**Estabilidade multi-seed** (`scripts/multiseed_stability.py`, 5 seeds, L_reg4+plus, `val_synth_gate`):
proj64 **0.860 ± 0.001** (std 0.001 — rock-solid), proj128 0.865 ± 0.002, proj256 0.861 ± 0.003,
proj32 0.860 ± 0.003. **O ganho é ESTÁVEL, não sorte de seed** (todos ~0.86 ± ~0.005). O `proj_dim`
está no ruído; mantido **proj64** (mais parcimonioso e de MENOR variância). Logo o headline
(AUROC livre-confound **~0.80** no teste; `val_synth_gate` ~0.86) é reprodutível.

Template do relatório (`run_experiment.py`) corrigido: a prosa hardcoded "toda clean é 2076×2152 /
~98% confound" virou **dinâmica** (usa `baseline_resolucao_trivial`: ~1.0 = confound forte; ~0.5 =
quebrado) — honesta p/ o baseline E p/ o plus. **Config consolidada = `configs/plus_L_reg4.yaml`**
(dados `processed_v3_plus` via `fetch_clean_extra`+`merge_clean_extra`; backbone L_reg4).
Pendente p/ Fase 5: atualizar RELATORIO_FINAL/DESIGN/README com os números novos.

### Resultados — Experimento consolidado `processed_v4_plus` (jun/2026)

Reexecução completa do pipeline (`run_experiment.py` + `report_processed_v3.py`) em
`data/processed_v3_plus` com a config consolidada `plus_L_reg4.yaml`, reunindo **todas** as métricas
pedidas (treino **e** teste) em `artifacts/reports/processed_v4_plus/` (commit `efe19a0`). **Reproduz**
os números das Fases 3–4 (held-out AUROC livre-confound **0.802** / AP 0.940; gate protótipo 0.751 /
fusão 0.691; coarse F1 0.671; fina 0.401) — estável, sem surpresas.

**Gate erro vs sem-erro (matrizes binárias treino + teste):**

| split | Acc | Precisão | Recall | F1 | AUROC | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---|
| Treino (in-sample) | 0.767 | 0.624 | 0.792 | 0.698 | 0.873 | 133/245/80/35 |
| **Teste (held-out)** | 0.619 | 0.463 | 0.657 | 0.543 | 0.691 | 44/76/51/23 |

**Categoria 5 classes (clean + 4 erros):** teste acc 0.505 / F1-macro 0.327 / AUROC-macro 0.596;
treino 0.753 / 0.742 / 0.916. **Por classe (fim-a-fim, teste):** melhor = **`black_bars`** (AUROC 0.783,
F1 0.533, n=22); pior = **`disordered_layout`** (classificação **0.00**, n=10 — o gate detecta ~60% mas
a categoria não é atribuída). Decomposição detecção×classificação e métricas livre-confound completas
no `README_processed_v4_plus.md` / `EXPERIMENT_RESULTS.md`.

**Entregáveis** (`artifacts/reports/processed_v4_plus/`): `clusters_treino.html` / `clusters_teste.html`
(plotly interativo), `confusion_matrix_binaria_{treino,teste}.png`,
`confusion_matrix_categoria_{treino,teste}.png`, `metricas_por_classe.{png,json}`,
`per_class_metrics_en.png`, `evaluation_report_heldout.json`, `README_processed_v4_plus.md`.
**`report_processed_v3.py` ganhou `--out`/`--label`** (retrocompatível) p/ direcionar a saída. O
experimento **reconfirma** o pendente OPCIONAL **Fase 2.b** (near-square) como o caminho p/ tirar
`disordered_layout` do zero e fechar o gap treino→teste <0.15.

### Resultados — Cross-eval + experimento #1 + Spec da Fase 2.b (jun/2026)

**Fase 2.b deixa de ser "OPCIONAL" e passa a ser a restrição que prende o domínio de produção** —
provado por dois experimentos. Spec executável: [`docs/SPEC_COLETA_FOLDABLE.md`](SPEC_COLETA_FOLDABLE.md).
Ferramenta de medida (critério de aceite): **`scripts/domain_slice_eval.py`** — avalia o held-out
restrito a um domínio (near-square / form-factor / v3test) reusando `evaluate(final_test=True)`.

1. **Cross-eval (modelo `plus+L_reg4` → test do `v3`, 41 clean near-square + 67 erros; 0 vazamento):**
   no domínio foldable o headline NÃO se confirma — free-confound **0.731** (vs 0.802 no plus-test;
   vs 0.721 do baseline = +0.01, ruído), especificidade **0.512** (20 FP/41). O 0.802 vem de clean
   **diversas** (phone/desktop), que não são o domínio de produção. Rodar no v3-test é a **avaliação
   desonesta abandonada** (confound trivial volta a 1.000; precisão infla por base-rate) — não reportar.
2. **Experimento #1 (reflow/síntese near-square AR — flag `synthetic.reflow_match_error_ar`, default OFF):**
   mirar o AR dos erros (mediana 0.96) em vez de `U(0.5,2.0)`. **Resultado NEGATIVO no alvo:**
   especificidade foldable **0.512 → 0.512** (os MESMOS 20 FP); só melhorou clean diversas
   (0.640 → 0.709) e regrediu a métrica-guia (0.802 → 0.776). **Prova que o gap é CONTEÚDO, não
   aspecto/resolução** — síntese de AR a partir de conteúdo phone/desktop não cobre o conteúdo foldable.
   Infra mantida (default OFF, testada); experimento revertido.

**Conclusão:** a única alavanca que move o domínio foldable é **dado foldable REAL** (Fase 2.b / Spec).

#### Fase 2.b — CABEADA ✅ (tooling pronto e testado; falta só a coleta) — commits `86c0964` · `58baa0a` · `9925bd5`

Todo o pipeline foldable está implementado e testado (**63 testes verdes**); o único pendente é a
**coleta de dados** (passo 1, requer emulador/device):

| Peça | Estado |
|---|---|
| `scripts/capture_foldable.py` — captura via adb/`--from-file`, `batch`, `audit` | ✅ pronto + testado |
| `data/fold_plan.csv` — plano exemplo (52 capturas, 33 telas, 19 apps, 4 postures) | ✅ |
| `merge_clean_extra.py` (§4.2) — `form_factor` real (destrava o controlado) + split por grupo | ✅ testado ponta a ponta |
| `scripts/domain_slice_eval.py` — fatia foldable-only (critério de aceite §5) | ✅ reproduz o cross-eval |
| `configs/plus_fold_L_reg4.yaml` — config do passo 3 (L_reg4, emb/reports próprios) | ✅ |
| **Coleta** — `capture_foldable.py batch` no emulador/device → ≥50 telas, 4 postures | ⬜ **pendente** |

**Decisão de método:** construir SOBRE o `processed_v3_plus` (mantém o de-confounding do pool público)
+ foldable → `processed_v3_plus_fold`; NÃO sobre o v3 cru (clean ficaria todo near-square → confound
poderia voltar). Execução pronta (spec [§7](SPEC_COLETA_FOLDABLE.md)): coletar → `merge_clean_extra
--src data/processed_v3_plus --extra data/clean_extra_fold --dest data/processed_v3_plus_fold` →
`run_experiment --config configs/plus_fold_L_reg4.yaml` → `domain_slice_eval --subset form-factor`.
**Gate de aceite: a especificidade foldable tem que sair de 0.512.**

#### Sem coleta possível → extrair dos dados atuais (A1–A4) + relatório honesto — jun/2026

Restrição nova: **a coleta de fotos foldable ficou indisponível**. Pivot: esgotar os métodos que
**não precisam de dados novos** e reportar honestamente. Relatório completo:
[`docs/RELATORIO_FOLDABLE.md`](RELATORIO_FOLDABLE.md). Reprodução: `scripts/foldable_operating_point.py`.

- **A2** (backbone p/ o foldable): B_reg4 ≈ L_reg4 **dentro do ruído** (A4: diff +0.03, IC95 [−0.04, 0.14]) → manter L_reg4.
- **A1** (recalibração): o lever real **não** foi o limiar por bucket (+0.02), foi trocar o gate
  **fusão → protótipo** (a fusão degrada o ranking: proto AUROC 0.751 > fusão 0.691 já no global).
- **Resultado:** L_reg4 + **gate de protótipo** + limiar foldable → especificidade foldable
  **0.512 → 0.683** (IC95 [0.571, 0.835]) com recall mantido (~0.64), **de graça**. AUROC foldable
  0.66 [0.55, 0.82] — modelo **moderado**, melhoria **real porém sample-bound**.
- **A4** (estabilidade): ICs ~±0.15 (41 clean) → **teto DUPLO provado**: conteúdo (#1) **e** amostra.
- **Decisão de produto:** adotar o gate de protótipo no bucket foldable (sem downside); escopo =
  taxonomia grossa + `black_bars`; reportar o falso-alarme foldable abertamente. Só **dado foldable
  real** (tooling CABEADO acima) levanta o teto.
