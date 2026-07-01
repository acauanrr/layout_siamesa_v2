# siamese-ui-error

Detecção de erro de **layout** em screenshots/fotos de UI de celular com uma **rede siamesa**
(cabeça de projeção de pesos compartilhados) sobre **DINOv2 ViT-S/14 congelado**, em **dois estágios**:

1. **Gate "tem erro?"** (Estágio 1) — decisão por **proximidade a protótipos do cluster "limpo"**
   fundida com uma cabeça auxiliar, no limiar calibrado.
2. **Categoria do erro** (Estágio 2, só se houver erro) — atribui a **categoria** por **k-NN de
   categoria** no espaço aprendido (SupCon multi-classe; ver [`docs/COMPARACAO_KNN_TRIPLET.md`](docs/COMPARACAO_KNN_TRIPLET.md)).

**Taxonomia (5 classes):** `clean` + 4 erros — `black_bars` · `disordered_layout` · `empty_space` · `overlay`.

> ### 🚀 Experimento completo em UM comando
> ```bash
> python scripts/run_experiment.py --processed data/processed_v3   # treina + avalia + teste held-out 1×
> #   --fresh  reconstrói tudo (re-extrai embeddings, sondas)
> ```
> Saída: **`artifacts/reports/EXPERIMENT_RESULTS.md`** (tabela + veredito) · `EXPERIMENT_RESULTS.json` ·
> `evaluation_report.json` (métricas completas) · matrizes de confusão. Os demais scripts são chamados
> por ele internamente.
>
> **Artefatos visuais** (clusters, matrizes, métricas por classe):
> ```bash
> python scripts/report_processed_v3.py --config configs/default.yaml   # → artifacts/reports/processed_v3/
> ```

---

## O ponto central: o confound de resolução

O dataset tem um **confound quase perfeito**: **toda** tela `clean` real é **2076×2152** (um único
device), enquanto as telas com erro têm resoluções heterogêneas. A regra trivial
*"resolução ≠ 2076×2152 ⇒ erro"* sozinha dá **AUROC 1.000** no teste — **sem olhar o layout**. Logo,
qualquer métrica global ingênua é ~98% **trapaça** (detecta o device, não o erro).

**Solução (núcleo do projeto):** injetar **erros sintéticos** nas próprias telas limpas, na **mesma
resolução** (`synthetic.py`), criando pares onde só o *conteúdo do erro* muda — isso força o modelo a
aprender o **erro**, não o device. Complementos anti-confound: variantes **reflow** limpas (`reflow.py`)
e **benign augment** (round-trip de resolução) quebram o atalho pelo lado limpo; e o **masking de
padding** nas estatísticas de patch impede que a borda cinza vire pista.

> **O modelo NÃO explora o atalho** (verificado): o baseline trivial de resolução dá AUROC **1.000**,
> mas o modelo dá **0.60** no teste real — se ele trapaceasse, todas as métricas seriam ~1.0. Mais: o
> AUROC **livre de confound (0.72)** é *maior* que o real (0.60) — o inverso de um trapaceiro. Detalhe
> e as 6 provas em [`docs/RELATORIO_FINAL_PROCESSED_V3.md`](docs/RELATORIO_FINAL_PROCESSED_V3.md).

> **Atualização (jun/2026 — Fases 2–3, ver [`docs/ROADMAP.md`](docs/ROADMAP.md)):** o confound foi
> **quebrado na origem** coletando telas limpas diversas multi-resolução (download-only:
> `scripts/fetch_clean_extra.py` → `merge_clean_extra.py` → `data/processed_v3_plus`). A regra trivial
> de resolução caiu de **1.000 → 0.661**; com o backbone `vit_large_patch14_reg4_dinov2` o **AUROC
> livre-de-confound subiu 0.72 → 0.80** e o **gap treino→teste caiu 0.40 → 0.18**. Pipeline reprodutível
> em `configs/plus_L_reg4.yaml`; resultado estável em multi-seed.

## Resultados (teste held-out · `processed_v3`)

Teste = **108 imagens** (41 limpas + 67 erros), avaliado **uma única vez** após congelar a config
(seleção só na validação; teste trancado por `siamese.protocol`). Treino = 273 reais (+419 sintéticos
+420 reflow); val = 68.

### Estágio 1 — "tem erro?"

| Avaliação | Acurácia | Precisão | Recall | AUROC | Leitura |
|---|---:|---:|---:|---:|---|
| **Livre de confound** (erro injetado na limpa, mesma resolução) | 0.68 ᵇ | — | 0.78 | **0.72** | medida **justa** (acaso 0.50) |
| **Erros reais** (ponto de operação) | 0.58 | 0.70 | 0.58 | 0.60 | rendimento no mundo real |
| Baseline trivial de resolução | 1.00 | — | — | **1.00** | teto de trapaça (diagnóstico) |

<sub>ᵇ acurácia **balanceada**. ⚠️ Na sonda livre-de-confound a **AP=0.90 engana** (80% positivos → acaso
da AP = 0.80); o sinal real é o **AUROC 0.72** (acaso 0.50).</sub>

### Estágio 2 — categoria do erro

| Taxonomia | Acurácia | F1-macro | Nota |
|---|---:|---:|---|
| **Grossa** (2 super-classes: região-morta / deslocado) | 0.64 | 0.64 [IC95 0.52–0.75] | primária |
| Fina (4 classes) | 0.43 | 0.35 | exploratória — confiável só em `black_bars` |

**`black_bars`** é a classe forte (melhor detectada **e** classificada); `disordered_layout`/`empty_space`
são fracas. Métricas por classe e clusters em `artifacts/reports/processed_v3/`.

> ⚖️ **Métrica para medir aprendizado e comparar modelos: AUROC livre de confound** — única imune ao
> atalho de resolução, independente de base-rate e de limiar (e é o critério de early-stop). **Não**
> lidere com acurácia/AUROC global (confundidos), AP da sonda (acaso 0.80) nem métricas de treino
> (ressubstituição). Se um modelo concorrente mostrar acurácia muito maior, verifique se não está só
> lendo a resolução.

## Arquitetura

```
TREINO: limpas reais + erros reais + ERROS sintéticos (mesma res) + LIMPAS-reflow
imagem ─► padding CINZA 518×518 (+máscara) ─► DINOv2 ViT-S/14 ❄ CONGELADO ─► CLS + mean/std dos patches de conteúdo (1152-d)
        └► cabeça de projeção compartilhada g(·)  [TREINÁVEL ~314k]  ─► z ∈ 64-d (L2-norm)
           perda: SupCon(z, τ=0.1) + 0.3·CE(aux, 5 classes)   ·   early-stop = AUROC val livre de confound
           ╞═ ESTÁGIO 1 (gate "tem erro?") ═══════════════════════════════════════════════
           │   score de protótipo limpo: 1 − cos(z, protótipo-limpo-mais-próximo)
           │   ⊕ cabeça aux: P(erro) = 1 − P(clean)   ─► fusão calibrada na VAL livre de confound
           │   ─► p(erro) ─► limiar (specificity-first, alvo 0.80)
           ╘═ ESTÁGIO 2 (categoria; só se E1=erro) ══════════════════════════════════════
               k-NN de CATEGORIA (vizinhos de erro de treino) → 4 fina / 2 grossa
```
Diagrama completo: [`docs/pipeline.mmd`](docs/pipeline.mmd) (renderiza no GitHub/VS Code).

## Estrutura

```
configs/default.yaml      configuração congelada (backbone · head · treino · decisão)
data/processed_v3/        FONTE DA VERDADE — dataset plano + labels.csv (gitignored: privacidade)
docs/                     DESIGN.md · RELATORIO_FINAL_PROCESSED_V3.md · pipeline.mmd · AUDITORIA_* · results_comparison.tex
src/siamese/
  config.py               dataclasses + carregamento do YAML
  manifest.py             taxonomia + parsing de metadados + split agrupado por ticket
  geometry.py             pré-processamento (padding cinza + máscara de patch)
  backbone.py             DINOv2 congelado (extrator de features)
  features.py             extração/cache de embeddings (lê labels.csv)
  synthetic.py            injeção dos erros sintéticos (anti-confound, lado do ERRO)
  reflow.py               variantes LIMPAS de layout legítimo (anti-confound, lado LIMPO)
  synth_features.py       embeddings de sintéticos + reflow + benign
  model.py                ProjectionHead · SiameseNet · SiamesePairHead
  losses.py               SupCon + contrastiva de pares
  train.py                treino da cabeça (early-stop livre de confound)
  decision.py             protótipos + seleção de limiar (F1 / precisão / especificidade)
  evaluate.py             avaliação honesta (controlado · sintético · baselines · falseabilidade · IC95)
  protocol.py             trava do teste held-out (anti-snooping)
  infer.py · localize.py · geometric.py   inferência · heatmaps · detector geométrico (apoio)
scripts/                  run_experiment (orquestra tudo) · report_processed_v3 (artefatos visuais)
                          · rebuild_processed_v3 · extract_features · make_synthetic · train · evaluate
                          · grid_search · nested_cv · ablation · visualize · predict · …
tests/                    suíte pytest (protocolo, isolamento de split, hiperparâmetros)
```

## Instalação

GPU de desenvolvimento: RTX 5070 Ti (Blackwell, sm_120) ⇒ PyTorch **cu128**.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -e .            # pacote `siamese` + deps (timm, scikit-learn, plotly, umap-learn…)
```
O backbone DINOv2 (`vit_small_patch14_dinov2.lvd142m`, ~85 MB) é baixado pelo `timm` na 1ª execução e
fica em cache em `~/.cache/huggingface/hub`.

## Pipeline (passo a passo)

O `run_experiment.py` faz tudo; abaixo o detalhamento (dataset plano `processed_v3` já vem splitado, então
os passos de split/materialização são pulados):

```bash
# 1. embeddings DINOv2 a partir de data/processed_v3 (padding cinza + patch stats)
python scripts/extract_features.py --processed data/processed_v3 --use-patch-stats --preprocess pad

# 2. sondas: sintético livre de confound (val/test) + reflow-clean (train/val/test)
python scripts/make_synthetic.py --config configs/default.yaml --processed data/processed_v3

# 3. treina a cabeça siamesa (segundos; backbone congelado)
python scripts/train.py --config configs/default.yaml

# 4. avaliação — DEV (val, iterar à vontade) e TESTE (1×, após congelar a config)
python scripts/evaluate.py --config configs/default.yaml                # DEV (val)
python scripts/evaluate.py --config configs/default.yaml --final-test   # TESTE held-out (1×)
```

Reconstruir o dataset do zero (dedup + split agrupado por ticket): `scripts/rebuild_processed_v3.py`.
Seleção honesta de hiperparâmetros (sem tocar o teste): `grid_search.py`, `nested_cv.py`, `multiseed_stability.py`.

## Inferência (roteamento por domínio)

A inferência **roteia pela resolução nativa**: telas **near-square (foldable)** → gate de **protótipo +
limiar foldable**; as demais → gate **fundido global**. Isso corrige o falso-alarme no foldable (espec
0.51 → 0.68), de graça. Ver [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

```bash
python scripts/predict.py --models artifacts/bb_L_reg4/models img1.png img2.png   # p(erro) + decisão
```
```python
from siamese.infer import Predictor
r = Predictor("artifacts/bb_L_reg4/models", route_foldable=True).predict("tela.png")   # default True
# {'decisao': 'ERRO', 'gate': 'foldable_prototipo'|'global_fusao', 'near_square': bool, 'p_erro': ...}
```

## Limitação central (honesta)

O detector é **bom no caso comum** (phone/desktop: AUROC livre-confound ~0.80) mas **triagem no
foldable** (near-square, o domínio de produção: AUROC ~0.66, especificidade 0.68 roteada). A clean
foldable vem de **16 sessões de 1 device** → o gargalo é **DADO**, com **teto DUPLO provado**: conteúdo
(sintetizar aspecto não move a especificidade) **e** tamanho de amostra (41 clean → IC ~±0.15). Só
**dado foldable real** levanta (infra de coleta pronta). Estado honesto e por-domínio:
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) · [`docs/RELATORIO_FOLDABLE.md`](docs/RELATORIO_FOLDABLE.md) ·
[`docs/DESIGN.md`](docs/DESIGN.md).

## Documentação

| Documento | Para quê |
|---|---|
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | **⭐ Comece aqui** — o que faz/não faz **por domínio**, pontos de operação, teto |
| [`docs/RELATORIO_FOLDABLE.md`](docs/RELATORIO_FOLDABLE.md) | Domínio foldable: melhor config extraível + os dois tetos (conteúdo + amostra) |
| [`docs/SPEC_COLETA_FOLDABLE.md`](docs/SPEC_COLETA_FOLDABLE.md) | O que levantaria o teto: spec de coleta foldable (infra pronta) |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Histórico completo das fases (diagnóstico → tooling → desfecho) |
| [`docs/RELATORIO_FINAL_PROCESSED_V3.md`](docs/RELATORIO_FINAL_PROCESSED_V3.md) | **Resultados + veredito** (confound, vazamento, métrica recomendada) |
| [`docs/COMPARACAO_KNN_TRIPLET.md`](docs/COMPARACAO_KNN_TRIPLET.md) | k-NN vs protótipo · Triplet vs SupCon (respostas à supervisão, com dados) |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Detalhamento técnico e justificativa de cada decisão |
| [`docs/pipeline.mmd`](docs/pipeline.mmd) | Diagrama do pipeline (Mermaid) |
| [`docs/AUDITORIA_PROCESSED_V3_TREINO_TESTE.md`](docs/AUDITORIA_PROCESSED_V3_TREINO_TESTE.md) | Auditoria do treino/teste |
| [`docs/results_comparison.tex`](docs/results_comparison.tex) | Tabela LaTeX (processed_v3 vs dataset_indt) |
