# siamese-ui-error

Detecção **binária** ("tem erro de layout" vs "não tem") em screenshots/fotos de UI de
celular, com uma **rede siamesa** (cabeça de projeção de pesos compartilhados) sobre o
backbone **DINOv2 ViT-S/14 congelado** e decisão por **proximidade a protótipos** do
cluster "limpo".

> **Resultado (test held-out, ponto balanceado):**
> **acurácia 0.85 · precisão 0.86 · recall 0.86 · F1 0.86 · AUROC 0.90 · AP 0.92**

### 📚 Documentação

| Documento | Para quê |
|---|---|
| **[`docs/RELATORIO_APRESENTACAO.md`](docs/RELATORIO_APRESENTACAO.md)** | **Apresentação para a equipe** — problemas/soluções, resultados, diferencial, acurácia |
| [`docs/pipeline.mmd`](docs/pipeline.mmd) | Diagrama do pipeline (Mermaid; renderiza no GitHub/VS Code) |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Detalhamento técnico e justificativa de cada decisão |

---

## Por que este projeto é diferente (o confound)

O dataset tem um **confound quase perfeito**: **toda** tela *sem-erro* é 2076×2152 (um único
device), enquanto as *com-erro* são heterogêneas. A regra trivial "resolução ≠ 2076×2152 ⇒
erro" já dá **AUROC 0.982 / acurácia 98%** — *sem olhar o layout*. Logo, qualquer métrica
global ingênua é ~98% **trapaça** (detecta o dispositivo, não o erro).

**Solução central:** **injetar erros sintéticos** nas próprias telas limpas (mesma
resolução), criando pares onde só o *conteúdo do erro* muda — forçando o modelo a aprender o
erro, não o device. A prova de detecção real está no **teste sintético livre de confound:
AUROC 0.88 / AP 0.97**.

## Resultados

Test = **54 imagens held-out** (nunca vistas), agrupado por ticket. (`scripts/evaluate.py`)

| Métrica | **Valor** | Obs. |
|---|---|---|
| **Acurácia** | **0.85** | IC 95%: 0.75–0.94 |
| **Precisão** | **0.86** | |
| **Recall** | **0.86** | |
| **F1** | **0.86** | |
| **AUROC** | **0.90** | livre de limiar (melhor p/ comparar modelos) |
| **AP (PR-AUC)** | **0.92** | livre de limiar |
| Detecção sintética (livre de confound) | **AUROC 0.88 / AP 0.97** | mede detecção real de erro |
| precision@10 | **1.00** | topo do ranking de suspeita |

Matriz de confusão (ponto balanceado): TP=24, TN=22, FP=4, FN=4 →
[`artifacts/reports/confusion_matrix.png`](artifacts/reports/confusion_matrix.png).

> ⚖️ **Comparação justa:** lidere com **AUROC/AP** (não dependem de limiar). Se um modelo
> concorrente mostrar acurácia muito maior neste dataset, verifique se ele não está apenas
> explorando o confound de resolução (que sozinho dá 98%). Ver relatório §6.

## Arquitetura

```
imagem ─► padding CINZA 518×518 (+máscara) ─► DINOv2 ViT-S/14 (CONGELADO) ─► CLS + mean/std dos patches (1152-d)
        └► cabeça de projeção compartilhada g(·) ─► z (128-d, L2-norm)   [perda: SupCon(z) + 0.3·BCE(aux)]
           ├► score de protótipo: 1 − cos(z, protótipo-limpo-mais-próximo)   ← a ideia de clustering
           └► cabeçalho auxiliar: Linear(128→1)                              ← detector binário direto
           └► fusão calibrada ─► p(erro) ─► limiar de operação (balanceado por padrão)
```

Diagrama completo com todas as relações: [`docs/pipeline.mmd`](docs/pipeline.mmd)
(renderiza no GitHub/VS Code) ou embutido no [relatório](docs/RELATORIO_APRESENTACAO.md#3-bis-diagrama-do-pipeline).

**Função de erro:** `L = SupCon(z) + 0.3 · BCE(cabeça auxiliar)`. Usamos **Supervised
Contrastive** (não Triplet clássica); a comparação **âncora vs protótipos** acontece na
decisão. Detalhes no relatório §4.1.

## Estrutura

```
configs/default.yaml         configuração (backbone, head, treino, decisão)
docs/                        RELATORIO_APRESENTACAO.md · DESIGN.md · pipeline.mmd
src/siamese/
  config.py                  dataclasses + carregamento do YAML
  manifest.py                parsing de metadados + split agrupado por ticket (sem vazamento)
  geometry.py                pré-processamento (padding cinza + máscara de patch)
  backbone.py                DINOv2 congelado (extrator de features)
  features.py                extração e cache de embeddings
  synthetic.py               injeção dos 5 tipos de erro sintético
  synth_features.py          embeddings dos erros sintéticos
  model.py                   ProjectionHead · SiameseNet · SiamesePairHead
  losses.py                  SupCon + contrastiva de pares
  train.py                   treino da cabeça
  decision.py                protótipos + seleção de limiar (F1 balanceado / alta precisão)
  evaluate.py                avaliação honesta (controlado, sintético, baselines, falseabilidade)
  localize.py                heatmaps (PatchCore + supervisionado)
  geometric.py               detector geométrico de black-region/empty-space
  infer.py                   inferência self-contained
scripts/                     build_splits · extract_features · make_synthetic · train · evaluate
                             · ablation · compare_preprocess · localize · visualize · dump_synthetic · predict
artifacts/                   embeddings/ models/ reports/ synthetic_images/ (gerados)
data/input/{no_erros,with_errors}/   dataset de entrada (172 limpas + 188 com erro)
```

## Instalação

GPU usada no desenvolvimento: RTX 5070 Ti (Blackwell, sm_120) ⇒ PyTorch **cu128**.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -e .            # instala o pacote `siamese` + dependências (timm, sklearn, plotly, umap...)
```

O backbone DINOv2 (`vit_small_patch14_dinov2.lvd142m`, ~85 MB) é baixado pelo `timm` na
primeira execução e fica em cache em `~/.cache/huggingface/hub` (acelera execuções futuras).

## Pipeline completo

```bash
# 1. splits agrupados por ticket + auditoria de confounds
python scripts/build_splits.py --input data/input --out data/splits

# 2. cache de embeddings DINOv2 (congelado, roda 1x)  — pad = padding cinza
python scripts/extract_features.py --splits data/splits --out artifacts/embeddings --use-patch-stats --preprocess pad

# 3. erros sintéticos (anti-confound) para train/val/test
python scripts/make_synthetic.py --config configs/default.yaml

# 4. treina a cabeça siamesa (segundos)
python scripts/train.py --config configs/default.yaml

# 5. avaliação honesta — imprime o PONTO DE OPERAÇÃO (acurácia/precisão/F1/AUROC)
python scripts/evaluate.py --config configs/default.yaml
```

Opcionais — as evidências do relatório:

```bash
python scripts/ablation.py --config configs/default.yaml          # prova: sintético quebra o confound
python scripts/compare_preprocess.py                              # resize vs pad (padding cinza)
python scripts/visualize.py --config configs/default.yaml --target-precisions 0.85,0.95  # → clusters_apresentacao.html (apresentar) + PNGs  [--extra-html: + HTML TP/TN/FP/FN]
python scripts/dump_synthetic.py --config configs/default.yaml    # salva as imagens sintéticas
```

## Inferência

```bash
python scripts/predict.py --models artifacts/models img1.png img2.png
python scripts/predict.py --models artifacts/models --dir data/input/with_errors
```
Saída: `p(erro)` por imagem, ordenado, com a decisão no limiar de operação.

```python
from siamese.infer import Predictor
pred = Predictor("artifacts/models")
print(pred.predict("alguma_tela.png"))   # {'p_erro': 0.71, 'decisao': 'ERRO', ...}
```

## Onde está o erro (mapas de calor)

```bash
# PatchCore (padrão): novidade vs telas limpas — frio em tela limpa, destaca conteúdo estranho
python scripts/localize.py --config configs/default.yaml --dir data/input/with_errors --n 12
# geométrico: localiza barras pretas/vazios (preciso p/ black-region; é EVIDÊNCIA, não decisão)
python scripts/localize.py --config configs/default.yaml --geometric img.png
```
Overlays em `artifacts/reports/heatmaps/`. **NB:** localização é *aid de atenção*, não
classificador (o `p(erro)` vem do modelo via `predict.py`). Ver `docs/DESIGN.md` §7b.

## Visualizações (ver o modelo funcionando)

`scripts/visualize.py` gera, em `artifacts/reports/`:

| Arquivo | Mostra |
|---|---|
| ⭐ **`clusters_apresentacao.html`** | **APRESENTAR** — antes×depois interativo: a clusterização e o protótipo, com o **roteiro do que falar** embutido (separação por distância sobe de AUROC 0.58 → 0.94) |
| `embedding_space.png` | DINOv2 cru (misturado) **vs** z aprendido (limpo vira cluster) + protótipo |
| `decision_space.png` | distância ao protótipo (limpo perto de 0) + curva PR |
| `outcome_space.png` | TEST por TP/TN/FP/FN — **onde o modelo erra** |
| `tradeoff_outcome.png` | precisão×recall lado a lado (ex.: 0.85 vs 0.95) |
| `embedding_interactive*.html` | acerto/erro (TP/TN/FP/FN) por limiar — **opcional**, gere com `--extra-html` |

## Modos de operação e variantes

- **Ponto de operação** (`decision.objective` no config):
  - `f1` (**padrão**) — balanceado, números justos para comparação (acurácia 0.85 / F1 0.86).
  - `precision` — alta precisão (1.00 @ recall 0.50, zero falsos-alarmes); usa `target_precision`.
- **Detector agnóstico de device** (robusto a confound): `train.use_real_errors: false`
  (treina só com limpas + sintéticos) e retreine.
- **CLS-only** (mais leve, pior em erros espaciais): `backbone.use_patch_stats: false`.

## Limitação central

Com a classe sem-erro vinda de **um único device**, **nenhuma arquitetura demonstra "alta
precisão independente de confound"** de forma estatisticamente sólida. A alavanca decisiva é
**coletar telas limpas diversas** (outros devices/resoluções, fotos sem erro,
landscape/laptop/tent). Ver relatório §8 e `docs/DESIGN.md` §1/§9.
