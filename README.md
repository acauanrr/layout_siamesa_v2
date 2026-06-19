# siamese-ui-error

Detecção de erro de layout em screenshots/fotos de UI de celular com uma **rede siamesa**
(cabeça de projeção de pesos compartilhados) sobre **DINOv2 ViT-S/14 congelado**, em
**dois estágios**:

1. **Gate "tem erro?"** (Estágio 1) — decisão por **proximidade a protótipos do cluster
   "limpo"** + cabeça auxiliar, com limiar calibrado (alta precisão).
2. **Clusterização por categoria** (Estágio 2) — quando há erro, atribui a **categoria**
   (`black_bars · disordered_layout · distortion · empty_space · orientation · overlay`)
   pelo **protótipo de categoria mais próximo**, treinado por SupCon multi-classe.

> **Resultado (test held-out = 131 imagens: 41 limpas + 90 erros; config do grid search; métricas em TREINO e TESTE):**
> **Estágio 1** — TESTE: ponto balanceado **F1 0.85 · acc 0.78 · AUROC 0.79 · AP 0.87**; detecção
> sintética livre de confound **AUROC 0.81 / AP 0.94**; controlado **AUROC 0.69**; alta-precisão
> **0.89 @ recall 0.28**; P@5 1.0. TREINO (in-sample): F1 0.99 — o gap treino→teste é o
> overfitting esperado em base pequena.
> **Estágio 2** — categoria: TESTE **F1-macro 0.23 / acc 0.36** (forte em black_bars/overlay; fraca
> nas raras/sutis), TREINO 1.0 (protótipos ajustados no treino) — ver §abaixo.

> ⚠️ **Mudanças recentes (jun/2026):** entrada de erros migrada de `with_errors/` (flat,
> binário) para **`errors_dataset/<categoria>/`**; imagens com **marcações vermelhas** humanas
> foram **excluídas** (35); pipeline estendido para **multi-cluster** e hiperparâmetros
> otimizados por **grid search**; **test ampliado** (`test_frac=0.24` → 41 telas limpas, ≥40,
> p/ estimativa robusta de falso-alarme); **3 imagens duplicadas** removidas → **541 imagens reais
> únicas**; fonte de dados = **`data/processed/`** (SSOT); treino com **early-stop sobre o sintético
> de validação** (livre de confound — estabiliza a seleção). Detalhes em [`docs/DESIGN.md` §10](docs/DESIGN.md).
> O detector **binário legado** continua reprodutível (`--source with_errors`, `train.multiclass: false`).

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
                             · grid_search · ablation · compare_preprocess · localize · visualize
                             · dump_synthetic · predict · audit_red_marks
artifacts/                   embeddings/ models/ reports/ synthetic_images/ (gerados)
data/input/no_erros/             172 telas limpas (label 0, category=clean; device único 2076×2152)
data/input/errors_dataset/<cat>/ 369 erros por categoria (após limpeza + dedup): black bars 112 ·
                                 disordered layout 55 · distortion 13 · empty space 67 ·
                                 orientation 7 · overlay 115
data/input/with_errors/          fonte ANTIGA flat (188), preservada só p/ o caminho binário legado
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

> 🎯 **`data/processed/` é a FONTE DA VERDADE** — o dataset categorizado (reais + sintéticos)
> que o modelo treina/valida/testa **e** que é compartilhado com as outras equipes. `data/input/`
> é só a **entrada bruta**: novas imagens chegam nela; após processar/corrigir, o que vale é o que
> está em `processed/`. `extract_features.py` lê de `processed/` (não de `input/`), então
> correções manuais em `processed/` são honradas.

```bash
# 1. split agrupado por ticket + estratificado por categoria (a partir de data/input/)
#    (--source errors_dataset é o padrão; use --source with_errors p/ o binário legado)
python scripts/build_splits.py --input data/input --out data/splits

# 2. MATERIALIZA o dataset canônico em data/processed/ (reais por categoria + sintéticos de treino)
python scripts/export_processed.py --config configs/default.yaml

# 3. embeddings DINOv2 a partir de data/processed/ (a FONTE DA VERDADE; pad = padding cinza)
python scripts/extract_features.py --processed data/processed --out artifacts/embeddings --use-patch-stats --preprocess pad

# 4. sonda sintética livre de confound (val/test) a partir de processed/{val,test}/real/clean
python scripts/make_synthetic.py --config configs/default.yaml

# 5. treina a cabeça siamesa multi-classe (segundos)
python scripts/train.py --config configs/default.yaml

# 6. avaliação honesta — Estágio 1 (gate) + Estágio 2 (categoria: matriz NxN, F1-macro)
python scripts/evaluate.py --config configs/default.yaml

# (opcional) grid search de hiperparâmetros, seleção pela métrica honesta (livre de limiar)
python scripts/grid_search.py --config configs/default.yaml --rank-by synth_auroc
```

> **Novas imagens / correções:** novas imagens entram em `data/input/` → re-rode os passos 1–2
> para regenerar `processed/`. Se você **corrigir/ajustar imagens diretamente em `processed/`**
> (mover de categoria, remover, editar), **NÃO** re-rode o passo 2 (ele reconstrói a partir de
> `input/` e sobrescreveria suas correções) — rode direto dos passos **3→6** (que leem `processed/`).

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
| ⭐ **`clusters_apresentacao.html`** | **APRESENTAR (gate, Estágio 1)** — antes×depois interativo: limpo forma cluster e erros se afastam, com o **roteiro do que falar** embutido (separação por distância sobe de AUROC ~0.60 → ~0.88) |
| ⭐ **`categorias_apresentacao.html`** | **APRESENTAR (multi-cluster, Estágio 2)** — antes×depois interativo colorido pelas **7 classes** (clean + 6 categorias) com os **protótipos de categoria**; mostra os clusters por tipo de erro |
| `embedding_categorias.png` | versão estática do espaço z colorido por categoria + protótipos (apoio do relatório) |
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
