# Model Card — detector de erro de layout de UI (siamese + DINOv2)

> Uma página, honesta e **por domínio**. Resume o que o modelo faz, o que **não** faz, os pontos de
> operação e o teto. Números do held-out (`processed_v3_plus`), avaliado 1× após congelar a config.

## Resumo
- Cabeça siamesa sobre **DINOv2 ViT-L/14 reg4 congelado**; decisão em **2 estágios** (gate "tem erro?"
  binário + categoria do erro). Config congelada: `configs/plus_L_reg4.yaml`.
- **Uso pretendido:** detectar/triar erros de layout em screenshots de UI. **Não** é um classificador
  de conteúdo geral nem funciona fora de screenshots de app.

## Desempenho POR DOMÍNIO (o principal)
O modelo **não é uniforme** — é bom no caso comum e **triagem** no foldable. Reporte assim:

| Domínio | Papel | AUROC livre-confound | Ponto de operação (held-out) |
|---|---|---|---|
| **Padrão** (phone/desktop) | **detector** | ~**0.80** | entrega automática |
| **Foldable near-square** (produção) | **triagem/assistência** | ~**0.66** [0.55, 0.82] | espec **0.68** [0.57, 0.84] / recall **0.64** (gate roteado) |

> ⚠️ **Foldable NÃO é detector autônomo** — é um filtro de triagem: sinaliza telas suspeitas para
> revisão humana. A especificidade não permite decisão automática sem revisão.

## Categoria do erro (Estágio 2, só quando há erro)
- **Grossa** (2 super-classes) — F1-macro ~**0.67**: **confiável**, reporte como primária.
- **Fina** (4 classes) — confiável **só em `black_bars`** (AUROC 0.78–0.90). `disordered_layout`
  (n=10) **não é confiável** — fora da promessa.

## Pontos de operação foldable (a decisão de produto — escolha um)
| Prioridade | Especificidade | Recall (erros pegos) |
|---|---:|---:|
| Equilíbrio (default do bundle) | ~0.68 | ~0.64 |
| Menos falso-alarme | ~0.73 | ~0.51 |
| Triagem (recall alto) | ~0.49 | ~0.81 |

Ajustável recalibrando o limiar foldable (`scripts/foldable_operating_point.py --target-spec ...`).

## Como usar (roteamento automático por domínio)
A inferência roteia pela **resolução nativa**: near-square (AR 0.85–1.18) → **gate de protótipo +
limiar foldable**; demais → **gate fundido + limiar global**. Ligado por padrão:
```python
from siamese.infer import Predictor
p = Predictor("artifacts/bb_L_reg4/models", route_foldable=True)   # default True
r = p.predict("tela.png")     # r["gate"] ∈ {global_fusao, foldable_prototipo}, r["near_square"]
```

## Limitações e teto (honesto — leia antes de confiar)
- **Foldable travado por DADOS:** a clean foldable são **16 sessões de 1 device/1 dia**. Teto **DUPLO
  provado**: **conteúdo** (exp. #1: sintetizar aspecto não move a especificidade, 0.512→0.512) **e
  tamanho de amostra** (41 clean → todo IC ~±0.15). **Só dado foldable real levanta** — infra de coleta
  pronta (`scripts/capture_foldable.py`, ver `docs/SPEC_COLETA_FOLDABLE.md`).
- **Não certificar** precisão alta nem detecção fina confiável no foldable.
- **`v3-test` é avaliação confundida** (resolução separa clean/erro) — **não reportar**. Liderar sempre
  por **AUROC livre-de-confound**; nunca por acc/AUROC global no real nem por AP da sonda.

## Dados
- **Treino:** `processed_v3_plus` = `processed_v3` (172 foldable, 277 erros reais) + pool público
  multi-resolução (de-confounding) + erros/reflow sintéticos casados. Splits agrupados por ticket, **0
  vazamento**; teste fisicamente trancado (`siamese.protocol`) durante toda a seleção.

## Reprodução
```bash
python scripts/run_experiment.py --config configs/plus_L_reg4.yaml --processed data/processed_v3_plus
python scripts/domain_slice_eval.py --config configs/plus_L_reg4.yaml --subset v3test        # por domínio
python scripts/foldable_operating_point.py --config configs/plus_L_reg4.yaml                  # ponto foldable + IC
```

**Docs:** `RELATORIO_FINAL_PROCESSED_V3.md` (baseline) · `RELATORIO_FOLDABLE.md` (foldable) ·
`SPEC_COLETA_FOLDABLE.md` (o que levantaria o teto) · `ROADMAP.md` (histórico completo).
