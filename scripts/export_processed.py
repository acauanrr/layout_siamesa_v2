#!/usr/bin/env python
"""Exporta data/processed/ — o DATASET CATEGORIZADO (reais + sinteticos) para COMPARTILHAR.

Benchmark comum para comparacao JUSTA entre modelos: mesmo split (train/val/test) e mesmas
imagens sinteticas que a rede siamesa usou. Estrutura ImageFolder-ready (split -> fonte ->
categoria), legivel direto por torchvision/Keras, sem parser:

  data/processed/
    train/real/<categoria>/        5 classes: clean + 4 categorias de erro
    train/synthetic/<categoria>/   4 classes sinteticas (anti-confound; identicas as do treino)
    val/real/<categoria>/          5 classes (selecao de modelo)
    test/real/<categoria>/         5 classes (benchmark de comparacao)
    manifest.csv                   indice de TODOS os arquivos + proveniencia
    DATASET_CARD.md                taxonomia, contagens, split, uso, reprodutibilidade, ressalvas

Determinismo: usa os MESMOS CSVs de data/splits/ e o MESMO seed/parametros sinteticos do
treino (n_errors=1, tipos com categoria real) -> as imagens sinteticas exportadas sao
IDENTICAS as que o modelo treinou (verificado contra artifacts/embeddings/train_synth.npz).

Uso:
    python scripts/build_splits.py --input data/input --out data/splits   # 1o (gera os CSVs)
    python scripts/export_processed.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from siamese.config import Config
from siamese.features import read_manifest
from siamese.backbone import load_image
from siamese.synthetic import inject, SYNTH_TO_CATEGORY, MULTICLASS_SYNTH_TYPES
from siamese.manifest import CATEGORY_TO_ID, CATEGORIES, ID_TO_CATEGORY

MANIFEST_COLS = ["arquivo", "split", "source", "category", "category_id", "label",
                 "tipos_erro", "parent", "origem", "orig_source", "kind"]

# descricoes curtas das categorias (para o dataset card)
CAT_DESC = {
    "clean": "tela sem erro de layout (classe negativa)",
    "black_bars": "regioes pretas (pillarbox/letterbox, tela girada, fit incorreto)",
    "disordered_layout": "elementos desalinhados/quebrados, espacos entre elementos",
    "distortion": "elementos visualmente distorcidos/esticados",
    "empty_space": "regiao grande vazia com o fundo da tela",
    "orientation": "aspect ratio/orientacao incorreta ao girar",
    "overlay": "elementos sobrepostos",
}


def _reset_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def _unique(dst_dir: Path, base: str) -> str:
    if not (dst_dir / base).exists():
        return base
    stem, suf = Path(base).stem, Path(base).suffix
    i = 1
    while (dst_dir / f"{stem}_{i}{suf}").exists():
        i += 1
    return f"{stem}_{i}{suf}"


def export_real(rows: list[dict], split_root: Path, split: str) -> list[dict]:
    """Copia as imagens REAIS para split_root/real/<categoria>/, preservando o nome."""
    base = split_root / "real"
    _reset_dir(base)
    man = []
    for r in rows:
        src = Path(r["path"])
        cat = r.get("category") or ("clean" if int(r["label"]) == 0 else "uncategorized")
        dst_dir = base / cat
        dst_dir.mkdir(parents=True, exist_ok=True)
        name = _unique(dst_dir, src.name)
        shutil.copy2(src, dst_dir / name)
        man.append({
            "arquivo": f"{split}/real/{cat}/{name}",
            "split": split, "source": "real", "category": cat,
            "category_id": CATEGORY_TO_ID.get(cat, ""),
            "label": r["label"], "tipos_erro": "", "parent": "",
            "origem": str(src), "orig_source": r.get("source", ""), "kind": r.get("kind", ""),
        })
    return man


def export_synthetic(clean_rows: list[dict], split_root: Path, split: str, *,
                     n_variants: int, max_errors: int, seed: int, multiclass: bool) -> list[dict]:
    """Regenera os erros SINTETICOS (identicos aos do treino) em split_root/synthetic/<categoria>/.

    multiclass=True (padrao): 1 erro por imagem (n_errors=1), restrito aos tipos com categoria
    real (MULTICLASS_SYNTH_TYPES); a categoria do arquivo = SYNTH_TO_CATEGORY[tipo].
    """
    base = split_root / "synthetic"
    _reset_dir(base)
    rng = random.Random(seed)            # MESMO seed do treino -> mesmas corrupcoes/ordem
    pool = MULTICLASS_SYNTH_TYPES if multiclass else None
    n_err = 1 if multiclass else max_errors
    man = []
    for i, r in enumerate(clean_rows):
        img = load_image(r["path"])
        stem = Path(r["path"]).stem
        for v in range(n_variants):
            corr, types = inject(img, rng, n_errors=n_err, types=pool)
            tstr = "+".join(types)
            primary = types[0] if types else ""
            cat = (SYNTH_TO_CATEGORY.get(primary) or "uncategorized") if multiclass else "synthetic_error"
            dst_dir = base / cat
            dst_dir.mkdir(parents=True, exist_ok=True)
            name = _unique(dst_dir, f"{stem}__{tstr}__v{v}.png")
            corr.save(dst_dir / name)
            man.append({
                "arquivo": f"{split}/synthetic/{cat}/{name}",
                "split": split, "source": "synthetic", "category": cat,
                "category_id": CATEGORY_TO_ID.get(cat, ""),
                "label": "1", "tipos_erro": tstr, "parent": str(i),
                "origem": str(Path(r["path"])), "orig_source": "synthetic", "kind": "synthetic",
            })
    return man


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)


def _verify_against_training(synth_man: list[dict], emb_dir: Path) -> str:
    npz = emb_dir / "train_synth.npz"
    if not npz.exists():
        return f"  (sem {npz} para verificar — rode make_synthetic.py p/ comparar)"
    import numpy as np
    applied = list(np.load(npz, allow_pickle=True)["applied"])
    gen = [m["tipos_erro"] for m in synth_man]
    if len(applied) != len(gen):
        return f"  ATENCAO: {len(gen)} sinteticos gerados != {len(applied)} em train_synth.npz"
    n_ok = sum(a == b for a, b in zip(applied, gen))
    if n_ok == len(gen):
        return f"  OK: {n_ok}/{len(gen)} sinteticos batem (tipos/ordem) com train_synth.npz -> sao os do treino"
    return f"  ATENCAO: so {n_ok}/{len(gen)} batem com train_synth.npz (config/seed mudaram?)"


def _dir_size_mb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6


def _dataset_card(out: Path, all_man: list[dict], cfg: Config, multiclass: bool, verify_msg: str) -> None:
    # contagens split x categoria (real) e synthetic (train)
    real = defaultdict(lambda: Counter())
    synth = Counter()
    for m in all_man:
        if m["source"] == "real":
            real[m["split"]][m["category"]] += 1
        else:
            synth[m["category"]] += 1
    splits = ["train", "val", "test"]
    cats = [c for c in CATEGORIES]

    def real_table() -> str:
        head = "| categoria | " + " | ".join(splits) + " | total |\n|---|" + "---|" * (len(splits) + 1) + "\n"
        body = ""
        for c in cats:
            vals = [real[s].get(c, 0) for s in splits]
            body += f"| `{c}` | " + " | ".join(str(v) for v in vals) + f" | {sum(vals)} |\n"
        tot = [sum(real[s].values()) for s in splits]
        body += f"| **TOTAL** | " + " | ".join(f"**{v}**" for v in tot) + f" | **{sum(tot)}** |\n"
        return head + body

    def overview_table() -> str:
        """Tabela COMPLETA do dataset por categoria: reais por split + sinteticos + totais."""
        head = ("| categoria | id | train (real) | train (synth) | val (real) | test (real) | "
                "**total reais** | **total (reais+synth)** |\n"
                "|---|---|---|---|---|---|---|---|\n")
        body = ""
        tot = [0, 0, 0, 0, 0, 0]
        for c in cats:
            tr_r = real["train"].get(c, 0); tr_s = synth.get(c, 0)
            v = real["val"].get(c, 0); t = real["test"].get(c, 0)
            real_tot = tr_r + v + t          # so reais (dataset base)
            all_tot = real_tot + tr_s        # reais + sinteticos
            body += (f"| `{c}` | {CATEGORY_TO_ID[c]} | {tr_r} | {tr_s} | {v} | {t} | "
                     f"**{real_tot}** | **{all_tot}** |\n")
            for i, x in enumerate((tr_r, tr_s, v, t, real_tot, all_tot)):
                tot[i] += x
        body += (f"| **TOTAL** | — | **{tot[0]}** | **{tot[1]}** | **{tot[2]}** | **{tot[3]}** | "
                 f"**{tot[4]}** | **{tot[5]}** |\n")
        return head + body

    synth_rows = "".join(f"| `{c}` | {synth[c]} |\n" for c in cats if synth.get(c))
    tax_rows = "".join(f"| {CATEGORY_TO_ID[c]} | `{c}` | {CAT_DESC.get(c,'')} |\n" for c in cats)

    card = f"""# Dataset — UI layout errors (categorizado) · benchmark compartilhado

Dataset para **comparacao justa entre modelos de deep learning**: todos treinam/testam no
**mesmo split** e (opcionalmente) com as **mesmas imagens sinteticas** anti-confound.

## Visao geral do dataset (reais + sinteticos)

{overview_table()}
> `total reais` = o dataset base de imagens reais; `total (reais+synth)` inclui a augmentacao
> sintetica anti-confound (existe SO no train). val/test sao 100% reais (benchmark honesto). O
> treino ve `train (real) + train (synth)`; `id` = rotulo canonico (`siamese.manifest`).

## Taxonomia (5 classes)

| id | classe (pasta) | descricao |
|---|---|---|
{tax_rows}
> O `id` acima e o mapeamento CANONICO do projeto (`siamese.manifest.CATEGORY_TO_ID`). O
> torchvision `ImageFolder` atribui ids em ordem alfabetica das pastas — use o `manifest.csv`
> (coluna `category_id`) ou este mapa para alinhar rotulos entre equipes.

## Estrutura

```
data/processed/
  train/real/<categoria>/        # 5 classes (clean + 4 erros) — imagens REAIS
  train/synthetic/<categoria>/   # 4 classes — erros SINTETICOS (anti-confound) injetados nas limpas de treino
  val/real/<categoria>/          # 5 classes — selecao de modelo
  test/real/<categoria>/         # 5 classes — BENCHMARK de comparacao (held-out)
  manifest.csv                   # indice de todos os arquivos + proveniencia
  DATASET_CARD.md                # este arquivo
```

## Contagens — imagens REAIS por split x categoria

{real_table()}
## Imagens SINTETICAS (somente train) por categoria

| categoria | n |
|---|---|
{synth_rows}
> Sinteticos: {cfg.synthetic.n_variants} variantes por imagem limpa de treino, **1 erro por
> imagem** (rotulo de categoria nao-ambiguo), tipos com correspondente real
> ({', '.join(MULTICLASS_SYNTH_TYPES)}); seed `{cfg.synthetic.seed}`. {verify_msg.strip()}

## Como o split foi feito (sem vazamento)

- **Erros — agrupados por ticket** (`IKSWW-\\d+`): todas as imagens de um mesmo bug vao para o
  MESMO split, nunca cruzam train/val/test.
- **Telas limpas — agrupadas por SESSAO de captura (timestamp) + near-duplicate perceptual
  (dHash)** ANTES do split. As capturas limpas sao sequenciais do mesmo device/sessao
  (quase-duplicatas; similaridade DINO ~0.99); cada componente de sessao e' atomico e fica num
  UNICO split. **Correcao da Fase 0 (jun/2026):** o agrupamento so-por-ticket NAO cobria as
  telas limpas e elas vazavam entre os splits — agora nao mais.
- **Estratificado por categoria**: cada classe (inclusive as raras `orientation`/`distortion`)
  aparece em train/val/test. Fracoes val/test = {cfg.val_frac:.2f}/{cfg.test_frac:.2f}; seed `{cfg.seed}`.
- **0 vazamento de grupo entre splits**, verificado em `build_splits.py` e travado pelos testes
  em `tests/test_split_isolation.py` (inclui a checagem da arvore materializada deste diretorio).

## Proveniencia

- Reais: `data/input/no_erros/` (limpas) + `data/input/errors_dataset/<categoria>/` (erros).
- **Limpeza (marcacoes):** imagens com **marcacoes vermelhas/anotacoes humanas** foram REMOVIDAS
  antes do split (evita o modelo aprender a anotacao). Auditoria em `artifacts/reports/red_marks_*`.
- **Deduplicacao (conteudo):** imagens de **conteudo identico** (verificacao por hash md5) foram
  removidas — ex.: a mesma imagem catalogada em 2 categorias e copias literais `_(1)`. O dataset
  real e' **541 imagens UNICAS**, **0 duplicatas**; **0 vazamento de grupo entre splits** (limpas
  agrupadas por sessao+dHash, erros por ticket — ver "Como o split foi feito").
- Sinteticos: gerados das proprias telas limpas de treino (mesma resolucao/device) — quebram o
  confound de resolucao (ver ressalvas).

## Como usar (ImageFolder)

```python
from torchvision.datasets import ImageFolder
# multi-classe (5 classes), apenas dados reais:
train = ImageFolder("data/processed/train/real")
test  = ImageFolder("data/processed/test/real")
# real + sintetico no treino: concatene train/real e train/synthetic (ou use o manifest.csv).
# binario (erro vs sem-erro): trate 'clean' como negativo e as 6 demais como positivo (coluna 'label').
```

`manifest.csv` lista TODO arquivo com: `arquivo, split, source(real|synthetic), category,
category_id, label(0/1), tipos_erro, parent, origem, orig_source, kind`.

## Reprodutibilidade

```bash
python scripts/build_splits.py --input data/input --out data/splits   # mesmo seed -> mesmo split
python scripts/export_processed.py --config configs/default.yaml       # regenera este diretorio
```
Sinteticos de **val/test** (sonda livre de confound) nao sao exportados aqui; regenere com
`make_synthetic.py` (seeds {cfg.synthetic.seed}+100 / +200) se precisar reproduzir aquela metrica.

## Ressalvas (importantes para comparar de forma honesta)

- **⚠️ Confound de resolucao (NAO reporte acuracia/AUROC global como skill):** as telas `clean`
  sao todas de **um unico device (2076x2152)**; os erros sao heterogeneos em resolucao. Um
  classificador trivial **so de resolucao** separa clean/erro com **~AUROC 0.98 sem olhar o
  layout**. Logo, qualquer metrica GLOBAL neste benchmark mede majoritariamente o device, nao a
  deteccao de erro. **Para comparar modelos de forma justa**, use: (a) **deteccao sintetica livre
  de confound** (erros injetados nas proprias limpas, mesma resolucao); (b) o **subconjunto
  controlado** (mesmo form-factor/orientacao/kind); e/ou (c) compare contra o **baseline de
  resolucao/padding** — um modelo so tem valor se SUPERAR esses baselines de confound.
- **Teste e' exploratorio/confundido:** as 41 limpas de teste sao todas 2076x2152 e provem de
  poucas sessoes de captura. Trate alegacoes de alta precisao com **IC95%** (amostras pequenas,
  ex. 17 TP / 1 FP, sao estatisticamente insuficientes).
- **Classes raras com suporte baixo:** `orientation` (~7) e `distortion` (~13) no total — F1 por
  classe nessas e' instavel; reporte com cautela e prefira intervalos de confianca.
- **Teto de separabilidade das categorias:** em features visuais congeladas (DINOv2), as 6
  categorias de erro tem F1-macro ~0.2 (categorias semanticamente sobrepostas + rotulo
  single-label para erros que coocorrem + classes raras). E um problema dificil — esperado.
"""
    out.write_text(card, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    cfg = Config.load(args.config)
    multiclass = cfg.train.multiclass
    splits = Path(cfg.paths.splits_dir)

    train_rows = read_manifest(splits / "train.csv")
    val_rows = read_manifest(splits / "val.csv")
    test_rows = read_manifest(splits / "test.csv")
    clean_train = [r for r in train_rows if int(r["label"]) == 0]

    print(f"Exportando dataset categorizado para {args.out}/ (multiclass={multiclass}) ...")
    man = []
    man += export_real(train_rows, args.out / "train", "train")
    synth = export_synthetic(clean_train, args.out / "train", "train",
                             n_variants=cfg.synthetic.n_variants,
                             max_errors=cfg.synthetic.max_errors_per_image,
                             seed=cfg.synthetic.seed, multiclass=multiclass)
    man += synth
    man += export_real(val_rows, args.out / "val", "val")
    man += export_real(test_rows, args.out / "test", "test")

    _write_manifest(args.out / "manifest.csv", man)
    verify_msg = (f"{len(synth)} sinteticos materializados em processed/train/synthetic/ — "
                  "FONTE da verdade; embedados por extract_features.py (train_synth.npz).")
    _dataset_card(args.out / "DATASET_CARD.md", man, cfg, multiclass, verify_msg)

    # resumo
    def by(split, source):
        return Counter(m["category"] for m in man if m["split"] == split and m["source"] == source)
    print("\nResumo (imagens fisicas):")
    for split in ("train", "val", "test"):
        rc = by(split, "real")
        print(f"  {split}/real/        {sum(rc.values()):4d}  {dict(sorted(rc.items()))}")
    sc = by("train", "synthetic")
    print(f"  train/synthetic/  {sum(sc.values()):4d}  {dict(sorted(sc.items()))}")
    print(f"\n  TOTAL = {len(man)} arquivos | disco: {_dir_size_mb(args.out):.0f} MB")
    print(f"\n  {verify_msg}")
    print(f"\nProximo passo: python scripts/extract_features.py --processed {args.out} "
          "--out artifacts/embeddings --use-patch-stats --preprocess pad")
    print(f"Manifest: {args.out}/manifest.csv  |  Card: {args.out}/DATASET_CARD.md")


if __name__ == "__main__":
    main()
