#!/usr/bin/env python
"""Exporta para data/processed/ os ARQUIVOS de imagem usados em treino e teste.

Estrutura gerada:

  data/processed/
    train/
      real/        imagens REAIS de treino (limpas label0 + erros reais label1)
      synthetic/   erros SINTETICOS de treino (limpas de treino corrompidas)
      manifest.csv arquivo, split, fonte, classe, label, tipos_erro, parent, origem
    test/
      real/        imagens REAIS de teste
      manifest.csv

Os sinteticos sao reproduzidos com o MESMO seed e a MESMA logica de `make_synthetic`
(`extract_synthetic`): mesmas imagens limpas de treino, mesma ordem, mesmo
`random.Random(cfg.synthetic.seed)`, mesmo `inject`. Logo sao IDENTICOS aos que o modelo
treinou (cujos embeddings estao em artifacts/embeddings/train_synth.npz). So usa PIL — nao
precisa de GPU nem do backbone. Ao final, valida os tipos gerados contra train_synth.npz.

Uso:
    python scripts/export_processed.py --config configs/default.yaml
    python scripts/export_processed.py --out data/processed
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from siamese.config import Config
from siamese.features import read_manifest
from siamese.backbone import load_image
from siamese.synthetic import inject

MANIFEST_COLS = ["arquivo", "split", "fonte", "classe", "label",
                 "tipos_erro", "parent", "origem", "source", "kind"]


def _reset_dir(d: Path) -> None:
    """Limpa o destino p/ a exportacao ser idempotente (re-rodar nao acumula lixo)."""
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def _unique(dst_dir: Path, base: str) -> str:
    """Garante nome unico no destino (raro, mas no_erros vs with_errors poderiam colidir)."""
    if not (dst_dir / base).exists():
        return base
    stem, suf = Path(base).stem, Path(base).suffix
    i = 1
    while (dst_dir / f"{stem}_{i}{suf}").exists():
        i += 1
    return f"{stem}_{i}{suf}"


def export_real(rows: list[dict], dst_dir: Path, split: str) -> list[dict]:
    """Copia as imagens reais (caminho em r['path']) para dst_dir, preservando o nome."""
    _reset_dir(dst_dir)
    man = []
    for r in rows:
        src = Path(r["path"])
        name = _unique(dst_dir, src.name)
        shutil.copy2(src, dst_dir / name)
        man.append({
            "arquivo": f"{dst_dir.name}/{name}",
            "split": split,
            "fonte": "real",
            "classe": "erro" if int(r["label"]) == 1 else "limpo",
            "label": r["label"],
            "tipos_erro": "",
            "parent": "",
            "origem": str(src),
            "source": r.get("source", ""),
            "kind": r.get("kind", ""),
        })
    return man


def export_synthetic(clean_rows: list[dict], dst_dir: Path, *, n_variants: int,
                     max_errors: int, seed: int) -> list[dict]:
    """Regenera os erros sinteticos de treino, identicos aos de extract_synthetic."""
    _reset_dir(dst_dir)
    rng = random.Random(seed)  # MESMO seed do treino -> mesmas corrupcoes, mesma ordem
    man = []
    for i, r in enumerate(clean_rows):
        img = load_image(r["path"])
        stem = Path(r["path"]).stem
        for v in range(n_variants):
            corr, types = inject(img, rng, n_errors=max_errors)
            tstr = "+".join(types)
            name = _unique(dst_dir, f"{stem}__{tstr}__v{v}.png")
            corr.save(dst_dir / name)
            man.append({
                "arquivo": f"{dst_dir.name}/{name}",
                "split": "train",
                "fonte": "synthetic",
                "classe": "erro",
                "label": "1",
                "tipos_erro": tstr,
                "parent": str(i),          # indice da limpa-mae (== parent em train_synth.npz)
                "origem": str(Path(r["path"])),
                "source": "synthetic",
                "kind": "synthetic",
            })
    return man


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)


def _verify_against_training(synth_man: list[dict], emb_dir: Path) -> str:
    """Confere que a sequencia de tipos gerada bate com train_synth.npz (prova de fidelidade)."""
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
        return f"  OK: {n_ok}/{len(gen)} sinteticos batem (tipos) com train_synth.npz -> sao os do treino"
    return f"  ATENCAO: so {n_ok}/{len(gen)} batem com train_synth.npz (config/seed mudaram?)"


def _dir_size_mb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    cfg = Config.load(args.config)
    splits = Path(cfg.paths.splits_dir)

    train_rows = read_manifest(splits / "train.csv")
    test_rows = read_manifest(splits / "test.csv")
    clean_train = [r for r in train_rows if int(r["label"]) == 0]

    print(f"Exportando para {args.out}/ ...")
    train_real = export_real(train_rows, args.out / "train" / "real", "train")
    synth = export_synthetic(clean_train, args.out / "train" / "synthetic",
                             n_variants=cfg.synthetic.n_variants,
                             max_errors=cfg.synthetic.max_errors_per_image,
                             seed=cfg.synthetic.seed)
    test_real = export_real(test_rows, args.out / "test" / "real", "test")

    _write_manifest(args.out / "train" / "manifest.csv", train_real + synth)
    _write_manifest(args.out / "test" / "manifest.csv", test_real)

    n_clean = sum(1 for m in train_real if m["classe"] == "limpo")
    n_err = sum(1 for m in train_real if m["classe"] == "erro")
    print("\nResumo:")
    print(f"  train/real/       {len(train_real):4d}  ({n_clean} limpas + {n_err} erros reais)")
    print(f"  train/synthetic/  {len(synth):4d}  (limpas de treino corrompidas)")
    print(f"  test/real/        {len(test_real):4d}  "
          f"({sum(1 for m in test_real if m['classe']=='limpo')} limpas + "
          f"{sum(1 for m in test_real if m['classe']=='erro')} erros reais)")
    print(f"  treino TOTAL = {len(train_real) + len(synth)} imagens "
          f"(reais + sinteticos) | tamanho em disco: {_dir_size_mb(args.out):.0f} MB")
    print("\nVerificacao de fidelidade dos sinteticos:")
    print(_verify_against_training(synth, Path(cfg.paths.emb_dir)))
    print(f"\nManifests: {args.out}/train/manifest.csv  e  {args.out}/test/manifest.csv")


if __name__ == "__main__":
    main()
