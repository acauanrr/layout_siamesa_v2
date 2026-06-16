#!/usr/bin/env python
"""Salva em disco as IMAGENS dos erros sinteticos (as mesmas que o modelo treinou).

Reproduz exatamente o que `make_synthetic.py` injeta (mesmo seed, mesmas imagens limpas de
treino), mas em vez de so cachear embeddings, salva os PNGs para inspecao manual:

  artifacts/synthetic_images/
    by_type/<tipo>/<stem>__<tipos>__v<i>.png     imagens corrompidas, agrupadas por tipo
    pairs/<stem>__<tipos>__v<i>.png              original | corrompida lado a lado
    contact_<tipo>.png                            mosaico de exemplos por tipo
    manifest.csv                                  origem, tipos e caminhos

Uso:
    python scripts/dump_synthetic.py --config configs/default.yaml [--source all]
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image

from siamese.config import Config
from siamese.features import read_manifest
from siamese.backbone import load_image
from siamese.synthetic import inject, ERROR_TYPES


def side_by_side(orig: Image.Image, corr: Image.Image, h: int = 420) -> Image.Image:
    def rz(im):
        w = int(im.width * h / im.height)
        return im.resize((w, h))
    a, b = rz(orig), rz(corr)
    canvas = Image.new("RGB", (a.width + b.width + 8, h), (255, 255, 255))
    canvas.paste(a, (0, 0)); canvas.paste(b, (a.width + 8, 0))
    return canvas


def contact_sheet(images: list[Image.Image], cols: int = 4, cell: int = 240) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (245, 245, 245))
    for i, im in enumerate(images):
        t = im.copy(); t.thumbnail((cell - 6, cell - 6))
        x = (i % cols) * cell + 3; y = (i // cols) * cell + 3
        sheet.paste(t, (x, y))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--source", choices=["train", "all"], default="train",
                    help="train = mesmas imagens que o modelo viu; all = todas as limpas")
    ap.add_argument("--out", type=Path, default=Path("artifacts/synthetic_images"))
    args = ap.parse_args()
    cfg = Config.load(args.config)

    if args.source == "train":
        rows = [r for r in read_manifest(Path(cfg.paths.splits_dir) / "train.csv")
                if int(r["label"]) == 0]
    else:
        d = Path(cfg.paths.input_dir) / "no_erros"
        rows = [{"path": str(p)} for p in sorted(d.glob("*.png"))]

    rng = random.Random(cfg.synthetic.seed)  # MESMO seed do treino -> mesmas corrupcoes
    out = args.out
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    for t in ERROR_TYPES:
        (out / "by_type" / t).mkdir(parents=True, exist_ok=True)

    manifest = []
    by_type_examples: dict[str, list] = {t: [] for t in ERROR_TYPES}
    n = 0
    for r in rows:
        orig = load_image(r["path"])
        stem = Path(r["path"]).stem
        for i in range(cfg.synthetic.n_variants):
            corr, types = inject(orig, rng, n_errors=cfg.synthetic.max_errors_per_image)
            tstr = "+".join(types)
            primary = types[0]
            fname = f"{stem}__{tstr}__v{i}.png"
            corr.save(out / "by_type" / primary / fname)
            side_by_side(orig, corr).save(out / "pairs" / fname)
            if len(by_type_examples[primary]) < 12:
                by_type_examples[primary].append(corr)
            manifest.append({"origem": stem, "tipos": tstr, "variante": i,
                             "arquivo": f"by_type/{primary}/{fname}",
                             "par": f"pairs/{fname}"})
            n += 1

    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["origem", "tipos", "variante", "arquivo", "par"])
        w.writeheader(); w.writerows(manifest)

    for t, ims in by_type_examples.items():
        if ims:
            contact_sheet(ims).save(out / f"contact_{t}.png")

    print(f"Salvos {n} erros sinteticos de {len(rows)} imagens limpas ({args.source}).")
    print(f"  {out}/by_type/<tipo>/   (corrompidas por tipo)")
    print(f"  {out}/pairs/            (original | corrompida)")
    print(f"  {out}/contact_<tipo>.png e manifest.csv")
    # distribuicao de tipos primarios
    from collections import Counter
    c = Counter(m["tipos"].split("+")[0] for m in manifest)
    print("  distribuicao (tipo primario):", dict(c))


if __name__ == "__main__":
    main()
