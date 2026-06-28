#!/usr/bin/env python
"""Fase 2 (download-only): baixa telas LIMPAS diversas de datasets publicos de UI e as
materializa em data/clean_extra/ com resolucoes NATIVAS variadas, p/ QUEBRAR o confound de
resolucao do processed_v3 (todas as 172 limpas reais sao 2076x2152, de 1 device/1 dia).

Fontes verificadas (docs/ROADMAP.md §2.3) — imagens reais via HF `datasets` (streaming):
  screenspot_v2  HongxinLi/ScreenSpot_v2    Apache-2.0  phone/tablet/desktop, 2 orientacoes
  groundui       agent-studio/GroundUI-18K  MIT         web/desktop/mobile (mobile = portrait)
  (screenspot_pro nao streama como parquet — desabilitado por padrao; ver --sources)

O ALVO (distribuicao dos ERROS reais) e' PORTRAIT-dominante + near-square: use --aspect-max p/
puxar portrait/near-square (mobile/unfold) e --aspect-min p/ landscape. --append acumula varias
passadas num so pool (dedup perceptual cruza as passadas via coluna phash do manifesto).

Principios: STREAMING + cap por fonte; NAO redimensiona (mantem nativo); dedup dHash; descarta
thumbnails (lado < --min-side); cada imagem = 1 grupo (anti-vazamento no split posterior).

Saida: data/clean_extra/<fonte>/*.png + data/clean_extra/labels_extra.csv
       (path, source, w, h, aspect, group, phash).

Uso (pool balanceado em 2 passadas):
    python scripts/fetch_clean_extra.py --sources groundui --aspect-max 1.15 --per-source 250
    python scripts/fetch_clean_extra.py --sources screenspot_v2 groundui --aspect-min 1.15 \
        --per-source 150 --append
"""
from __future__ import annotations

import argparse
import csv
import io
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# (chave -> nome HF, split, coluna-da-imagem preferida)
SOURCES = {
    "screenspot_v2":  ("HongxinLi/ScreenSpot_v2", "test", "image"),
    "groundui":       ("agent-studio/GroundUI-18K", "train", "image"),
    "screenspot_pro": ("likaixin/ScreenSpot-Pro", "train", "image"),  # nao streama (parquet ausente)
}
DEFAULT_SOURCES = ["screenspot_v2", "groundui"]
PHASH_MAX_DIST = 6


def dhash(img: Image.Image, hash_size: int = 8) -> int:
    """dHash perceptual de 64 bits (gradiente horizontal em 9x8 cinza). Sem dependencia extra."""
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _as_pil(v):
    if isinstance(v, Image.Image):
        return v.convert("RGB")
    if isinstance(v, dict) and v.get("bytes"):
        try:
            return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
        except Exception:
            return None
    return None


def _pick_image(example, col):
    v = _as_pil(example.get(col))
    if v is not None:
        return v
    for val in example.values():          # auto-deteccao se a coluna mudou de nome
        v = _as_pil(val)
        if v is not None:
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/clean_extra"))
    ap.add_argument("--per-source", type=int, default=200)
    ap.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES)
    ap.add_argument("--min-side", type=int, default=800)
    ap.add_argument("--aspect-min", type=float, default=0.0)   # W/H >= isto
    ap.add_argument("--aspect-max", type=float, default=99.0)  # W/H <= isto
    ap.add_argument("--scan-limit", type=int, default=40000)
    ap.add_argument("--append", action="store_true", help="acumula no pool/manifesto existente")
    args = ap.parse_args()

    from datasets import load_dataset

    args.out.mkdir(parents=True, exist_ok=True)
    man = args.out / "labels_extra.csv"
    cols = ["path", "source", "w", "h", "aspect", "group", "phash"]

    rows: list[dict] = []
    seen: list[int] = []
    if args.append and man.exists():
        with open(man, newline="") as f:
            rows = list(csv.DictReader(f))
        seen = [int(r["phash"]) for r in rows if r.get("phash")]
        print(f"[append] pool existente: {len(rows)} imagens")

    def is_dup(h: int) -> bool:
        return any(_hamming(h, s) <= PHASH_MAX_DIST for s in seen)

    for key in args.sources:
        if key not in SOURCES:
            print(f"  [skip] fonte desconhecida: {key}")
            continue
        name, split, col = SOURCES[key]
        (args.out / key).mkdir(parents=True, exist_ok=True)
        print(f"\n== {key} ({name}) aspect[{args.aspect_min},{args.aspect_max}] alvo {args.per_source} ==")
        try:
            ds = load_dataset(name, split=split, streaming=True)
        except Exception as e:
            print(f"  [erro] load_dataset falhou ({e}); pulando {key}")
            continue
        kept = read = 0
        for read, ex in enumerate(ds, 1):
            if kept >= args.per_source or read > args.scan_limit:
                break
            img = _pick_image(ex, col)
            if img is None:
                continue
            w, h = img.size
            if min(w, h) < args.min_side:
                continue
            asp = w / h
            if not (args.aspect_min <= asp <= args.aspect_max):
                continue
            ph = dhash(img)
            if is_dup(ph):
                continue
            seen.append(ph)
            fn = f"{key}_{read:06d}_{w}x{h}.png"
            img.save(args.out / key / fn)
            rows.append({"path": f"clean_extra/{key}/{fn}", "source": key, "w": w, "h": h,
                         "aspect": round(asp, 3), "group": f"extra:{key}:{read:06d}", "phash": ph})
            kept += 1
            if kept % 50 == 0:
                print(f"  {kept}/{args.per_source} (lidos {read}) ...")
        print(f"  {key}: +{kept} (lidos {read})")

    with open(man, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nTOTAL pool: {len(rows)} limpas -> {man}")
    if rows:
        ar = Counter()
        for r in rows:
            a = float(r["aspect"])
            ar["<0.5" if a < 0.5 else "0.5-0.8" if a < 0.8 else
               "0.8-1.1" if a < 1.1 else "1.1-1.5" if a < 1.5 else ">1.5"] += 1
        print("cobertura por aspecto (W/H):", dict(sorted(ar.items())))
        print("por fonte:", dict(Counter(r["source"] for r in rows)))


if __name__ == "__main__":
    main()
