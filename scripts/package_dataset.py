#!/usr/bin/env python
"""Empacota uma copia LIMPA de um dataset (padrao plano + labels.csv) para COMPARTILHAR.

Inclui SOMENTE o que esta no labels.csv (exclui _clean_pool e qualquer cruft) + labels.csv +
DATASET_CARD.md. Valida labels.csv x disco, gera SHA256SUMS.txt e, opcionalmente, uma visao
torchvision ImageFolder (<split>/<categoria>/) e um .tar.gz.

Uso:
    python scripts/package_dataset.py --root data/processed_v3 --out dist/processed_v3
    python scripts/package_dataset.py --root data/dataset_indt --out dist/dataset_indt --imagefolder --zip
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="dataset de origem (com labels.csv)")
    ap.add_argument("--out", type=Path, required=True, help="diretorio do pacote de saida")
    ap.add_argument("--imagefolder", action="store_true",
                    help="materializa tambem out/imagefolder/<split>/<categoria>/ (torchvision)")
    ap.add_argument("--zip", action="store_true", help="gera <out>.tar.gz")
    args = ap.parse_args()

    root, out = args.root, args.out
    lp = root / "labels.csv"
    if not lp.exists():
        sys.exit(f"{lp} ausente — nada a empacotar.")
    rows = list(csv.DictReader(open(lp, newline="")))
    man = {r["path"] for r in rows}

    # --- validacao ---
    problems = []
    for r in rows:
        if not (root / r["path"]).exists():
            problems.append(f"manifesto aponta p/ arquivo faltando: {r['path']}")
    disk = []
    for sp in SPLITS:
        d = root / sp
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG and "Zone.Identifier" not in p.name:
                rel = str(p.relative_to(root))
                disk.append(rel)
                if rel not in man:
                    problems.append(f"imagem em split fora do manifesto: {rel}")
    zone = [p for p in root.rglob("*") if "Zone.Identifier" in p.name]
    if zone:
        problems.append(f"{len(zone)} arquivos Zone.Identifier presentes")
    if problems:
        print("VALIDACAO FALHOU — pacote NAO gerado:")
        for x in problems[:25]:
            print("  -", x)
        sys.exit(1)
    print(f"Validacao OK: {len(rows)} no manifesto = {len(disk)} imagens em splits; 0 cruft.")

    # --- copia limpa (so o que esta no manifesto) ---
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    sums = []
    for r in rows:
        dst = out / r["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / r["path"], dst)
        sums.append((sha256(dst), r["path"]))
    shutil.copy2(lp, out / "labels.csv")
    sums.append((sha256(out / "labels.csv"), "labels.csv"))
    card = root / "DATASET_CARD.md"
    if card.exists():
        shutil.copy2(card, out / "DATASET_CARD.md")
    else:
        print("  [aviso] DATASET_CARD.md ausente na origem — pacote sem card.")
    (out / "SHA256SUMS.txt").write_text(
        "\n".join(f"{h}  {p}" for h, p in sorted(sums, key=lambda x: x[1])) + "\n")

    # --- visao ImageFolder (opcional) ---
    if args.imagefolder:
        n = 0
        for r in rows:
            d = out / "imagefolder" / r["split"] / r["category"]
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / r["path"], d / Path(r["path"]).name)
            n += 1
        print(f"  ImageFolder: out/imagefolder/<split>/<categoria>/ ({n} arquivos)")

    # --- resumo ---
    by = Counter((r["split"], r["source"]) for r in rows)
    print(f"\nPacote em {out}/")
    for k in sorted(by):
        print(f"  {k[0]}/{k[1]}: {by[k]}")
    extras = "labels.csv, SHA256SUMS.txt" + (", DATASET_CARD.md" if card.exists() else "")
    print(f"  + {extras}")
    if (root / "_clean_pool").exists():
        print("  (nota: _clean_pool/ da origem foi EXCLUIDO do pacote, como esperado)")

    # --- zip (opcional) ---
    if args.zip:
        tarp = out.parent / (out.name + ".tar.gz")
        with tarfile.open(tarp, "w:gz") as t:
            t.add(out, arcname=out.name)
        print(f"  arquivo: {tarp} ({tarp.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
