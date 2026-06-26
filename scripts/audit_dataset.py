#!/usr/bin/env python
"""AUDITORIA (somente leitura) de um dataset no padrao novo (pastas planas):

    <root>/{train,val,test}/{real,synthetic}/*.png

Reporta, sem modificar nada:
  1. Inventario por (split, fonte): imagens, lixo (Zone.Identifier), outros nao-imagem.
  2. Corrupcao: arquivos que o PIL nao consegue abrir/decodificar.
  3. Duplicatas EXATAS (sha256 dos bytes) — intra-split e CRUZANDO splits (vazamento).
  4. Near-duplicates perceptuais (dHash 64b, Hamming <= --near-dist) — foco em cruzamento.
  5. VAZAMENTO por ticket/sessao: mesmo grupo (IKSWW-xxxx / sessao de captura) em >1 split.
  6. Cobertura de rotulo: real via --labels (basename, depois ticket); sintetico via nome.
  7. Balanceamento de classe por split (clean + 4 erros).
  8. Anomalias de nome: dupla extensao, competitor, boundbox (case), espacos, nao-ascii.
  9. Propriedades de imagem: modo (RGBA/P/L), tamanho, aspect ratio, imagens minusculas.

Uso:
    python scripts/audit_dataset.py --root data/processed_v3 --labels data/splits/all.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # nao alertar em telas grandes

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")
SOURCES = ("real", "synthetic")

# tipo sintetico embutido no nome (parent__TIPO__vN.png) -> categoria canonica
SYNTH_TO_CATEGORY = {
    "black_region": "black_bars",
    "empty_space": "empty_space",
    "overlay": "overlay",
    "disorder": "disordered_layout",
    "cropped": None,  # fora das 5 classes
}
CANON = ["clean", "black_bars", "disordered_layout", "empty_space", "overlay"]

RE_TICKET = re.compile(r"(IKSWW[-_]\d+)", re.IGNORECASE)
RE_SHOT_TS = re.compile(r"(\d{8})[_-](\d{6})")


def ticket_or_session(name: str) -> str:
    m = RE_TICKET.search(name)
    if m:
        return m.group(1).upper().replace("_", "-")
    m = RE_SHOT_TS.search(name)
    if m:
        return f"shot:{m.group(1)}_{m.group(2)}"
    return f"file:{Path(name).stem}"


def synth_category(name: str) -> str | None:
    parts = Path(name).stem.split("__")
    if len(parts) >= 2:
        return SYNTH_TO_CATEGORY.get(parts[1].lower(), "UNMAPPED:" + parts[1])
    return None


def dhash(path: Path, size: int = 8) -> int | None:
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((size + 1, size), Image.BILINEAR)
        a = np.asarray(g, dtype=np.int16)
        diff = a[:, 1:] > a[:, :-1]
        bits = 0
        for v in diff.flatten():
            bits = (bits << 1) | int(v)
        return bits
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labels(csv_path: Path):
    """Devolve (by_name, by_ticket): basename->category e ticket->categoria majoritaria."""
    by_name, tick_cats = {}, defaultdict(Counter)
    if not csv_path or not csv_path.exists():
        return by_name, {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            name = Path(r["path"]).name
            cat = (r.get("category") or "").strip()
            if cat:
                by_name[name] = cat
                tick_cats[ticket_or_session(name)][cat] += 1
    by_ticket = {t: c.most_common(1)[0][0] for t, c in tick_cats.items()}
    return by_name, by_ticket


def hr(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/processed_v3"))
    ap.add_argument("--labels", type=Path, default=Path("data/splits/all.csv"))
    ap.add_argument("--near-dist", type=int, default=4, help="Hamming dHash p/ near-dup")
    ap.add_argument("--json", type=Path, default=None, help="grava relatorio JSON")
    args = ap.parse_args()

    by_name, by_ticket = load_labels(args.labels)
    print(f"Rotulos de referencia: {len(by_name)} nomes / {len(by_ticket)} tickets "
          f"({args.labels})" if by_name else f"(sem manifesto de rotulos em {args.labels})")

    # --- varredura ---
    records = []   # dict por imagem
    junk = defaultdict(list)   # (split,source) -> [zone/outros]
    for split in SPLITS:
        for source in SOURCES:
            base = args.root / split / source
            if not base.is_dir():
                continue
            for p in sorted(base.rglob("*")):
                if not p.is_file():
                    continue
                if p.name.endswith(":Zone.Identifier") or "Zone.Identifier" in p.name:
                    junk[(split, source)].append(("zone", p))
                    continue
                if p.suffix.lower() not in IMG_EXTS:
                    junk[(split, source)].append(("other", p))
                    continue
                records.append({"path": p, "split": split, "source": source,
                                "name": p.name, "ticket": ticket_or_session(p.name)})

    if not records:
        sys.exit(f"Nenhuma imagem encontrada em {args.root}/<split>/<source>/")

    # --- propriedades + hashes (uma passada de I/O) ---
    corrupt = []
    for r in records:
        try:
            with Image.open(r["path"]) as im:
                im.load()
                r["mode"], r["size"] = im.mode, im.size
        except Exception as e:
            r["mode"], r["size"] = None, None
            corrupt.append((r["path"], repr(e)))
        r["sha"] = sha256(r["path"])
        r["dhash"] = dhash(r["path"])

    # ---------- 1. INVENTARIO ----------
    hr("1. INVENTARIO (imagens | Zone.Identifier | outros nao-imagem)")
    print(f"  {'split/fonte':22s} {'imgs':>6s} {'zone':>6s} {'outros':>7s}  extensoes")
    tot_img = tot_zone = tot_other = 0
    for split in SPLITS:
        for source in SOURCES:
            imgs = [r for r in records if r["split"] == split and r["source"] == source]
            z = sum(1 for k, _ in junk[(split, source)] if k == "zone")
            o = sum(1 for k, _ in junk[(split, source)] if k == "other")
            if not imgs and not z and not o:
                continue
            exts = Counter(Path(r["name"]).suffix.lower() for r in imgs)
            tot_img += len(imgs); tot_zone += z; tot_other += o
            print(f"  {split+'/'+source:22s} {len(imgs):6d} {z:6d} {o:7d}  {dict(exts)}")
    print(f"  {'TOTAL':22s} {tot_img:6d} {tot_zone:6d} {tot_other:7d}")
    if tot_other:
        print("  outros nao-imagem (amostra):")
        for (s, src), items in junk.items():
            for k, p in items:
                if k == "other":
                    print(f"    {p}")

    # ---------- 2. CORRUPCAO ----------
    hr(f"2. CORRUPCAO ({len(corrupt)} arquivos ilegiveis)")
    for p, e in corrupt[:30]:
        print(f"  {p}  -> {e}")
    if not corrupt:
        print("  nenhum — todas as imagens abrem.")

    # ---------- 3. DUPLICATAS EXATAS ----------
    hr("3. DUPLICATAS EXATAS (sha256 dos bytes)")
    by_sha = defaultdict(list)
    for r in records:
        by_sha[r["sha"]].append(r)
    dups = {h: rs for h, rs in by_sha.items() if len(rs) > 1}
    n_cross = sum(1 for rs in dups.values() if len({x["split"] for x in rs}) > 1)
    print(f"  conjuntos de duplicatas exatas: {len(dups)}  "
          f"(cruzando splits: {n_cross}  | intra-split: {len(dups) - n_cross})")
    extra = sum(len(rs) - 1 for rs in dups.values())
    print(f"  arquivos redundantes (poderiam ser removidos): {extra}")
    shown = 0
    for h, rs in sorted(dups.items(), key=lambda x: -len(x[1])):
        cross = len({x["split"] for x in rs}) > 1
        if shown < 25:
            tag = " <-- CRUZA SPLITS (VAZAMENTO)" if cross else ""
            print(f"  [{len(rs)}x]{tag}")
            for x in rs:
                print(f"      {x['split']}/{x['source']}/{x['name']}")
            shown += 1

    # ---------- 4. NEAR-DUPLICATES ----------
    hr(f"4. NEAR-DUPLICATES perceptuais (dHash Hamming <= {args.near_dist})")
    hs = [(r, r["dhash"]) for r in records if r["dhash"] is not None]
    near_cross, near_intra = [], []
    for i in range(len(hs)):
        ri, hi = hs[i]
        for j in range(i + 1, len(hs)):
            rj, hj = hs[j]
            if ri["sha"] == rj["sha"]:
                continue  # ja contam como exata
            if hamming(hi, hj) <= args.near_dist:
                (near_cross if ri["split"] != rj["split"] else near_intra).append((ri, rj))
    print(f"  pares near-dup: {len(near_cross) + len(near_intra)}  "
          f"(cruzando splits: {len(near_cross)} | intra-split: {len(near_intra)})")
    print("  --- amostra CRUZANDO splits (suspeitos de vazamento) ---")
    for ri, rj in near_cross[:25]:
        print(f"    {ri['split']}:{ri['name']}  ~  {rj['split']}:{rj['name']}")

    # ---------- 5. VAZAMENTO por ticket/sessao ----------
    hr("5. VAZAMENTO por ticket/sessao (mesmo grupo em >1 split)")
    # reais: por ticket; sinteticos: pelo parent (stem antes do '__')
    tk_splits = defaultdict(set)
    tk_files = defaultdict(list)
    for r in records:
        if r["source"] == "real":
            key = r["ticket"]
        else:
            key = "synthparent:" + r["name"].split("__")[0]
        tk_splits[key].add(r["split"])
        tk_files[key].append(r)
    leaks = {k: sorted(s) for k, s in tk_splits.items() if len(s) > 1}
    real_leaks = {k: v for k, v in leaks.items() if not k.startswith("synthparent:")}
    print(f"  grupos REAIS cruzando splits: {len(real_leaks)} (esperado 0)")
    for k in sorted(real_leaks)[:40]:
        files = tk_files[k]
        loc = ", ".join(f"{x['split']}/{x['source']}:{x['name']}" for x in files)
        print(f"    {k} -> {real_leaks[k]}")
    # cruzamento parent-sintetico (train) vs clean real (val/test) — vazamento sutil
    parent_stems = {r["name"].split("__")[0] for r in records if r["source"] == "synthetic"}
    real_stems_by_split = defaultdict(set)
    for r in records:
        if r["source"] == "real":
            real_stems_by_split[r["split"]].add(Path(r["name"]).stem)
    cross_parent = {ps for ps in parent_stems
                    if ps in real_stems_by_split["val"] or ps in real_stems_by_split["test"]}
    print(f"\n  parents de sintetico (train) que tambem sao REAL em val/test: {len(cross_parent)}")
    for ps in sorted(cross_parent)[:20]:
        print(f"    {ps}")

    # ---------- 6. COBERTURA DE ROTULO ----------
    hr("6. COBERTURA DE ROTULO")
    for r in records:
        if r["source"] == "synthetic":
            r["category"] = synth_category(r["name"]) or "UNMAPPED"
            r["label_src"] = "nome-sintetico"
        else:
            nm = r["name"]
            if nm in by_name:
                r["category"], r["label_src"] = by_name[nm], "csv:nome"
            elif r["ticket"] in by_ticket:
                r["category"], r["label_src"] = by_ticket[r["ticket"]], "csv:ticket"
            else:
                r["category"], r["label_src"] = None, "SEM-ROTULO"
    real = [r for r in records if r["source"] == "real"]
    src_counts = Counter(r["label_src"] for r in real)
    unlab = [r for r in real if r["category"] is None]
    print(f"  reais: {len(real)} | com rotulo: {len(real) - len(unlab)} | SEM rotulo: {len(unlab)}")
    print(f"  fonte do rotulo (reais): {dict(src_counts)}")
    if unlab:
        print(f"  --- reais SEM rotulo (amostra de {min(30, len(unlab))}/{len(unlab)}) ---")
        for r in unlab[:30]:
            print(f"    {r['split']}/{r['name']}")
    unmapped = [r for r in records if r["source"] == "synthetic" and r["category"] == "UNMAPPED"]
    if unmapped:
        print(f"  sinteticos sem categoria mapeada (ex.: cropped): {len(unmapped)}")

    # ---------- 7. BALANCEAMENTO ----------
    hr("7. BALANCEAMENTO DE CLASSE por split (reais com rotulo + sinteticos)")
    for source in SOURCES:
        print(f"\n  --- fonte: {source} ---")
        cats = sorted({(r.get("category") or "NONE") for r in records if r["source"] == source})
        print(f"  {'categoria':20s}" + "".join(f"{sp:>8s}" for sp in SPLITS) + f"{'total':>8s}")
        for c in cats:
            row = [sum(1 for r in records if r["source"] == source and r["split"] == sp
                       and (r.get("category") or "NONE") == c) for sp in SPLITS]
            print(f"  {c:20s}" + "".join(f"{v:8d}" for v in row) + f"{sum(row):8d}")

    # ---------- 8. ANOMALIAS DE NOME ----------
    hr("8. ANOMALIAS DE NOME")
    double_ext = [r for r in records if re.search(r"\.(png|jpe?g)\.(png|jpe?g)$", r["name"], re.I)]
    spaces = [r for r in records if " " in r["name"]]
    nonascii = [r for r in records if any(ord(ch) > 127 for ch in r["name"])]
    bb_case = Counter(m.group(0) for r in records
                      if (m := re.search(r"bound[bB]ox", r["name"])))
    comp = [r for r in records if "competitor" in r["name"].lower()]
    print(f"  dupla extensao (.png.png): {len(double_ext)}")
    for r in double_ext[:15]:
        print(f"    {r['split']}/{r['name']}")
    print(f"  com espaco no nome: {len(spaces)}")
    for r in spaces[:10]:
        print(f"    {r['split']}/{r['name']}")
    print(f"  nome nao-ascii: {len(nonascii)}")
    for r in nonascii[:10]:
        print(f"    {r['split']}/{r['name']}")
    print(f"  variacoes 'boundBox' case: {dict(bb_case)}")
    print(f"  variantes '_competitor': {len(comp)}")

    # ---------- 9. PROPRIEDADES DE IMAGEM ----------
    hr("9. PROPRIEDADES DE IMAGEM")
    modes = Counter(r["mode"] for r in records if r["mode"])
    print(f"  modos de cor: {dict(modes)}")
    sizes = [r["size"] for r in records if r["size"]]
    if sizes:
        ws = np.array([s[0] for s in sizes]); hs_ = np.array([s[1] for s in sizes])
        ar = ws / np.maximum(hs_, 1)
        print(f"  largura  : min={ws.min()} med={int(np.median(ws))} max={ws.max()}")
        print(f"  altura   : min={hs_.min()} med={int(np.median(hs_))} max={hs_.max()}")
        print(f"  aspect   : min={ar.min():.2f} med={np.median(ar):.2f} max={ar.max():.2f}")
        tiny = [r for r in records if r["size"] and (r["size"][0] < 64 or r["size"][1] < 64)]
        print(f"  imagens minusculas (<64px lado): {len(tiny)}")
        for r in tiny[:10]:
            print(f"    {r['split']}/{r['name']} {r['size']}")

    # ---------- resumo executivo ----------
    hr("RESUMO EXECUTIVO")
    print(f"  imagens totais .............. {tot_img}")
    print(f"  Zone.Identifier (lixo) ...... {tot_zone}")
    print(f"  outros nao-imagem ........... {tot_other}")
    print(f"  corrompidas ................. {len(corrupt)}")
    print(f"  dup. exatas (conjuntos) ..... {len(dups)}  (redundantes: {extra}, cruzando splits: {n_cross})")
    print(f"  near-dup cruzando splits .... {len(near_cross)}")
    print(f"  VAZAMENTO grupos reais ...... {len(real_leaks)}")
    print(f"  parent-synth em val/test .... {len(cross_parent)}")
    print(f"  reais SEM rotulo ............ {len(unlab)} / {len(real)}")
    print(f"  dupla-extensao .............. {len(double_ext)}")

    if args.json:
        rep = {
            "totais": {"imgs": tot_img, "zone": tot_zone, "other": tot_other},
            "corrupt": [str(p) for p, _ in corrupt],
            "dup_sets": [[f"{x['split']}/{x['source']}/{x['name']}" for x in rs] for rs in dups.values()],
            "near_cross": [[f"{a['split']}:{a['name']}", f"{b['split']}:{b['name']}"] for a, b in near_cross],
            "real_leaks": real_leaks,
            "cross_parent": sorted(cross_parent),
            "unlabeled_real": [f"{r['split']}/{r['name']}" for r in unlab],
            "double_ext": [f"{r['split']}/{r['name']}" for r in double_ext],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(f"\nRelatorio JSON: {args.json}")


if __name__ == "__main__":
    main()
