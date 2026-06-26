#!/usr/bin/env python
"""Reconstroi data/dataset_indt com CLEAN SO SINTETICO (nao existe clean real de device).

As ~16 telas limpas de APP sao o unico clean disponivel. Para uma avaliacao honesta (sem
resubstituicao), elas sao particionadas por APP em conjuntos DISJUNTOS P_tr/P_va/P_te:
  _clean_pool/      pool estavel das 16 limpas (fora dos splits; reprodutibilidade/re-run)
  train/real/       = P_tr  (clean, negativos label 0)
  train/synthetic/  = erros 4-classe injetados em P_tr (convencao {parent}__{tipo}__vN.png)
  val/real/         = erros REAIS (108) + CLEAN SINTETICO (variantes reflow/benign de P_va, label 0)
  test/real/        = erros REAIS (160) + CLEAN SINTETICO (variantes de P_te, label 0)

O clean de val/test e' SINTETICO (dominio app). make_synthetic gera val_synth/test_synth (erros
injetados nessas limpas) e val_reflow/test_reflow. Como o clean sintetico entra em val.npz/test.npz
(source=real), o modelo treina/avalia SEM mudanca de codigo. Headline honesto = AUROC livre-de-
confound + recall nos erros reais; especificidade/precisao no dominio DEVICE nao sao mensuraveis
(nao ha clean de device) — limitacao aceita.

Anti-vazamento: P_tr/P_va/P_te disjuntos por APP; train/synthetic so de P_tr; erros reais ja split
por ticket. Le o estado atual via labels.csv (clean pool e erros sao estaveis entre re-runs).

Uso:
    python scripts/rebuild_dataset_indt.py --dry-run
    python scripts/rebuild_dataset_indt.py --apply
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image  # noqa: E402

from siamese.manifest import _parse_meta, _group_key, category_id  # noqa: E402
from siamese.synthetic import inject, SYNTH_TO_CATEGORY, MULTICLASS_SYNTH_TYPES  # noqa: E402
from siamese.reflow import reflow_augment, DEFAULT_REFLOW_WEIGHTS  # noqa: E402
from siamese.synth_features import benign_augment  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LABELS_COLS = ["path", "split", "source", "category", "category_id", "label",
               "group", "form_factor", "orientation", "kind", "is_competitor", "has_boundbox"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def imgs_in(d: Path):
    if not d or not d.is_dir():
        return []
    return [p for p in sorted(d.rglob("*")) if p.is_file()
            and "Zone.Identifier" not in p.name and p.suffix.lower() in IMG_EXTS]


def app_key(name: str) -> str:
    """Chave de APP p/ agrupar telas limpas (anti-vazamento). Pacote (com.x.y) nao tem '_',
    entao o 1o token por '_' isola o app; 'metadata_screenshot_*' -> 'metadata' (agrupa)."""
    return name.split("_")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/dataset_indt"))
    ap.add_argument("--n-variants", type=int, default=12, help="erros sinteticos por tela limpa (train)")
    ap.add_argument("--n-clean-variants", type=int, default=8, help="clean sinteticos por parent (val/test)")
    ap.add_argument("--val-apps", type=int, default=2, help="qtos APPS held-out para val")
    ap.add_argument("--test-apps", type=int, default=2, help="qtos APPS held-out para test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synth-seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.apply):
        ap.error("use --dry-run ou --apply")

    root = args.root
    labels_path = root / "labels.csv"
    if not labels_path.exists():
        sys.exit(f"{labels_path} ausente — rode a versao anterior do rebuild primeiro.")
    rows = list(csv.DictReader(open(labels_path, newline="")))

    # --- pool estavel das limpas (bootstrap do estado atual, se necessario) ---
    pool_dir = root / "_clean_pool"
    if not pool_dir.is_dir():
        pool_dir.mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r["category"] == "clean" and r["source"] == "real":
                src = root / r["path"]
                if src.exists():
                    shutil.copy2(src, pool_dir / src.name)
        print(f"[bootstrap] _clean_pool criado com {len(imgs_in(pool_dir))} telas limpas.")
    pool = imgs_in(pool_dir)
    if not pool:
        sys.exit("pool de limpas vazio — nada a fazer.")

    # --- erros reais (estaveis) do labels.csv, por split ---
    errors = {"val": [], "test": []}
    for r in rows:
        if r["label"] == "1" and r["source"] == "real" and r["split"] in errors:
            p = root / r["path"]
            if p.exists():
                errors[r["split"]].append((p, r["category"]))

    # --- particiona o pool por APP (disjunto) ---
    apps = defaultdict(list)
    for p in pool:
        apps[app_key(p.name)].append(p)
    app_names = sorted(apps)
    rng = random.Random(args.seed)
    rng.shuffle(app_names)
    va_apps = set(app_names[:args.val_apps])
    te_apps = set(app_names[args.val_apps:args.val_apps + args.test_apps])
    tr_apps = set(app_names[args.val_apps + args.test_apps:])
    P = {"train": [p for a in tr_apps for p in apps[a]],
         "val":   [p for a in va_apps for p in apps[a]],
         "test":  [p for a in te_apps for p in apps[a]]}
    print(f"Apps: {len(app_names)} -> train={sorted(tr_apps)}\n      val={sorted(va_apps)} test={sorted(te_apps)}")
    print(f"Parents clean: train(P_tr)={len(P['train'])} val(P_va)={len(P['val'])} test(P_te)={len(P['test'])}")
    print(f"Erros reais: val={len(errors['val'])} test={len(errors['test'])}")
    n_syn = len(P["train"]) * args.n_variants
    n_vclean = len(P["val"]) * args.n_clean_variants
    n_tclean = len(P["test"]) * args.n_clean_variants
    print(f"A gerar: train/synthetic={n_syn} (erros) | val clean sint={n_vclean} | test clean sint={n_tclean}")
    if not P["val"] or not P["test"] or not P["train"]:
        sys.exit("ERRO: alguma particao ficou vazia (ajuste --val-apps/--test-apps).")

    if args.dry_run:
        print("\n[dry-run] nada escrito.")
        return

    # ------------- STAGING -------------
    staging = root.parent / (root.name + ".rebuild")
    if staging.exists():
        shutil.rmtree(staging)
    for sp in ("train/real", "train/synthetic", "val/real", "test/real", "_clean_pool"):
        (staging / sp).mkdir(parents=True, exist_ok=True)

    labels_rows = []

    def add(relpath, split, source, cat, label, group, meta, kind=None):
        m = dict(meta)
        if kind:
            m["kind"] = kind
        labels_rows.append({
            "path": relpath, "split": split, "source": source, "category": cat,
            "category_id": category_id(cat), "label": label, "group": group,
            "form_factor": m.get("form_factor", ""), "orientation": m.get("orientation", ""),
            "kind": m.get("kind", ""), "is_competitor": m.get("is_competitor", ""),
            "has_boundbox": m.get("has_boundbox", ""),
        })

    used = defaultdict(set)

    def uniq(name, key):
        u = used[key]
        if name in u:
            st, sf = Path(name).stem, Path(name).suffix
            i = 1
            while f"{st}_{i}{sf}" in u:
                i += 1
            name = f"{st}_{i}{sf}"
        u.add(name)
        return name

    # pool preservado (fora dos splits; nao entra no labels.csv)
    for p in pool:
        shutil.copy2(p, staging / "_clean_pool" / p.name)

    # train/real = P_tr clean (negativos)
    for p in sorted(P["train"], key=lambda p: p.name):
        nm = uniq(p.name, ("train", "real"))
        shutil.copy2(p, staging / "train" / "real" / nm)
        add(f"train/real/{nm}", "train", "real", "clean", 0,
            f"indtclean:{Path(nm).stem}", _parse_meta(nm, "no_erros"))

    # train/synthetic = erros 4-classe de P_tr
    rng_s = random.Random(args.synth_seed)
    n_made = 0
    for p in sorted(P["train"], key=lambda p: p.name):
        img = Image.open(p).convert("RGB")
        stem = re.sub(r"_clean$", "", p.stem)
        for v in range(args.n_variants):
            corr, types = inject(img, rng_s, n_errors=1, types=MULTICLASS_SYNTH_TYPES)
            cat = SYNTH_TO_CATEGORY.get(types[0]) or "overlay"
            nm = uniq(f"{stem}__{'+'.join(types)}__v{v}.png", ("train", "synthetic"))
            corr.save(staging / "train" / "synthetic" / nm)
            add(f"train/synthetic/{nm}", "train", "synthetic", cat, 1,
                f"synthparent:{stem}", {}, kind="synthetic")
            n_made += 1

    # val/test = erros reais + CLEAN SINTETICO (variantes de P_va/P_te)
    rng_c = random.Random(args.synth_seed + 7)
    for split in ("val", "test"):
        ddir = staging / split / "real"
        for p, cat in sorted(errors[split], key=lambda x: x[0].name):
            nm = uniq(p.name, (split, "real"))
            shutil.copy2(p, ddir / nm)
            add(f"{split}/real/{nm}", split, "real", cat, 1,
                _group_key(Path(nm), "errors_dataset"), _parse_meta(nm, "errors_dataset"))
        for p in sorted(P[split], key=lambda p: p.name):
            img = Image.open(p).convert("RGB")
            stem = re.sub(r"_clean$", "", p.stem)
            for v in range(args.n_clean_variants):
                out, ops = reflow_augment(img, rng_c, ops_weights=DEFAULT_REFLOW_WEIGHTS, max_ops=2)
                out = benign_augment(out, rng_c)
                nm = uniq(f"{stem}__clean__v{v}.png", (split, "real"))
                out.save(ddir / nm)
                add(f"{split}/real/{nm}", split, "real", "clean", 0,
                    f"indtclean_{split}:{stem}", _parse_meta(stem, "no_erros"),
                    kind="synthetic_clean")

    print(f"\nGerados: train/synthetic={n_made} | clean sint val/test")

    with (staging / "labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABELS_COLS)
        w.writeheader(); w.writerows(labels_rows)

    # ------------- AUTO-VERIFICACAO -------------
    # disjuncao por app entre splits (anti-vazamento de parent)
    assert not (tr_apps & va_apps) and not (tr_apps & te_apps) and not (va_apps & te_apps), \
        "apps cruzando splits"
    staged = [p for sp in ("train", "val", "test") for p in (staging / sp).rglob("*")
              if p.is_file() and p.suffix.lower() in IMG_EXTS]
    zone = [p for p in staging.rglob("*") if "Zone.Identifier" in p.name]
    shas = defaultdict(list)
    for p in staged:
        shas[sha256(p)].append(p)
    crossdup = sum(1 for ps in shas.values()
                   if len({pp.relative_to(staging).parts[0] for pp in ps}) > 1)
    dup_any = sum(len(ps) - 1 for ps in shas.values())
    # vazamento de ticket entre val/test (erros reais)
    tk = defaultdict(set)
    for r in labels_rows:
        if r["source"] == "real" and r["label"] == 1:
            tk[r["group"]].add(r["split"])
    leak = {g: sorted(v) for g, v in tk.items() if len(v) > 1}
    # cada split tem clean (label 0) e erro (label 1)?
    has = {sp: {"clean": False, "err": False} for sp in ("train", "val", "test")}
    for r in labels_rows:
        has[r["split"]]["clean" if r["label"] == 0 else "err"] = True
    print("\n--- auto-verificacao (staging) ---")
    print(f"  imagens={len(staged)} zone={len(zone)} dup_exatas={dup_any} "
          f"dup_cruzando_split={crossdup} vazamento_ticket={len(leak)}")
    print(f"  splits com clean&erro: val={has['val']} test={has['test']} "
          f"train_clean={has['train']['clean']}")
    bad = (zone or crossdup or leak or not has['val']['clean'] or not has['val']['err']
           or not has['test']['clean'] or not has['test']['err'] or not has['train']['clean'])
    if bad:
        print("  FALHA — nao vou trocar. staging em:", staging)
        sys.exit(1)
    print("  OK: clean sintetico em val/test, 0 lixo, 0 dup cruzando, 0 vazamento de ticket, "
          "apps disjuntos por split.")

    # ------------- SWAP -------------
    backup = root.parent / (root.name + ".old")
    if backup.exists():
        shutil.rmtree(backup)
    root.rename(backup)
    staging.rename(root)
    shutil.rmtree(backup)

    print(f"\nOK: {root} reconstruido. labels.csv: {root}/labels.csv")
    by = Counter((r["split"], r["source"], "clean" if r["label"] == 0 else "err") for r in labels_rows)
    for k in sorted(by):
        print(f"  {k[0]}/{k[1]} [{k[2]}]: {by[k]}")
    print(f"\nProximo: python scripts/run_experiment.py --config configs/dataset_indt.yaml "
          f"--processed {root} --fresh")


if __name__ == "__main__":
    main()
