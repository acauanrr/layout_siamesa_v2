#!/usr/bin/env python
"""Fase 2.b (passo 1 da `docs/SPEC_COLETA_FOLDABLE.md`): captura telas LIMPAS **foldable** —
o domínio de produção que o cross-eval + o experimento #1 provaram ser o gargalo (problema de
CONTEÚDO, não de aspecto). Materializa em data/clean_extra_fold/ com um manifesto ESTENDIDO
(form_factor/orientation/device) que o `merge_clean_extra.py` (já destravado) consome.

Fonte da imagem (qualquer uma):
  - **adb** (emulador foldable do Android Studio OU device físico): `adb exec-out screencap -p`.
  - **--from-file**: importa um PNG já capturado (device físico, press kit curado, etc.).

POR QUE foldable e não só near-square: reflowar conteúdo phone/desktop p/ AR 0.96 NÃO move o
domínio (espec foldable 0.512->0.512 no #1). Só **conteúdo foldable real** tira a especificidade
do chão. Por isso: muitos APPS/telas distintos, em ≥4 postures (unfold/fold/tent/laptop) e 2
orientações, em resoluções NATIVAS (não redimensionar) — espelhando a distribuição dos erros.

Convenções (compatíveis com fetch_clean_extra/merge_clean_extra):
  - imagens: data/clean_extra_fold/<device>/<arquivo>.png (resolução nativa, sem moldura de device).
  - manifesto: data/clean_extra_fold/labels_extra.csv
      path, source, w, h, aspect, group, phash, form_factor, orientation, device
  - dedup PERCEPTUAL (dHash, dist<=6) cruzando passadas via coluna phash (--append acumula).
  - **group = conteúdo** `fold:<app>:<screen>` (device-independente) -> postures/orientações E
    o MESMO conteúdo em devices distintos ficam no MESMO split (anti-vazamento mais forte que
    device:app:screen; o device vai em coluna própria p/ estratificar). Override: --group.

Uso:
  python scripts/capture_foldable.py devices                 # lista seriais adb + resolução
  python scripts/capture_foldable.py capture --device pixel_fold --form-factor unfold \
         --orientation portrait --app settings --screen wifi          # captura via adb
  python scripts/capture_foldable.py capture --device zfold5 --form-factor fold \
         --app chrome --screen news --from-file /tmp/shot.png         # importa PNG
  python scripts/capture_foldable.py batch --device pixel_fold --plan data/fold_plan.csv
  python scripts/capture_foldable.py audit                   # progresso vs metas da spec
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MANIFEST_COLS = ["path", "source", "w", "h", "aspect", "group", "phash",
                 "form_factor", "orientation", "device"]
FORM_FACTORS = ("unfold", "fold", "tent", "laptop")
ORIENTATIONS = ("portrait", "landscape")
PHASH_MAX_DIST = 6


# ---------------------------------------------------------------- dedup perceptual (= fetch_clean_extra)
def dhash(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def is_dup(h: int, seen: list[int]) -> bool:
    return any(_hamming(h, s) <= PHASH_MAX_DIST for s in seen)


# ---------------------------------------------------------------- manifesto
def load_manifest(out: Path) -> tuple[list[dict], list[int]]:
    man = out / "labels_extra.csv"
    if not man.exists():
        return [], []
    with open(man, newline="") as f:
        rows = list(csv.DictReader(f))
    seen = [int(r["phash"]) for r in rows if r.get("phash")]
    return rows, seen


def write_manifest(out: Path, rows: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels_extra.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        wr.writeheader()
        wr.writerows(rows)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "x"


# ---------------------------------------------------------------- ingestão (núcleo testável)
def ingest(img: Image.Image, out: Path, rows: list[dict], seen: list[int], *,
           device: str, app: str, screen: str, form_factor: str,
           orientation: str | None = None, source: str = "foldable",
           group: str | None = None) -> dict | None:
    """Adiciona UMA tela limpa ao pool (salva PNG + linha no manifesto). Devolve a linha, ou
    None se for duplicata perceptual. Resolução NATIVA preservada (não redimensiona)."""
    if form_factor not in FORM_FACTORS:
        raise ValueError(f"form_factor invalido: {form_factor!r} (use {FORM_FACTORS}) — "
                         "o subconjunto controlado em evaluate.py depende de 'unfold' etc.")
    w, h = img.size
    ph = dhash(img)
    if is_dup(ph, seen):
        return None
    orientation = orientation or ("portrait" if h >= w else "landscape")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation invalida: {orientation!r} (use {ORIENTATIONS})")
    dev = _slug(device)
    grp = group or f"fold:{_slug(app)}:{_slug(screen)}"          # conteúdo (anti-vazamento)
    stem = f"{dev}_{_slug(app)}_{_slug(screen)}_{form_factor}_{orientation}"
    (out / dev).mkdir(parents=True, exist_ok=True)
    fn = f"{stem}.png"
    if (out / dev / fn).exists():                                # colisão legítima -> sufixo phash
        fn = f"{stem}_{ph & 0xfffff:05x}.png"
    img.convert("RGB").save(out / dev / fn)
    row = {"path": f"{out.name}/{dev}/{fn}", "source": source, "w": w, "h": h,
           "aspect": round(w / h, 3), "group": grp, "phash": ph,
           "form_factor": form_factor, "orientation": orientation, "device": dev}
    rows.append(row)
    seen.append(ph)
    return row


# ---------------------------------------------------------------- adb
def adb_screencap(serial: str | None = None) -> Image.Image:
    cmd = ["adb"] + (["-s", serial] if serial else []) + ["exec-out", "screencap", "-p"]
    try:
        p = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError:
        sys.exit("adb não encontrado no PATH. Instale o platform-tools do Android SDK, ou use "
                 "--from-file <png> (device físico / arquivo curado).")
    if p.returncode != 0 or not p.stdout:
        sys.exit(f"adb screencap falhou: {p.stderr.decode(errors='ignore')[:300]}\n"
                 "Confira `adb devices` (emulador foldable ligado?) ou use --from-file.")
    return Image.open(io.BytesIO(p.stdout)).convert("RGB")


def get_image(args) -> Image.Image:
    if args.from_file:
        return Image.open(args.from_file).convert("RGB")
    return adb_screencap(args.serial)


def _near_square(asp: float) -> bool:
    return 0.85 <= asp <= 1.18


# ---------------------------------------------------------------- subcomandos
def cmd_devices(args) -> None:
    try:
        p = subprocess.run(["adb", "devices", "-l"], capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("adb não encontrado no PATH (platform-tools do Android SDK).")
    print(p.stdout.strip() or "(nenhum device/emulador)")
    print("\nDica: ligue um AVD foldable (Pixel Fold / 7.6\" Fold-in / Resizable) no Android Studio,")
    print("ou conecte um device físico (depuração USB). Use o serial em --serial.")


def cmd_capture(args) -> None:
    out = Path(args.out)
    rows, seen = load_manifest(out)
    img = get_image(args)
    row = ingest(img, out, rows, seen, device=args.device, app=args.app, screen=args.screen,
                 form_factor=args.form_factor, orientation=args.orientation, source=args.source,
                 group=args.group)
    if row is None:
        print(f"[dup] tela ignorada (duplicata perceptual de algo já no pool). Nada salvo.")
        return
    write_manifest(out, rows)
    tag = "near-square ⭐" if _near_square(float(row["aspect"])) else f"AR {row['aspect']}"
    print(f"[ok] {row['w']}x{row['h']} ({tag}) {row['form_factor']}/{row['orientation']} "
          f"grupo={row['group']} -> {row['path']}  | pool: {len(rows)} imgs")


def cmd_batch(args) -> None:
    """Percorre um plano (CSV: app,screen,form_factor[,orientation]) guiando o operador: para cada
    linha, posicione a tela/posture no device e tecle Enter p/ capturar (ou 's' p/ pular)."""
    plan = Path(args.plan)
    if not plan.exists():
        sys.exit(f"plano não encontrado: {plan} (CSV com colunas app,screen,form_factor[,orientation])")
    out = Path(args.out)
    rows, seen = load_manifest(out)
    with open(plan, newline="") as f:
        items = list(csv.DictReader(f))
    print(f"[batch] {len(items)} capturas planejadas no device '{args.device}'. "
          "Enter=capturar · s+Enter=pular · q+Enter=sair.\n")
    done = 0
    for i, it in enumerate(items, 1):
        ff = (it.get("form_factor") or "").strip()
        app, screen = (it.get("app") or "").strip(), (it.get("screen") or "").strip()
        orient = (it.get("orientation") or "").strip() or None
        prompt = (f"  [{i}/{len(items)}] prepare '{app}/{screen}' em {ff}"
                  f"{('/' + orient) if orient else ''} e tecle Enter: ")
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            print("\n(entrada encerrada — saindo do batch)"); break
        if ans == "q":
            break
        if ans == "s":
            continue
        try:
            img = get_image(args)
            row = ingest(img, out, rows, seen, device=args.device, app=app, screen=screen,
                         form_factor=ff, orientation=orient, source=args.source)
        except (ValueError, SystemExit) as e:
            print(f"      [erro] {e}"); continue
        if row is None:
            print("      [dup] ignorada."); continue
        done += 1
        write_manifest(out, rows)        # persiste a cada captura (resiliente)
        print(f"      [ok] {row['w']}x{row['h']} AR {row['aspect']} -> {row['path']}")
    print(f"\n[batch] {done} novas capturas. pool: {len(rows)} imgs -> {out}/labels_extra.csv")


def cmd_audit(args) -> None:
    out = Path(args.out)
    rows, _ = load_manifest(out)
    if not rows:
        sys.exit(f"pool vazio em {out}/labels_extra.csv — capture algo antes.")
    groups = {r["group"] for r in rows}
    devices = Counter(r.get("device", "") for r in rows)
    ff = Counter(r.get("form_factor", "") for r in rows)
    orient = Counter(r.get("orientation", "") for r in rows)
    asp = np.array([float(r["aspect"]) for r in rows])
    near = int(((asp >= 0.85) & (asp <= 1.18)).sum())
    gsizes = Counter()
    for r in rows:
        gsizes[r["group"]] += 1
    apps = {r["group"].split(":")[1] if ":" in r["group"] else r["group"] for r in rows}

    print(f"== AUDIT do pool foldable ({out}) ==")
    print(f"imagens: {len(rows)} | grupos (telas): {len(groups)} | apps distintos: ~{len(apps)}")
    print(f"near-square (0.85-1.18): {near} ({100*near//max(1,len(rows))}%)  | "
          f"AR mediana {np.median(asp):.2f} [{asp.min():.2f}..{asp.max():.2f}]")
    print(f"devices: {dict(devices)}")
    print(f"form_factor: {dict(ff)}")
    print(f"orientation: {dict(orient)}")
    print(f"telas com >1 posture/orient: {sum(1 for g in gsizes.values() if g > 1)}/{len(groups)}")

    # progresso vs metas da SPEC_COLETA_FOLDABLE.md §1.1
    def chk(ok): return "✅" if ok else "⬜"
    print("\n-- metas da spec (§1.1) --")
    print(f" {chk(len(rows) >= 300)} ≥300 imagens                      ({len(rows)})")
    print(f" {chk(len(groups) >= 50)} ≥50 grupos (telas)                ({len(groups)})")
    print(f" {chk(len([d for d in devices if d]) >= 3)} ≥3 devices/perfis                 ({len([d for d in devices if d])})")
    print(f" {chk(sum(1 for k in FORM_FACTORS if ff.get(k)) >= 4)} 4 postures (unfold/fold/tent/laptop) ({sum(1 for k in FORM_FACTORS if ff.get(k))}/4)")
    print(f" {chk(orient.get('portrait',0) > 0 and orient.get('landscape',0) > 0)} portrait E landscape")
    print(f" {chk(near >= 120)} cobertura near-square forte        ({near})")
    print("\nProx.: merge_clean_extra.py --extra {0} --dest data/processed_v3_fold --apply".format(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="lista seriais adb")

    def _common(p):
        p.add_argument("--out", type=Path, default=Path("data/clean_extra_fold"))
        p.add_argument("--device", required=True, help="nome do device p/ metadados (ex.: pixel_fold)")
        p.add_argument("--serial", default=None, help="serial adb alvo (default: 1o device)")
        p.add_argument("--source", default="foldable")

    pc = sub.add_parser("capture", help="captura/importa UMA tela limpa")
    _common(pc)
    pc.add_argument("--form-factor", required=True, choices=FORM_FACTORS)
    pc.add_argument("--orientation", default=None, choices=[*ORIENTATIONS, None])
    pc.add_argument("--app", required=True)
    pc.add_argument("--screen", required=True)
    pc.add_argument("--group", default=None, help="override do grupo (default: fold:<app>:<screen>)")
    pc.add_argument("--from-file", type=Path, default=None, help="importa PNG em vez de usar adb")

    pb = sub.add_parser("batch", help="percorre um plano CSV (app,screen,form_factor[,orientation])")
    _common(pb)
    pb.add_argument("--plan", type=Path, required=True)
    pb.add_argument("--from-file", type=Path, default=None)

    pa = sub.add_parser("audit", help="progresso do pool vs metas da spec")
    pa.add_argument("--out", type=Path, default=Path("data/clean_extra_fold"))

    args = ap.parse_args()
    {"devices": cmd_devices, "capture": cmd_capture, "batch": cmd_batch, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    main()
