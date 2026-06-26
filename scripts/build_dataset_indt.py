#!/usr/bin/env python
"""Distribui as imagens REAIS de data/input/input_indt/<categoria>/ entre
data/dataset_indt/{val,test}/real/.

Politica (mesma filosofia anti-vazamento de scripts/build_splits.py):

  * AGRUPAMENTO POR TICKET (IKSWW-XXXXX): todas as capturas e variantes do mesmo
    ticket (screenshot1/2..., _boundBox, _competitor) ficam no MESMO split, pois
    sao quase-duplicatas da mesma tela. Nenhum ticket cruza val/test.
  * ESTRATIFICACAO POR CATEGORIA: cada categoria mantem ~val_frac dos ARQUIVOS em
    val e o resto em test. Tickets que aparecem em 2 categorias sao alocados
    primeiro (globalmente) para nao quebrar a atomicidade do grupo.
  * Pastas de destino sao PLANAS (sem subpasta de categoria). Em caso de nomes de
    arquivo iguais entre categorias, o nome de destino recebe o sufixo __<categoria>.

train/synthetic NAO e' tocado (o treino deste dataset e' 100% sintetico).

Uso:
    python scripts/build_dataset_indt.py --dry-run          # so mostra o plano
    python scripts/build_dataset_indt.py --move --clear-dest  # executa (move + limpa destino)
    python scripts/build_dataset_indt.py --copy --clear-dest  # executa (copia)
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

TICKET_RE = re.compile(r"IKSWW[-_](\d+)", re.IGNORECASE)


def ticket_of(name: str) -> str:
    m = TICKET_RE.search(name)
    return f"IKSWW-{m.group(1)}" if m else f"__noticket__:{name}"


def cat_slug(cat: str) -> str:
    return re.sub(r"\s+", "_", cat.strip()).lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/input/input_indt"))
    ap.add_argument("--out", type=Path, default=Path("data/dataset_indt"))
    ap.add_argument("--val-frac", type=float, default=0.40,
                    help="fracao (por arquivo, por categoria) que vai para val; resto vai p/ test")
    ap.add_argument("--seed", type=int, default=42)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--move", action="store_true", help="MOVE os arquivos (esvazia input_indt)")
    g.add_argument("--copy", action="store_true", help="COPIA os arquivos (mantem input_indt)")
    ap.add_argument("--dry-run", action="store_true", help="so mostra o plano, nao escreve nada")
    ap.add_argument("--clear-dest", action="store_true",
                    help="apaga os arquivos atuais em out/{val,test}/real antes de popular")
    args = ap.parse_args()

    if not (args.move or args.copy or args.dry_run):
        ap.error("escolha uma acao: --dry-run, --move ou --copy")

    rng = random.Random(args.seed)

    # 1) coleta arquivos por categoria (estrutura plana dentro de cada categoria)
    cats = sorted((d for d in args.input.iterdir() if d.is_dir()), key=lambda p: p.name)
    files: list[tuple[Path, str, str]] = []  # (src, categoria, basename)
    for d in cats:
        for p in sorted(d.iterdir()):
            if p.is_file():
                files.append((p, d.name, p.name))
    if not files:
        sys.exit(f"Nenhum arquivo em {args.input} (ja foi movido?).")

    # 2) agrupa por ticket
    groups: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for src, cat, base in files:
        groups[ticket_of(base)].append((src, cat, base))

    # alvos por categoria (contando ARQUIVOS)
    total: dict[str, int] = defaultdict(int)
    for _, cat, _ in files:
        total[cat] += 1
    test_target = {c: round((1 - args.val_frac) * n) for c, n in total.items()}
    val_target = {c: total[c] - test_target[c] for c in total}

    cur = {"val": defaultdict(int), "test": defaultdict(int)}
    assign: dict[str, str] = {}  # ticket -> split

    def group_cats(members):
        cc: dict[str, int] = defaultdict(int)
        for _, cat, _ in members:
            cc[cat] += 1
        return cc

    def remaining(split: str, ccounts) -> int:
        tgt = val_target if split == "val" else test_target
        return sum(tgt[c] - cur[split][c] for c in ccounts)

    def place(ticket: str, members, split: str) -> None:
        assign[ticket] = split
        for _, cat, _ in members:
            cur[split][cat] += 1

    # ordem deterministica: embaralha (seed) e depois ordena por tamanho desc (estavel)
    items = list(groups.items())
    rng.shuffle(items)
    multi = [(t, m) for t, m in items if len(group_cats(m)) > 1]
    single = [(t, m) for t, m in items if len(group_cats(m)) == 1]

    # 3) grupos multi-categoria primeiro (globais), maiores primeiro
    for t, m in sorted(multi, key=lambda x: -len(x[1])):
        cc = group_cats(m)
        split = "test" if remaining("test", cc) >= remaining("val", cc) else "val"
        place(t, m, split)

    # 4) grupos de categoria unica, por categoria, maiores primeiro
    by_cat = defaultdict(list)
    for t, m in single:
        by_cat[m[0][1]].append((t, m))
    for cat in sorted(by_cat):
        for t, m in sorted(by_cat[cat], key=lambda x: -len(x[1])):
            cc = group_cats(m)
            split = "test" if remaining("test", cc) >= remaining("val", cc) else "val"
            place(t, m, split)

    # 5) garante cada categoria presente nos DOIS splits (defensivo)
    for c in sorted(total):
        for need, other in (("val", "test"), ("test", "val")):
            if cur[need][c] == 0:
                cand = sorted(((t, m) for t, m in single
                               if m[0][1] == c and assign[t] == other),
                              key=lambda x: len(x[1]))
                if cand:
                    t, m = cand[0]
                    for _, cc, _ in m:
                        cur[other][cc] -= 1
                    place(t, m, need)

    # 6) plano de destino + desambiguacao de nomes iguais entre categorias
    base_count: dict[str, int] = defaultdict(int)
    for _, _, base in files:
        base_count[base] += 1

    def dest_name(base: str, cat: str) -> str:
        if base_count[base] > 1:
            p = Path(base)
            return f"{p.stem}__{cat_slug(cat)}{p.suffix}"
        return base

    plan: list[tuple[Path, Path, str, str]] = []  # (src, dest, split, cat)
    for t, members in groups.items():
        split = assign[t]
        ddir = args.out / split / "real"
        for src, cat, base in members:
            plan.append((src, ddir / dest_name(base, cat), split, cat))

    # --- verificacoes ---
    # nenhum ticket cruza splits (por construcao, mas confirmamos)
    leaks = [t for t in groups if t not in assign]
    assert not leaks, f"tickets sem split: {leaks}"
    # destinos unicos
    dests = [d for _, d, _, _ in plan]
    dup_dest = {d for d in dests if dests.count(d) > 1}
    assert not dup_dest, f"colisao de destino apos desambiguacao: {sorted(dup_dest)[:5]}"

    # --- auditoria ---
    print(f"Fonte: {args.input}  ->  {args.out}/{{val,test}}/real")
    print(f"val_frac={args.val_frac:.0%}  seed={args.seed}  "
          f"acao={'DRY-RUN' if args.dry_run else ('MOVE' if args.move else 'COPY')}\n")

    tk_split = defaultdict(lambda: defaultdict(set))   # cat -> split -> {tickets}
    fl_split = defaultdict(lambda: defaultdict(int))   # cat -> split -> n_arquivos
    for t, members in groups.items():
        sp = assign[t]
        for _, cat, _ in members:
            tk_split[cat][sp].add(t)
            fl_split[cat][sp] += 1

    print("--- Distribuicao por categoria (arquivos | tickets) ---")
    print(f"  {'categoria':18s} {'val':>14s} {'test':>14s} {'total':>7s}")
    tv = tt = 0
    for c in sorted(total):
        v, te = fl_split[c]["val"], fl_split[c]["test"]
        vt, tte = len(tk_split[c]["val"]), len(tk_split[c]["test"])
        tv += v
        tt += te
        pct = f"{v / total[c]:.0%}" if total[c] else "-"
        print(f"  {c:18s} {v:5d} ({vt:3d}tk) {te:5d} ({tte:3d}tk) {total[c]:7d}   val={pct}")
    n = tv + tt
    print(f"  {'TOTAL':18s} {tv:5d}        {tt:5d}        {n:7d}   "
          f"val={tv / n:.0%} / test={tt / n:.0%}")

    # colisoes de nome tratadas
    collided = sorted({b for b, c in base_count.items() if c > 1})
    if collided:
        print(f"\n--- Nomes iguais entre categorias (desambiguados com __<categoria>): "
              f"{len(collided)} ---")
        for b in collided:
            print(f"  {b}")

    print(f"\nArquivos a {'mover' if args.move else 'copiar' if args.copy else 'distribuir'}: "
          f"{len(plan)}  (val={tv}, test={tt})")

    if args.dry_run:
        print("\n[dry-run] nada foi escrito.")
        return

    # --- execucao ---
    for sp in ("val", "test"):
        (args.out / sp / "real").mkdir(parents=True, exist_ok=True)

    if args.clear_dest:
        removed = 0
        for sp in ("val", "test"):
            ddir = args.out / sp / "real"
            for p in ddir.iterdir():
                if p.is_file():
                    p.unlink()
                    removed += 1
        print(f"\nLimpeza: {removed} arquivos antigos removidos de {{val,test}}/real")

    action = shutil.move if args.move else shutil.copy2
    for src, dest, _, _ in plan:
        action(str(src), str(dest))
    print(f"OK: {len(plan)} arquivos {'movidos' if args.move else 'copiados'}.")

    # resumo final no disco
    for sp in ("val", "test"):
        d = args.out / sp / "real"
        print(f"  {sp}/real: {sum(1 for _ in d.iterdir())} arquivos")
    if args.move:
        leftover = sum(1 for _ in args.input.rglob('*') if _.is_file())
        print(f"  input_indt restante: {leftover} arquivos")


if __name__ == "__main__":
    main()
