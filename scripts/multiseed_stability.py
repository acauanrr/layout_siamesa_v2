#!/usr/bin/env python
"""Estabilidade multi-seed dos top configs (Fase 2.3 da auditoria) — SO val, teste trancado.

Roda cada config sob N seeds, mede a metrica de selecao honesta `val_synth_gate` (gate
sintetico livre de confound na val), e aplica a REGRA DO 1-SE: escolhe o config mais
PARCIMONIOSO (menor proj_dim) cuja media esteja dentro de 1 erro-padrao do melhor. Isso evita
declarar vencedor por sorte de seed (criterio de aceite #4: estabilidade de seed).

Uso: python scripts/multiseed_stability.py [--seeds 5]
"""
from __future__ import annotations

import argparse
import copy
import tempfile

import numpy as np

from siamese.config import Config
from siamese.train import train_head

# (rotulo, proj_dim, weight_decay) — top combos do grid + o config atual como baseline
CONFIGS = [
    ("proj64_wd1e-4", 64, 1e-4),     # vencedor do grid
    ("proj32_wd1e-3", 32, 1e-3),     # menor + mais regularizado
    ("proj128_wd1e-4", 128, 1e-4),
    ("proj256_wd1e-4(atual)", 256, 1e-4),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    base = Config.load(args.config)
    seeds = list(range(args.seeds))

    rows = []
    for label, pd, wd in CONFIGS:
        gates, auxs, cats = [], [], []
        for sd in seeds:
            cfg = copy.deepcopy(base)
            cfg.head.proj_dim = pd
            cfg.train.weight_decay = wd
            cfg.seed = sd
            tmp = tempfile.mkdtemp()
            cfg.paths.models_dir = tmp + "/m"
            cfg.paths.reports_dir = tmp + "/r"
            res = train_head(cfg, device=args.device)
            bm = res["best_metrics"]
            gates.append(float(bm["val_synth_gate"]))
            auxs.append(float(bm["val_ap_aux"]))
            cats.append(float(bm["val_cat_f1"]))
        g = np.array(gates, dtype=float)
        se = float(g.std(ddof=1) / np.sqrt(len(g)))
        rows.append({
            "label": label, "proj_dim": pd, "wd": wd,
            "mean": float(g.mean()), "std": float(g.std(ddof=1)), "se": se,
            "min": float(g.min()), "max": float(g.max()),
            "gates": [round(x, 3) for x in gates],
            "aux_mean": float(np.mean(auxs)), "cat_f1_mean": float(np.mean(cats)),
        })
        print(f"  [done] {label:22s} gate={g.mean():.3f}±{se:.3f}  seeds={[round(x,3) for x in gates]}")

    rows.sort(key=lambda r: -r["mean"])
    best = rows[0]
    thr = best["mean"] - best["se"]                       # 1 SE abaixo do melhor
    within = [r for r in rows if r["mean"] >= thr]
    pick = min(within, key=lambda r: r["proj_dim"])       # parcimonia: menor proj_dim dentro de 1 SE

    print("\n=========== ESTABILIDADE MULTI-SEED (val_synth_gate, %d seeds) ===========" % len(seeds))
    print(f"  {'config':24s} {'mean':>6s} {'±SE':>6s} {'std':>6s} {'min':>6s} {'max':>6s} {'catF1':>6s}")
    for r in rows:
        flag = "  <- MELHOR" if r is best else ("  <- 1-SE pick" if r is pick else "")
        print(f"  {r['label']:24s} {r['mean']:6.3f} {r['se']:6.3f} {r['std']:6.3f} "
              f"{r['min']:6.3f} {r['max']:6.3f} {r['cat_f1_mean']:6.3f}{flag}")
    print(f"\n  Melhor media: {best['label']} ({best['mean']:.3f}); limiar 1-SE = {thr:.3f}")
    print(f"  Dentro de 1 SE: {[r['label'] for r in within]}")
    print(f"  >>> ESCOLHA (regra do 1-SE, mais parcimonioso): {pick['label']} "
          f"(proj_dim={pick['proj_dim']}, weight_decay={pick['wd']})")
    print(f"\n  NB: selecao 100% na VAL (teste nunca lido). Congele esta config antes do --final-test.")


if __name__ == "__main__":
    main()
