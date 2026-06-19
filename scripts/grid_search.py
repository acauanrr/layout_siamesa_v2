#!/usr/bin/env python
"""Grid search de hiperparametros para a rede siamesa (Estagio 1 gate + Estagio 2 clusters).

Generaliza scripts/compare_preprocess.py: varre o PRODUTO CARTESIANO de uma grade dada por
chaves dotted (ex.: train.aux_weight, head.proj_dim, decision.k_prototypes), e para cada
ponto: copia a config, aplica os valores, ISOLA os diretorios de artefatos, RE-EXTRAI os
embeddings SO se o eixo tocar backbone.*/synthetic.* (senao reusa o cache -> treino em
segundos), treina e avalia.

SELECAO HONESTA (evita data-snooping no limiar): ranqueia pela metrica INDEPENDENTE DE
LIMIAR e livre de confound -- AUROC sintetico do gate (sintetico_livre_de_confound.modelo_
fusao.auroc). Reporta tambem o F1_macro de categoria (Estagio 2) e o ponto de operacao.
NUNCA seleciona pelo TEST diretamente.

Uso:
    python scripts/grid_search.py --config configs/default.yaml            # grade padrao (barata)
    python scripts/grid_search.py --rank-by cat_f1                         # ranqueia por clusterizacao
    python scripts/grid_search.py --grid '{"train.aux_weight":[0.1,0.3,0.6]}'
    python scripts/grid_search.py --max-combos 8
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import time
from pathlib import Path

import torch

from siamese.config import Config
from siamese.backbone import DinoV2Backbone, BackboneConfig
from siamese.features import extract_processed
from siamese.synth_features import extract_synthetic
from siamese.train import train_head
from siamese.evaluate import evaluate

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Grade PADRAO sobre eixos BARATOS (treino em segundos; reusa embeddings cacheados).
# Eixos caros (backbone.*/synthetic.*) podem ser adicionados via --grid (re-extraem features).
DEFAULT_GRID = {
    "train.aux_weight": [0.3, 0.6],
    "train.temperature": [0.07, 0.1],
    "head.proj_dim": [128, 256],
    "decision.k_prototypes": [1, 3],
}


def _set(cfg: Config, dotted: str, value) -> None:
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def _needs_extract(keys) -> bool:
    return any(k.startswith("backbone.") or k.startswith("synthetic.") for k in keys)


def _extract_for(cfg: Config, processed: Path = Path("data/processed")) -> None:
    """Re-extrai embeddings da FONTE DA VERDADE (data/processed/) quando o eixo toca o backbone.
    NB: eixos SINTETICOS (n_variants etc.) exigem re-rodar export_processed.py antes — aqui so
    re-embedamos os arquivos ja' materializados em processed/."""
    bb = DinoV2Backbone(BackboneConfig(
        model_name=cfg.backbone.model_name, size=cfg.backbone.size,
        use_patch_stats=cfg.backbone.use_patch_stats,
        preprocess=cfg.backbone.preprocess, device=DEVICE))
    emb = Path(cfg.paths.emb_dir)
    # reais (train/val/test) + train_synth materializado
    extract_processed(processed, emb, bb, batch_size=cfg.backbone.batch_size)
    # sonda livre de confound val/test (de processed/{val,test}/real/clean)
    for split, seed in [("val", cfg.synthetic.seed + 100), ("test", cfg.synthetic.seed + 200)]:
        d = processed / split / "real" / "clean"
        rows = [{"path": str(p.resolve())} for p in sorted(d.iterdir()) if p.suffix.lower() in _EXTS]
        if rows:
            extract_synthetic(None, emb / f"{split}_synth.npz", bb,
                              n_variants=cfg.synthetic.n_variants,
                              max_errors_per_image=cfg.synthetic.max_errors_per_image,
                              seed=seed, batch_size=cfg.backbone.batch_size,
                              multiclass=cfg.train.multiclass, clean_rows=rows)


def _metrics(rep: dict) -> dict:
    """Extrai as metricas-chave de um relatorio de evaluate()."""
    synt = rep.get("sintetico_livre_de_confound", {}).get("modelo_fusao", {})
    ctrl = rep.get("primaria_subconjunto_controlado", {}).get("modelo_fusao", {})
    op = rep.get("ponto_operacao", {})
    e2 = rep.get("estagio2_categoria", {}).get("por_prototipo", {})
    e2a = rep.get("estagio2_categoria", {}).get("por_aux_head", {})
    return {
        "synth_auroc": float(synt.get("auroc", float("nan"))),   # <- metrica de selecao honesta (gate)
        "synth_ap": float(synt.get("ap", float("nan"))),
        "controlled_auroc": float(ctrl.get("auroc", float("nan"))),
        "op_precision": float(op.get("precisao", float("nan"))),
        "op_recall": float(op.get("recall", float("nan"))),
        "op_f1": float(op.get("f1", float("nan"))),
        "cat_f1": float(e2.get("f1_macro", float("nan"))),        # <- clusterizacao (protótipo)
        "cat_acc": float(e2.get("accuracy", float("nan"))),
        "cat_f1_aux": float(e2a.get("f1_macro", float("nan"))),
    }


def run_point(base: Config, combo: dict, idx: int, reuse_emb: str | None) -> dict:
    cfg = copy.deepcopy(base)
    for k, v in combo.items():
        _set(cfg, k, v)
    root = Path("artifacts/grid") / f"pt{idx:03d}"
    cfg.paths.models_dir = str(root / "models")
    cfg.paths.reports_dir = str(root / "reports")
    if reuse_emb is not None:
        cfg.paths.emb_dir = reuse_emb           # reusa cache (eixos baratos)
    else:
        cfg.paths.emb_dir = str(root / "emb")   # isola e re-extrai (eixos de backbone/synth)
        _extract_for(cfg)
    train_head(cfg, device=DEVICE)
    rep = evaluate(cfg, device=DEVICE)
    return _metrics(rep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--grid", type=str, default=None, help="JSON {chave.dotted: [valores]}")
    ap.add_argument("--rank-by", default="synth_auroc",
                    choices=["synth_auroc", "synth_ap", "cat_f1", "controlled_auroc", "op_f1"],
                    help="metrica de selecao (padrao: synth_auroc -- honesta, livre de limiar)")
    ap.add_argument("--max-combos", type=int, default=0, help="limita o numero de combos (0=todos)")
    ap.add_argument("--out", type=Path, default=Path("artifacts/reports/grid_results.csv"))
    args = ap.parse_args()

    base = Config.load(args.config)
    grid = json.loads(args.grid) if args.grid else DEFAULT_GRID
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
    if args.max_combos and len(combos) > args.max_combos:
        print(f"AVISO: {len(combos)} combos -> limitando aos primeiros {args.max_combos} "
              f"(use --max-combos 0 p/ todos)")
        combos = combos[:args.max_combos]

    needs_extract = _needs_extract(keys)
    reuse_emb = None if needs_extract else base.paths.emb_dir
    print(f"GRID: {len(combos)} combos sobre {keys}")
    print(f"  re-extrai embeddings por combo? {needs_extract}  (eixos caros de backbone/synthetic)")
    print(f"  selecao por: {args.rank_by} (honesta/livre de limiar)\n")

    rows = []
    for i, combo in enumerate(combos):
        t0 = time.time()
        try:
            m = run_point(base, combo, i, reuse_emb)
        except Exception as e:
            print(f"  [{i+1}/{len(combos)}] {combo} -> ERRO: {e}")
            continue
        m_row = {**combo, **m, "secs": round(time.time() - t0, 1)}
        rows.append(m_row)
        print(f"  [{i+1}/{len(combos)}] {combo} -> synth_auroc={m['synth_auroc']:.3f} "
              f"cat_f1={m['cat_f1']:.3f} op_f1={m['op_f1']:.3f} ({m_row['secs']}s)")

    if not rows:
        print("Nenhum combo concluido."); return

    rows.sort(key=lambda r: (-(r[args.rank_by] if r[args.rank_by] == r[args.rank_by] else -1)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = keys + ["synth_auroc", "synth_ap", "controlled_auroc", "op_precision", "op_recall",
                   "op_f1", "cat_f1", "cat_acc", "cat_f1_aux", "secs"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n========= RANKING (top, por {args.rank_by}) =========")
    head = "  " + "  ".join(f"{k.split('.')[-1]:>10s}" for k in keys) + \
           f"  {'synth_AUROC':>11s} {'cat_F1':>7s} {'op_F1':>6s} {'ctrl_AUROC':>10s}"
    print(head)
    for r in rows[:10]:
        vals = "  ".join(f"{str(r[k]):>10s}" for k in keys)
        print(f"  {vals}  {r['synth_auroc']:>11.3f} {r['cat_f1']:>7.3f} "
              f"{r['op_f1']:>6.3f} {r['controlled_auroc']:>10.3f}")
    best = rows[0]
    print(f"\nMELHOR ({args.rank_by}={best[args.rank_by]:.3f}): "
          f"{ {k: best[k] for k in keys} }")
    print(f"Resultados completos: {args.out}")
    print("NB: selecao por metrica livre de limiar (anti-snooping). Reavalie o melhor combo")
    print("    no TEST uma unica vez com scripts/evaluate.py apos fixar a config vencedora.")


if __name__ == "__main__":
    main()
