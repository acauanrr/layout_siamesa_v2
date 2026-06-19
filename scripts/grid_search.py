#!/usr/bin/env python
"""Grid search de hiperparametros para a rede siamesa (Estagio 1 gate + Estagio 2 clusters).

Generaliza scripts/compare_preprocess.py: varre o PRODUTO CARTESIANO de uma grade dada por
chaves dotted (ex.: train.aux_weight, head.proj_dim, decision.k_prototypes), e para cada
ponto: copia a config, aplica os valores, ISOLA os diretorios de artefatos, RE-EXTRAI os
embeddings SO se o eixo tocar backbone.*/synthetic.* (senao reusa o cache -> treino em
segundos), treina e avalia.

SELECAO HONESTA (Fase 0 — anti-snooping, problema #2 da auditoria): ranqueia EXCLUSIVAMENTE
por metricas de VALIDACAO devolvidas por train_head() — por padrao o gate sintetico livre de
confound na val (`val_synth_gate`). NAO chama evaluate() e NAO carrega NENHUM artefato `test*`
(a trava de siamese.protocol bloqueia fisicamente). O melhor combo deve ser reavaliado UMA vez
no TESTE com `scripts/evaluate.py --final-test`, depois de congelado.

[Antes: ranqueava por `sintetico_livre_de_confound` calculado em evaluate() a partir de
test_synth.npz + as limpas de TESTE -> a selecao enxergava o teste (snooping). Corrigido.]

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
# NB: NAO importamos evaluate aqui — a selecao usa SO metricas de val de train_head (anti-snooping).

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
    # sonda livre de confound SO da VAL (de processed/val/real/clean). O grid NUNCA gera nem le
    # test_synth: a selecao e' so na val (anti-snooping). test_synth so e' criado no --final-test.
    if cfg.synthetic.enabled:
        d = processed / "val" / "real" / "clean"
        rows = [{"path": str(p.resolve())} for p in sorted(d.iterdir()) if p.suffix.lower() in _EXTS]
        if rows:
            extract_synthetic(None, emb / "val_synth.npz", bb,
                              n_variants=cfg.synthetic.n_variants,
                              max_errors_per_image=cfg.synthetic.max_errors_per_image,
                              seed=cfg.synthetic.seed + 100, batch_size=cfg.backbone.batch_size,
                              multiclass=cfg.train.multiclass, clean_rows=rows)


def _metrics(res: dict) -> dict:
    """Extrai as metricas-chave de SELECAO do retorno de train_head() (SO val — anti-snooping)."""
    bm = res.get("best_metrics", {})
    def g(k):
        return float(bm.get(k, float("nan")))
    return {
        "val_synth_gate": g("val_synth_gate"),          # <- metrica de selecao honesta (gate livre de confound)
        "val_synth_gate_proto": g("val_synth_gate_proto"),
        "val_ap_aux": g("val_ap_aux"),
        "val_ap": g("val_ap"),                          # gate confundido (legado, p/ referencia)
        "val_cat_f1": g("val_cat_f1"),                  # clusterizacao por categoria (Estagio 2)
        "best_sel": float(res.get("best_sel", float("nan"))),
        "best_epoch": int(res.get("best_epoch", -1)),
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
    # SELECAO: so train_head -> metricas de VAL. NUNCA evaluate()/teste (anti-snooping).
    res = train_head(cfg, device=DEVICE)
    return _metrics(res)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--grid", type=str, default=None, help="JSON {chave.dotted: [valores]}")
    ap.add_argument("--rank-by", default="val_synth_gate",
                    choices=["val_synth_gate", "val_synth_gate_proto", "val_ap_aux", "val_ap",
                             "val_cat_f1", "best_sel"],
                    help="metrica de VAL para selecao (padrao: val_synth_gate -- gate livre de confound)")
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
    print(f"  selecao por: {args.rank_by} (metrica de VAL; teste NUNCA tocado)\n")

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
        print(f"  [{i+1}/{len(combos)}] {combo} -> val_synth_gate={m['val_synth_gate']:.3f} "
              f"val_ap_aux={m['val_ap_aux']:.3f} val_cat_f1={m['val_cat_f1']:.3f} ({m_row['secs']}s)")

    if not rows:
        print("Nenhum combo concluido."); return

    rows.sort(key=lambda r: (-(r[args.rank_by] if r[args.rank_by] == r[args.rank_by] else -1)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = keys + ["val_synth_gate", "val_synth_gate_proto", "val_ap_aux", "val_ap",
                   "val_cat_f1", "best_sel", "best_epoch", "secs"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"\n========= RANKING (top, por {args.rank_by}) =========")
    head = "  " + "  ".join(f"{k.split('.')[-1]:>10s}" for k in keys) + \
           f"  {'synthGate':>9s} {'ap_aux':>7s} {'cat_F1':>7s} {'sel':>6s}"
    print(head)
    for r in rows[:10]:
        vals = "  ".join(f"{str(r[k]):>10s}" for k in keys)
        print(f"  {vals}  {r['val_synth_gate']:>9.3f} {r['val_ap_aux']:>7.3f} "
              f"{r['val_cat_f1']:>7.3f} {r['best_sel']:>6.3f}")
    best = rows[0]
    print(f"\nMELHOR ({args.rank_by}={best[args.rank_by]:.3f}): "
          f"{ {k: best[k] for k in keys} }")
    print(f"Resultados completos: {args.out}")
    print("NB: selecao SO por metrica de VALIDACAO (anti-snooping; o teste nao foi lido).")
    print("    Congele a config vencedora e reavalie no TESTE UMA vez:")
    print("    python scripts/evaluate.py --config <vencedora> --final-test")


if __name__ == "__main__":
    main()
