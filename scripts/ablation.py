#!/usr/bin/env python
"""Ablacao: efeito de treinar com erros REAIS e/ou SINTETICOS.

Mostra quantitativamente que os erros sinteticos sao a alavanca anti-confound:
o modelo treinado SO com sinteticos nao consegue aprender o confound de resolucao
(porque nunca ve a resolucao pareada com a label de erro), mas ainda detecta o
CONTEUDO do erro (deteccao sintetica livre de confound).

Uso:
    python scripts/ablation.py --config configs/patchstats.yaml
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from siamese.config import Config
from siamese.train import train_head
from siamese.evaluate import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = ap.parse_args()
    base = Config.load(args.config)

    combos = [
        ("real+sintetico", True, True),
        ("so sintetico",   False, True),
        ("so real",        True, False),
    ]
    rows = []
    for name, use_real, use_synth in combos:
        cfg = copy.deepcopy(base)
        cfg.train.use_real_errors = use_real
        cfg.train.use_synthetic = use_synth
        print(f"\n########## {name} (real={use_real}, synth={use_synth}) ##########")
        train_head(cfg)
        rep = evaluate(cfg)   # modo DEV (val); o teste NAO e' tocado (anti-snooping)
        syn = rep["sintetico_livre_de_confound"].get("modelo_fusao", {"auroc": float("nan"), "ap": float("nan")})
        glob = rep["global_vs_baselines"]["modelo_fusao"]
        ctrl = rep.get("primaria_subconjunto_controlado", {}).get("modelo_fusao", {"auroc": float("nan")})
        f = rep["falseabilidade"]
        rows.append({
            "treino": name,
            "sintetico_AUROC": syn["auroc"], "sintetico_AP": syn["ap"],
            "global_AUROC": glob["auroc"],
            "controlado_AUROC": ctrl["auroc"],
            "prediz_resolucao_AUROC": f.get("auroc_modelo_predizendo_resolucao", float("nan")),
            "prediz_erro_AUROC": f.get("auroc_modelo_predizendo_erro", float("nan")),
        })

    print("\n\n================= TABELA DE ABLACAO =================")
    hdr = ["treino", "sintetico_AUROC", "sintetico_AP", "global_AUROC",
           "controlado_AUROC", "prediz_resolucao_AUROC", "prediz_erro_AUROC"]
    print(f"{'treino':16s} {'synt_AUROC':>10s} {'synt_AP':>8s} {'glob_AUROC':>10s} "
          f"{'ctrl_AUROC':>10s} {'->resol':>8s} {'->erro':>7s}")
    for r in rows:
        print(f"{r['treino']:16s} {r['sintetico_AUROC']:10.3f} {r['sintetico_AP']:8.3f} "
              f"{r['global_AUROC']:10.3f} {r['controlado_AUROC']:10.3f} "
              f"{r['prediz_resolucao_AUROC']:8.3f} {r['prediz_erro_AUROC']:7.3f}")
    print("\nLeitura: 'so sintetico' deve detectar conteudo (synt_AUROC alto) SEM rastrear")
    print("resolucao (->resol baixo). 'so real' tende a rastrear resolucao (->resol alto =")
    print("->erro), evidenciando que aprende o confound, nao o layout.")


if __name__ == "__main__":
    main()
