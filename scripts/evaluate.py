#!/usr/bin/env python
"""Avaliacao honesta do detector siames (metrica controlada + baselines + falseabilidade).

Uso:
    python scripts/evaluate.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from siamese.config import Config
from siamese.evaluate import evaluate


def _fmt(d):
    return f"AUROC={d['auroc']:.3f} AP={d['ap']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = ap.parse_args()
    cfg = Config.load(args.config)
    rep = evaluate(cfg)

    print("\n================ AVALIACAO HONESTA ================")
    print(f"Test: {rep['n_test']} imagens")
    op = rep.get("ponto_operacao", {})
    if op:
        ci = op.get("ci95_acuracia", [float('nan'), float('nan')])
        print(f"\n>>> PONTO DE OPERACAO ({op['objetivo']}, padrao) — NUMEROS PRINCIPAIS <<<")
        print(f"    Acuracia={op['acuracia']:.3f} (IC95 {ci[0]:.2f}-{ci[1]:.2f})  Precisao={op['precisao']:.3f}  "
              f"Recall={op['recall']:.3f}  F1={op['f1']:.3f}")
        print(f"    AUROC={op['auroc']:.3f}  AP={op['ap']:.3f}  | confusao={op['confusao']}")
    print("\n--- 1) GLOBAL: modelo vs baselines de CONFOUND (cuidado: global e confundido) ---")
    for k, v in rep["global_vs_baselines"].items():
        print(f"  {k:34s} {_fmt(v)}")

    if "primaria_subconjunto_controlado" in rep:
        c = rep["primaria_subconjunto_controlado"]
        print(f"\n--- 2) PRIMARIA: subconjunto controlado unfold-portrait-screenshot (n={c['n']}, erro={c['n_erro']}) ---")
        print(f"  modelo_fusao     {_fmt(c['modelo_fusao'])}  IC95 AUROC={tuple(round(x,3) for x in c['ci95_fusao_auroc'])}")
        print(f"  baseline_confound{_fmt(c['baseline_confound'])}")
        print("  -> o modelo so tem valor se SUPERAR o baseline de confound aqui.")

    s = rep["sintetico_livre_de_confound"]
    print(f"\n--- 3) SINTETICO livre de confound (clean={s['n_clean']} vs synth={s['n_synth']}) ---")
    print(f"  modelo_fusao {_fmt(s['modelo_fusao'])}  modelo_proto {_fmt(s['modelo_proto'])}")

    a = rep["auditoria_same_resolution"]
    print(f"\n--- 4) AUDITORIA same-resolution (erros reais 2076x2152, held-out): n={a['n']} ---")
    for it in a["itens"]:
        flag = "INDEP" if it["independente"] else "sessao"
        print(f"  [{flag}] {it['file'][:50]:50s} fused={it['fused']:.3f}")

    f = rep["falseabilidade"]
    print("\n--- 5) FALSEABILIDADE ---")
    if "auroc_modelo_predizendo_resolucao" in f:
        print(f"  modelo prediz RESOLUCAO: AUROC={f['auroc_modelo_predizendo_resolucao']:.3f} | "
              f"prediz ERRO: AUROC={f['auroc_modelo_predizendo_erro']:.3f}")
    print(f"  label-shuffle no estrato (deveria ~0.5): AUROC={f['auroc_label_shuffle_no_estrato']:.3f}")

    print("\n--- 6) LIMIAR por precisao-alvo (fixado na VAL, medido no TEST) ---")
    for k, v in rep["limiar_por_precisao"].items():
        print(f"  {k}: test_precision={v['test_precision']:.3f} test_recall={v['test_recall']:.3f} "
              f"(tp={v['test_tp']} fp={v['test_fp']} fn={v['test_fn']})")
    print(f"\n  precision@K: {rep['precision_at_k']}")
    ci = rep["ci95_global_auroc_fusao"]
    print(f"  IC95 AUROC global (fusao): ({ci[0]:.3f}, {ci[1]:.3f})")
    print(f"\nRelatorio completo: {cfg.paths.reports_dir}/evaluation_report.json")
    print(f"Graficos: {cfg.paths.reports_dir}/evaluation_plots.png")


if __name__ == "__main__":
    main()
