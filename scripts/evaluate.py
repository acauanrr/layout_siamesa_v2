#!/usr/bin/env python
"""Avaliacao honesta do detector siames (metrica controlada + baselines + falseabilidade).

Por PADRAO roda em modo DEV: reporta sobre a VALIDACAO e NAO toca o teste (seguro p/ iterar).
O TESTE held-out so e' avaliado com --final-test, que destrava o acesso UMA vez — use somente
apos congelar arquitetura/pesos/limiares (criterio de aceite #6: teste processado uma so vez).

Uso:
    python scripts/evaluate.py --config configs/default.yaml                # DEV (val)
    python scripts/evaluate.py --config configs/default.yaml --final-test   # TESTE (1x, final)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from siamese.config import Config
from siamese.evaluate import evaluate
from siamese.protocol import allow_test_access


def _fmt(d):
    return f"AUROC={d['auroc']:.3f} AP={d['ap']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--final-test", action="store_true",
                    help="DESTRAVA e avalia no TESTE held-out (uma unica vez, apos congelar a config). "
                         "Sem esta flag, reporta sobre a VALIDACAO (modo DEV).")
    args = ap.parse_args()
    cfg = Config.load(args.config)

    if args.final_test:
        print("\n*** --final-test: DESTRAVANDO o conjunto de TESTE held-out (use UMA vez). ***")
        allow_test_access(True)
    rep = evaluate(cfg, final_test=args.final_test)

    print("\n================ AVALIACAO HONESTA ================")
    print(f"MODO: {rep.get('_modo', '?')}  |  held-out reportado: {rep.get('_holdout', '?')}")
    if not args.final_test:
        print("  (DEV: numeros sobre a VAL, in-sample. Os numeros vinculantes so saem com --final-test.)")
    print(f"{rep.get('_holdout','?')}: {rep['n_test']} imagens")
    op = rep.get("ponto_operacao", {})
    optr = rep.get("ponto_operacao_treino", {})
    ho = rep.get("_holdout", "test").upper()        # 'VAL' (dev) ou 'TEST' (final)
    if op:
        ci = op.get("ci95_acuracia", [float('nan'), float('nan')])
        print(f"\n>>> PONTO DE OPERACAO ({op['objetivo']}, limiar fixado na VAL) — TREINO vs {ho} <<<")
        print(f"  {'':6s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'bAcc':>6s} {'MCC':>6s} "
              f"{'AUROC':>6s} {'AP':>6s} {'Brier':>6s}  confusao")
        if optr:
            print(f"  TREINO {optr['acuracia']:6.3f} {optr['precisao']:6.3f} {optr['recall']:6.3f} "
                  f"{optr['f1']:6.3f} {optr['balanced_accuracy']:6.3f} {optr['mcc']:6.3f} "
                  f"{optr['auroc']:6.3f} {optr['ap']:6.3f} {optr['brier']:6.3f}  {optr['confusao']}")
        print(f"  {ho:6s} {op['acuracia']:6.3f} {op['precisao']:6.3f} {op['recall']:6.3f} "
              f"{op['f1']:6.3f} {op['balanced_accuracy']:6.3f} {op['mcc']:6.3f} "
              f"{op['auroc']:6.3f} {op['ap']:6.3f} {op['brier']:6.3f}  {op['confusao']}")
        kind = "held-out (vinculante)" if ho == "TEST" else "in-sample (DEV/val — NAO vinculante)"
        print(f"  ({ho} {kind}: acc IC95 {ci[0]:.2f}-{ci[1]:.2f}; especif={op['especificidade']:.2f} "
              f"FPR={op['fpr']:.2f} ECE={op['ece']:.3f}. Treino e' ressubstituicao (gap/overfitting).)")
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
    if "modelo_fusao" in s:
        print(f"  modelo_fusao {_fmt(s['modelo_fusao'])}  modelo_proto {_fmt(s['modelo_proto'])}")
    else:
        print(f"  {s.get('nota', 'indisponivel')}")

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

    print(f"\n--- 6) LIMIAR por precisao-alvo (fixado na VAL, medido em {ho}) ---")
    for k, v in rep["limiar_por_precisao"].items():
        print(f"  {k}: {ho.lower()}_precision={v['test_precision']:.3f} {ho.lower()}_recall={v['test_recall']:.3f} "
              f"(tp={v['test_tp']} fp={v['test_fp']} fn={v['test_fn']})")
    print(f"\n  precision@K: {rep['precision_at_k']}")
    ci = rep["ci95_global_auroc_fusao"]
    print(f"  IC95 AUROC global (fusao): ({ci[0]:.3f}, {ci[1]:.3f})")

    if "estagio2_categoria" in rep:
        e2 = rep["estagio2_categoria"]; e2tr = rep.get("estagio2_categoria_treino", {})
        print(f"\n--- ESTAGIO 2) CLUSTERIZACAO POR CATEGORIA (TREINO vs {ho}) ---")
        print(f"  classes: {', '.join(e2['classes'])}")
        if e2tr:
            mt = e2tr["por_prototipo"]
            print(f"  TREINO (n_erro={e2tr['n_erro']:3d})  prototipo: acc={mt['accuracy']:.3f}  F1_macro={mt['f1_macro']:.3f} (ressubstituicao — NAO e' resultado)")
        for via in ("por_prototipo", "por_aux_head"):
            m = e2[via]
            print(f"  {ho:6s} (n_erro={e2['n_erro']:3d})  [{via:13s}] acc={m['accuracy']:.3f}  F1_macro={m['f1_macro']:.3f}")
            print(f"                   F1/classe: " +
                  "  ".join(f"{k}={v:.2f}" for k, v in m["f1_por_classe"].items()))
        print(f"  matrizes -> {cfg.paths.reports_dir}/confusion_matrix_categoria{{,_treino}}.png")

    print(f"\nRelatorio completo: {cfg.paths.reports_dir}/evaluation_report.json")
    print(f"Graficos: {cfg.paths.reports_dir}/evaluation_plots.png")


if __name__ == "__main__":
    main()
