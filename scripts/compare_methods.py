#!/usr/bin/env python
"""Comparação empírica DEV (val, livre de confound) — responde às perguntas da supervisão:
SupCon vs Triplet · gate protótipo vs k-NN · Estágio 2 protótipo vs k-NN.

NÃO toca o teste held-out (tudo em modo DEV / validação; artefatos em diretórios temporários).
Treina UM modelo por função de perda e avalia cada combinação de decisão (barata, sobre o mesmo z).

Uso:
    python scripts/compare_methods.py --config configs/default.yaml
    python scripts/compare_methods.py --config configs/default.yaml --final-test-best   # roda o
        vencedor no teste held-out 1× (só depois de inspecionar a tabela de validação)
"""
from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

from siamese.config import Config
from siamese.train import train_head
from siamese.evaluate import evaluate

# (loss, margin) a treinar; gate × stage2 a avaliar sobre cada modelo
TRAININGS = [("supcon", None), ("triplet", 0.2), ("triplet", 0.5)]
GATES = ["prototype", "knn"]
STAGE2 = ["prototype", "knn"]


def _collect(rep: dict) -> dict:
    s = rep.get("sintetico_livre_de_confound", {})
    op = rep.get("ponto_operacao", {})
    e2 = rep.get("estagio2_categoria", {}).get("oraculo", {})
    return {
        "synth_auroc": s.get("modelo_proto", {}).get("auroc"),
        "spec": op.get("especificidade"), "acc": op.get("acuracia"),
        "prec": op.get("precisao"), "recall": op.get("recall"), "f1": op.get("f1"),
        "e2_grossa": e2.get("grossa", {}).get("f1_macro"),
        "e2_fina_proto": e2.get("fina", {}).get("por_prototipo", {}).get("f1_macro"),
        "e2_fina_knn": e2.get("fina", {}).get("por_knn", {}).get("f1_macro"),
    }


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "  —  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--final-test-best", action="store_true",
                    help="após a tabela DEV, roda o vencedor (maior synth_auroc) no TESTE held-out 1×")
    args = ap.parse_args()
    base = Config.load(args.config)
    out_dir = Path(base.paths.reports_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="cmp_methods_"))
    for loss, margin in TRAININGS:
        cfg = copy.deepcopy(base)
        cfg.train.loss = loss
        if margin is not None:
            cfg.train.triplet_margin = margin
        mdir = tmp / f"{loss}_{margin}"; mdir.mkdir(parents=True, exist_ok=True)
        cfg.paths.models_dir = str(mdir)
        tag = f"{loss}" + (f"(m={margin})" if margin is not None else "")
        print(f"\n=== treinando {tag} ===")
        info = train_head(cfg)
        print(f"    early-stop {info['sel_name']}={info['best_sel']:.3f} @ep{info['best_epoch']}")
        for gate in GATES:
            for s2 in STAGE2:
                ev = copy.deepcopy(cfg)
                ev.decision.gate_method = gate
                ev.decision.stage2_method = s2
                ev.paths.reports_dir = str(mdir / f"rep_{gate}_{s2}")
                Path(ev.paths.reports_dir).mkdir(parents=True, exist_ok=True)
                rep = evaluate(ev, final_test=False)   # DEV (val) — NÃO toca teste
                m = _collect(rep)
                rows.append({"loss": tag, "gate": gate, "stage2": s2, **m})

    # tabela
    hdr = f"{'loss':14s} {'gate':10s} {'stage2':10s} | {'synthAUROC':>10s} {'spec':>6s} {'acc':>6s} {'prec':>6s} {'f1':>6s} | {'E2grossa':>8s} {'E2fina(p/k)':>12s}"
    print("\n" + "=" * len(hdr)); print(" COMPARAÇÃO (VALIDAÇÃO livre de confound — teste NÃO tocado)"); print("=" * len(hdr))
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['loss']:14s} {r['gate']:10s} {r['stage2']:10s} | {_fmt(r['synth_auroc']):>10s} "
              f"{_fmt(r['spec']):>6s} {_fmt(r['acc']):>6s} {_fmt(r['prec']):>6s} {_fmt(r['f1']):>6s} | "
              f"{_fmt(r['e2_grossa']):>8s} {_fmt(r['e2_fina_proto'])}/{_fmt(r['e2_fina_knn'])}")
    print("-" * len(hdr))
    print(" synthAUROC = gate livre de confound na VAL (métrica primária; acaso 0.50)")
    print(" spec/acc/prec/f1 = ponto de operação na VAL (in-sample; comparativo relativo)")
    print(" E2grossa = F1-macro do método canônico; E2fina(p/k) = fina por protótipo / por k-NN")

    best = max(rows, key=lambda r: (r["synth_auroc"] or 0))
    print(f"\n VENCEDOR (val synthAUROC): {best['loss']} · gate={best['gate']} · stage2={best['stage2']} "
          f"-> {_fmt(best['synth_auroc'])}")
    base_row = next(r for r in rows if r["loss"].startswith("supcon") and r["gate"] == "prototype" and r["stage2"] == "prototype")
    print(f" BASELINE (supcon·proto·proto): synthAUROC {_fmt(base_row['synth_auroc'])} · "
          f"spec {_fmt(base_row['spec'])} · E2grossa {_fmt(base_row['e2_grossa'])}")
    (out_dir / "compare_methods.json").write_text(json.dumps(
        {"rows": rows, "winner": best, "baseline": base_row}, indent=2, ensure_ascii=False))
    print(f"\n Tabela salva: {out_dir/'compare_methods.json'}")
    if args.final_test_best:
        print("\n (--final-test-best): rode o vencedor com scripts/run_experiment.py usando uma config "
              "que fixe loss/gate/stage2 do vencedor, p/ o teste held-out 1× — não automatizado aqui "
              "para manter a trava explícita.)")


if __name__ == "__main__":
    main()
