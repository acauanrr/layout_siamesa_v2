#!/usr/bin/env python
"""Grouped Nested CV + calibracao de limiar OUT-OF-FOLD (Fase 2). SO dados de desenvolvimento
(train+val reais + sinteticos de treino); o TESTE externo NUNCA e' tocado (trava de protocol).

Monta o pool ancorando cada SINTETICO ao grupo de SESSAO da sua imagem-mae (sem vazamento entre
folds), roda OOF sob N seeds, calibra o limiar nas predicoes OOF (REAIS) e reporta a taxonomia:
  - RESSUBSTITUICAO: in-sample (so diagnostico de capacidade/overfitting).
  - OOF (estimativa interna de generalizacao): metrica real, com IC95 agrupado, media+-SE entre seeds.
  - TESTE EXTERNO: continua reservado ao --final-test (1x), nao calculado aqui.

Uso: python scripts/nested_cv.py [--folds 5] [--seeds 5] [--objective f1|precision]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, balanced_accuracy_score, matthews_corrcoef, brier_score_loss)

from siamese.config import Config
from siamese.features import load_embeddings
from siamese.manifest import category_id, CATEGORIES
from siamese.cv import run_oof
from siamese.decision import select_threshold_max_f1, select_threshold_for_precision
from siamese.evaluate import grouped_bootstrap_ci, _expected_calibration_error


def _build_pool(emb_dir: Path):
    """Pool de desenvolvimento = train(real) + val(real) + train_synth, com o GRUPO de cada
    sintetico = grupo de sessao da imagem-mae (parent). Devolve X, ycat, label, groups, is_real."""
    tr = load_embeddings(emb_dir / "train.npz")
    va = load_embeddings(emb_dir / "val.npz")
    parts = [(tr, True), (va, True)]
    # stem da imagem limpa -> grupo de sessao (das reais train+val)
    stem2group = {}
    for z, _ in parts:
        for p, g, lab in zip(z["path"], z["group"], z["label"]):
            if int(lab) == 0:
                stem2group[Path(str(p)).stem] = str(g)
    syn_path = emb_dir / "train_synth.npz"
    syn = load_embeddings(syn_path) if syn_path.exists() else None

    Xs, ys, labs, grps, real = [], [], [], [], []
    for z, is_real in parts:
        Xs.append(z["emb"])
        ys.append(np.array([category_id(str(c)) for c in z["category"]], dtype=np.int64))
        labs.append(z["label"].astype(np.int64))
        grps.append(np.array([str(g) for g in z["group"]]))
        real.append(np.full(len(z["label"]), is_real))
    if syn is not None:
        Xs.append(syn["emb"])
        ys.append(np.array([category_id(str(c)) for c in syn["category"]], dtype=np.int64))
        labs.append(syn["label"].astype(np.int64))
        # ancora ao grupo da mae; orfaos (nao deveriam existir) viram grupo proprio
        sg = [stem2group.get(str(par), f"synthorphan:{i}") for i, par in enumerate(syn["parent"])]
        grps.append(np.array(sg))
        real.append(np.zeros(len(syn["label"]), dtype=bool))
    return (np.concatenate(Xs).astype(np.float32), np.concatenate(ys), np.concatenate(labs),
            np.concatenate(grps), np.concatenate(real))


def _metrics(y, score, thr, groups):
    pred = (score > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auroc": float(roc_auc_score(y, score)), "ap": float(average_precision_score(y, score)),
        "acuracia": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "precisao": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "especificidade": float(tn / (tn + fp)) if tn + fp else 0.0,
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "brier": float(brier_score_loss(y, score)), "ece": _expected_calibration_error(y, score),
        "confusao": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--objective", choices=["f1", "precision"], default=None)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    objective = args.objective or cfg.decision.objective
    num_classes = len(CATEGORIES) if cfg.train.multiclass else 1

    X, ycat, label, groups, is_real = _build_pool(Path(cfg.paths.emb_dir))
    print(f"Pool de desenvolvimento: {len(X)} itens ({int(is_real.sum())} reais + "
          f"{int((~is_real).sum())} sinteticos) | grupos unicos: {len(set(groups))} | "
          f"{args.folds} folds x {args.seeds} seeds | objetivo limiar: {objective}")

    per_seed, thrs = [], []
    s_first = None
    for sd in range(args.seeds):
        oof = run_oof(X, ycat, label, groups, cfg=cfg, seed=sd, n_splits=args.folds,
                      num_classes=num_classes, device=args.device,
                      epochs=args.epochs, patience=args.patience)
        s = oof["fused"][is_real]; y = label[is_real]; g = groups[is_real]
        if s_first is None:
            s_first = s
        if objective == "precision":
            thr, _ = select_threshold_for_precision(s, y, cfg.decision.target_precision)
        else:
            thr, _ = select_threshold_max_f1(s, y)
        m = _metrics(y, s, thr, g)
        m["threshold"] = float(thr)
        per_seed.append(m); thrs.append(float(thr))
        print(f"  seed {sd}: OOF AUROC={m['auroc']:.3f} AP={m['ap']:.3f} F1={m['f1']:.3f} "
              f"bAcc={m['balanced_accuracy']:.3f} MCC={m['mcc']:.3f} espec={m['especificidade']:.3f} "
              f"(thr={thr:.3f})")

    # agregado multi-seed: media +- SE
    keys = ["auroc", "ap", "f1", "balanced_accuracy", "mcc", "precisao", "recall",
            "especificidade", "fpr", "brier", "ece"]
    agg = {}
    for k in keys:
        v = np.array([m[k] for m in per_seed], dtype=float)
        agg[k] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                  "min": float(v.min()), "max": float(v.max())}
    thr_mean, thr_se = float(np.mean(thrs)), float(np.std(thrs, ddof=1) / np.sqrt(len(thrs)) if len(thrs) > 1 else 0.0)

    # IC95 agrupado da AUROC OOF (reusa a 1a seed; o limiar reportado e' a media entre seeds)
    ci_auroc = grouped_bootstrap_ci(roc_auc_score, label[is_real], s_first, groups[is_real])

    report = {
        "_modo": "OOF (Grouped Nested CV — estimativa interna de generalizacao; teste NAO tocado)",
        "n_real": int(is_real.sum()), "n_synth": int((~is_real).sum()),
        "folds": args.folds, "seeds": args.seeds, "objetivo_limiar": objective,
        "limiar_oof": {"mean": thr_mean, "se": thr_se},
        "oof_metrics_mean_se": agg,
        "ci95_auroc_oof_agrupado": list(ci_auroc),
        "por_seed": per_seed,
        "nota": ("Limiar calibrado em predicoes OUT-OF-FOLD (nao in-sample) -> robusto. Compare "
                 "com o ponto de operacao calibrado na val (instavel). RESSUBSTITUICAO e TESTE "
                 "EXTERNO sao reportados a parte (train in-sample / --final-test)."),
    }
    out = Path(cfg.paths.reports_dir) / "nested_cv_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=========== OOF (Grouped Nested CV) — media +- SE entre seeds ===========")
    for k in keys:
        a = agg[k]
        print(f"  {k:18s} {a['mean']:.3f} +- {a['se']:.3f}   (min {a['min']:.3f} / max {a['max']:.3f})")
    print(f"  limiar OOF          {thr_mean:.3f} +- {thr_se:.3f}")
    print(f"  IC95 AUROC OOF (agrupado por sessao/ticket): ({ci_auroc[0]:.3f}, {ci_auroc[1]:.3f})")
    print(f"\nRelatorio: {out}")
    print("NB: OOF = estimativa interna honesta; o TESTE externo nao foi tocado (use --final-test 1x).")


if __name__ == "__main__":
    main()
