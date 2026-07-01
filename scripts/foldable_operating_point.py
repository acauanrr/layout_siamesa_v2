#!/usr/bin/env python
"""Ponto de operação no DOMÍNIO foldable via gate PROTÓTIPO + limiar recalibrado no bucket
(consolida as análises A1/A4 da Fase 2.b "sem dados novos"; ver docs/RELATORIO_FOLDABLE.md).

Motivo: o gate fundido + limiar GLOBAL erram no foldable (clean near-square pontua alto -> espec
~0.51). O gate de PROTÓTIPO separa melhor o foldable, e recalibrar o limiar nas clean foldable da
val dá um ponto de operação melhor — SEM dado novo. Mede com IC95 bootstrap AGRUPADO por ticket
(amostra pequena -> sempre reportar com IC). Scoring 100% via primitivas auditadas
(fit_prototypes / PrototypeDecider), sem reimplementar nada.

Fatia (--subset): v3test (default; as foldable near-square originais) | near-square (AR) | form-factor.
So toca o TESTE -> semântica de --final-test (config congelada, UMA vez).

Uso:
  python scripts/foldable_operating_point.py --config configs/plus_L_reg4.yaml
  python scripts/foldable_operating_point.py --config configs/plus_L_reg4.yaml --target-spec 0.90
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from siamese.config import Config            # noqa: E402
from siamese.features import load_embeddings  # noqa: E402
from siamese.train import load_model          # noqa: E402
from siamese.evaluate import model_embeddings  # noqa: E402
from siamese.decision import fit_prototypes, PrototypeDecider  # noqa: E402
from siamese.protocol import allow_test_access  # noqa: E402
from sklearn.metrics import roc_auc_score      # noqa: E402


def _v3_names(ref: Path, split: str, only_clean: bool) -> set:
    out = set()
    with open(ref / "labels.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == split and (not only_clean or r["category"] == "clean"):
                out.add(Path(r["path"]).name)
    return out


def _fold_mask(npz, args, only_clean=False):
    """Máscara das linhas REAIS na fatia foldable (mesma semântica do domain_slice_eval)."""
    paths = npz["path"]
    if args.subset == "v3test":
        names = _v3_names(Path(args.ref_dataset), args.split, only_clean)   # args.split: 'val'|'test'
        return np.array([Path(p).name in names for p in paths])
    if args.subset == "near-square":
        ar = np.array([_aspect(p) for p in paths])
        m = (ar >= args.ar_lo) & (ar <= args.ar_hi)
        return m & (npz["label"] == 0) if only_clean else m
    if args.subset == "form-factor":
        want = {s.strip() for s in args.form_factors.split(",") if s.strip()}
        ff = npz["form_factor"] if "form_factor" in npz else np.array([""] * len(paths))
        m = np.array([f in want for f in ff])
        return m & (npz["label"] == 0) if only_clean else m
    sys.exit(f"--subset desconhecido: {args.subset}")


def _aspect(p):
    try:
        with Image.open(p) as im:
            w, h = im.size
        return w / h if h else 0.0
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--subset", default="v3test", choices=["v3test", "near-square", "form-factor"])
    ap.add_argument("--ref-dataset", type=Path, default=Path("data/processed_v3"))
    ap.add_argument("--form-factors", default="unfold,fold,tent,laptop")
    ap.add_argument("--ar-lo", type=float, default=0.85); ap.add_argument("--ar-hi", type=float, default=1.18)
    ap.add_argument("--target-spec", type=float, default=0.80, help="especificidade-alvo na val foldable")
    ap.add_argument("--boot", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    emb = Path(cfg.paths.emb_dir)
    allow_test_access(True)
    model = load_model(Path(cfg.paths.models_dir) / "siamese_head.pt", device="cpu")
    tr = load_embeddings(emb / "train.npz"); va = load_embeddings(emb / "val.npz"); te = load_embeddings(emb / "test.npz")
    z_tr, _ = model_embeddings(model, tr["emb"], "cpu")
    z_va, _ = model_embeddings(model, va["emb"], "cpu")
    z_te, _ = model_embeddings(model, te["emb"], "cpu")
    dec = PrototypeDecider(fit_prototypes(z_tr[tr["label"] == 0], k=cfg.decision.k_prototypes, seed=cfg.seed),
                           threshold=0.0, target_precision=cfg.decision.target_precision)
    sp_va, sp_te = dec.scores(z_va), dec.scores(z_te)

    # fatia foldable
    args.split = "test"
    fold = _fold_mask(te, args)
    y, grp, sp = te["label"][fold], te["group"][fold], sp_te[fold]
    args.split = "val"
    fold_val_clean = sp_va[(va["label"] == 0) & _fold_mask(va, args, only_clean=True)]
    allval_clean = sp_va[va["label"] == 0]
    nclean, nerr = int((y == 0).sum()), int((y == 1).sum())
    if nclean == 0 or nerr == 0 or len(fold_val_clean) == 0:
        sys.exit(f"fatia degenerada (clean={nclean}, err={nerr}, val_fold_clean={len(fold_val_clean)})")
    print(f"[{args.subset}] foldable test: clean={nclean} err={nerr} grupos={len(set(grp))} | "
          f"val foldable clean={len(fold_val_clean)} | score=protótipo")

    thr_for = lambda neg, t: float(np.quantile(neg, t))
    op = lambda thr: (float(np.mean(sp[y == 0] < thr)), float(np.mean(sp[y == 1] >= thr)))

    print(f"\nSeparabilidade foldable (proto AUROC) = {roc_auc_score(y, sp):.3f}  <- fronteira (livre de limiar)")
    print("\n=== limiar GLOBAL (clean diversa) vs FOLDABLE (clean foldable da val) — medido no teste foldable ===")
    print(f"{'alvo_spec':>9s} | {'GLOBAL spec/rec':>17s} | {'FOLDABLE spec/rec':>19s}")
    for t in (0.60, 0.70, 0.80, 0.90):
        sg, rg = op(thr_for(allval_clean, t)); sf, rf = op(thr_for(fold_val_clean, t))
        print(f"{t:>9.2f} | {sg:>7.3f} / {rg:<7.3f} | {sf:>8.3f} / {rf:<8.3f}")

    # bootstrap agrupado por ticket no ponto-alvo (limiar foldable)
    thr = thr_for(fold_val_clean, args.target_spec)
    uniq = np.unique(grp); g2 = {g: np.where(grp == g)[0] for g in uniq}; rng = np.random.default_rng(0)
    A, S, R = [], [], []
    for _ in range(args.boot):
        idx = np.concatenate([g2[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        yy, ss = y[idx], sp[idx]
        if len(np.unique(yy)) < 2:
            continue
        A.append(roc_auc_score(yy, ss)); cl, er = ss[yy == 0], ss[yy == 1]
        if len(cl) and len(er):
            S.append(float(np.mean(cl < thr))); R.append(float(np.mean(er >= thr)))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.median(v)), float(np.percentile(v, 97.5))]
    sp_pt, rc_pt = op(thr)
    res = {"config": str(args.config), "subset": args.subset, "target_spec": args.target_spec,
           "n_clean": nclean, "n_err": nerr, "proto_auroc_ci": ci(A),
           "especificidade_ci": ci(S), "especificidade_pt": sp_pt,
           "recall_ci": ci(R), "recall_pt": rc_pt}
    g = lambda c: f"{c[1]:.3f} [{c[0]:.3f}, {c[2]:.3f}]"
    print(f"\n=== PONTO DE OPERAÇÃO (proto + limiar foldable@{args.target_spec}) — IC95 bootstrap agrupado ({args.boot}x) ===")
    print(f"  AUROC          {g(res['proto_auroc_ci'])}")
    print(f"  especificidade {g(res['especificidade_ci'])}  (pt {sp_pt:.3f})")
    print(f"  recall         {g(res['recall_ci'])}  (pt {rc_pt:.3f})")
    out = args.out or (Path(cfg.paths.reports_dir) / f"foldable_op_{args.subset}.json")
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=2))
    print(f"\nJSON: {out}")


if __name__ == "__main__":
    main()
