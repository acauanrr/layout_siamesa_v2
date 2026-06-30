#!/usr/bin/env python
"""Avaliacao do TESTE held-out RESTRITA a um DOMINIO (fatia) — isola a metrica honesta no
subconjunto que importa (ex.: foldable near-square), sem deixar clean diversas mascararem o numero.

POR QUE (lição do cross-eval + #1): o headline (free-confound 0.80 no plus-test) e' puxado por clean
DIVERSAS; no dominio de PRODUCAO (foldable near-square) o numero e' outro (free-confound 0.73, espec
0.51). Esta ferramenta mede a fatia que importa — e' o **criterio de aceite #1/#2** da
`docs/SPEC_COLETA_FOLDABLE.md` (a especificidade foldable TEM que sair de 0.512).

COMO (sem reimplementar scoring): constroi um emb_dir filtrado (train/val/sondas-de-treino
SYMLINKADOS -> MODELO byte-identico) e roda o MESMO `siamese.evaluate.evaluate(final_test=True)`
auditado, so trocando o test set. Self-check: roda tambem no held-out CHEIO e confere que reproduz
o evaluation_report.json (garante que a fatia e' confiavel).

Fatias (--subset):
  full          held-out inteiro (so o self-check)
  near-square   AR(w/h) da imagem real em [--ar-lo, --ar-hi] (default 0.85..1.18) -> dominio foldable
  form-factor   form_factor da imagem real in --form-factors (ex.: unfold,fold,tent,laptop)
  v3test        imagem real pertence ao split test de --ref-dataset (default data/processed_v3)

So toca o TESTE -> exige config CONGELADA (mesma semantica do evaluate --final-test; UMA vez).

Uso:
  python scripts/domain_slice_eval.py --config configs/plus_L_reg4.yaml --subset near-square
  python scripts/domain_slice_eval.py --config configs/<fold>.yaml --subset form-factor \
         --form-factors unfold,fold,tent,laptop
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from siamese.config import Config            # noqa: E402
from siamese.evaluate import evaluate        # noqa: E402
from siamese.protocol import allow_test_access  # noqa: E402

# arquivos que DEFINEM o modelo (prototipos do train, calibracao na val) -> symlink, nunca filtra
UNCHANGED = ["train.npz", "val.npz", "train_synth.npz", "val_synth.npz",
             "train_reflow.npz", "val_reflow.npz"]


def _aspect(path: str) -> float:
    try:
        with Image.open(path) as im:
            w, h = im.size
        return (w / h) if h else 0.0
    except Exception:
        return 0.0


def build_predicate(test: dict, args):
    """Devolve mascara booleana sobre as linhas REAIS do test (clean+erro) que caem na fatia."""
    paths = test["path"]
    if args.subset == "full":
        return np.ones(len(paths), dtype=bool)
    if args.subset == "near-square":
        ar = np.array([_aspect(p) for p in paths])
        return (ar >= args.ar_lo) & (ar <= args.ar_hi)
    if args.subset == "form-factor":
        want = {s.strip() for s in args.form_factors.split(",") if s.strip()}
        ff = test["form_factor"] if "form_factor" in test else np.array([""] * len(paths))
        return np.array([f in want for f in ff])
    if args.subset == "v3test":
        ref = Path(args.ref_dataset) / "labels.csv"
        if not ref.exists():
            sys.exit(f"--ref-dataset sem labels.csv: {ref}")
        members = set()
        with open(ref) as f:
            for r in csv.DictReader(f):
                if r["split"] == "test":
                    members.add(Path(r["path"]).name)
        return np.array([Path(p).name in members for p in paths])
    sys.exit(f"--subset desconhecido: {args.subset}")


def build_emb_dir(dst: Path, src: Path, keep_real: np.ndarray | None):
    """Monta o emb_dir da avaliacao. keep_real=None -> held-out cheio (tudo symlink)."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for fn in UNCHANGED:
        os.symlink((src / fn).resolve(), dst / fn)
    test = np.load(src / "test.npz", allow_pickle=True)
    if keep_real is None:
        for fn in ["test.npz", "test_synth.npz", "test_reflow.npz"]:
            os.symlink((src / fn).resolve(), dst / fn)
        return int((test["label"] == 0).sum()), int(len(test["label"]))

    np.savez(dst / "test.npz", **{k: test[k][keep_real] for k in test.files})
    # parent das sondas synth/reflow indexa o array CLEAN-ONLY do test original (ordem preservada).
    # mantem so as sondas cuja clean-mae caiu na fatia; remapeia o parent p/ a nova indexacao clean.
    clean_global = np.where(test["label"] == 0)[0]               # idx global de cada clean (em ordem)
    old2new, nn = {}, 0
    for i, gidx in enumerate(clean_global):
        if keep_real[gidx]:
            old2new[i] = nn; nn += 1
    for fn in ["test_synth.npz", "test_reflow.npz"]:
        p = src / fn
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        kp = np.array([int(x) in old2new for x in d["parent"]], dtype=bool)
        out = {k: d[k][kp] for k in d.files}
        out["parent"] = np.array([old2new[int(x)] for x in d["parent"][kp]], dtype=d["parent"].dtype)
        np.savez(dst / fn, **out)
    sub = np.load(dst / "test.npz", allow_pickle=True)
    return int((sub["label"] == 0).sum()), int(len(sub["label"]))


def run(cfg_path: Path, emb_dir: Path, src: Path, rep_dir: Path) -> dict:
    cfg = Config.load(cfg_path)
    cfg.paths.emb_dir = str(emb_dir)
    cfg.paths.models_dir = str((src / "models").resolve())   # head treinado (NAO muda)
    cfg.paths.reports_dir = str(rep_dir)
    allow_test_access(True)                                  # fatia do held-out (config congelada)
    return evaluate(cfg, final_test=True)


def grab(rep: dict) -> dict:
    s = rep.get("sintetico_livre_de_confound", {})
    op = rep.get("ponto_operacao", {})
    glob = rep.get("global_vs_baselines", {})
    e2 = rep.get("estagio2_categoria", {}).get("oraculo", {})
    g = lambda d, *k: (d.get(k[0], {}) or {}).get(k[1]) if len(k) == 2 else d.get(k[0])
    return {
        "n_test": rep.get("n_test"),
        "free_confound_AUROC": g(s, "modelo_proto", "auroc"),
        "free_confound_n_clean": s.get("n_clean"), "free_confound_n_synth": s.get("n_synth"),
        "gate_proto_AUROC": g(glob, "modelo_proto", "auroc"),
        "gate_fusao_AUROC": g(glob, "modelo_fusao", "auroc"),
        "baseline_resolucao_trivial": g(glob, "baseline_resolucao_trivial", "auroc"),
        "especificidade": op.get("especificidade"), "acuracia": op.get("acuracia"),
        "precisao": op.get("precisao"), "recall": op.get("recall"),
        "bAcc": op.get("balanced_accuracy"), "confusao": op.get("confusao"),
        "coarse_F1": g(e2, "grossa", "f1_macro"),
        "fine_F1": (e2.get("fina", {}).get("por_prototipo", {}) or {}).get("f1_macro"),
    }


def _f(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--subset", default="near-square",
                    choices=["full", "near-square", "form-factor", "v3test"])
    ap.add_argument("--ar-lo", type=float, default=0.85)
    ap.add_argument("--ar-hi", type=float, default=1.18)
    ap.add_argument("--form-factors", default="unfold,fold,tent,laptop")
    ap.add_argument("--ref-dataset", type=Path, default=Path("data/processed_v3"))
    ap.add_argument("--out", type=Path, default=None, help="JSON do resultado (default: <reports_dir>/domain_slice_<subset>.json)")
    args = ap.parse_args()

    cfg0 = Config.load(args.config)
    src = Path(cfg0.paths.emb_dir)
    if not (src / "test.npz").exists():
        sys.exit(f"emb_dir sem test.npz: {src} (rode run_experiment.py antes)")
    import tempfile
    work = Path(tempfile.mkdtemp(prefix="domain_slice_"))   # emb_dirs filtrados (scratch, limpo no fim)

    test = np.load(src / "test.npz", allow_pickle=True)
    keep = build_predicate(test, args)
    is_clean = test["label"] == 0
    n_clean_slice = int((keep & is_clean).sum())
    n_err_slice = int((keep & ~is_clean).sum())
    print(f"[slice={args.subset}] {int(keep.sum())} imgs (clean={n_clean_slice}, erro={n_err_slice}) "
          f"de {len(keep)} no held-out cheio")
    if n_clean_slice == 0 or len(np.unique(test["label"][keep])) < 2:
        sys.exit(f"fatia degenerada (clean={n_clean_slice}, classes={len(np.unique(test['label'][keep]))}) "
                 f"-> escolha outro --subset ou confira form_factor/resolucoes.")

    # (a) self-check no held-out CHEIO  (b) fatia do dominio
    nc_f, na_f = build_emb_dir(work / "emb_full", src, None)
    full = grab(run(args.config, work / "emb_full", src, work / "rep_full"))
    nc_s, na_s = build_emb_dir(work / "emb_slice", src, keep)
    sli = grab(run(args.config, work / "emb_slice", src, work / "rep_slice"))
    shutil.rmtree(work, ignore_errors=True)   # metricas ja em full/sli -> limpa o scratch

    # decomposicao da especificidade: clean DENTRO vs FORA da fatia (do held-out cheio)
    cf, cs = full["confusao"] or {}, sli["confusao"] or {}
    tn_full, fp_full = cf.get("TN", 0), cf.get("FP", 0)
    tn_sl, fp_sl = cs.get("TN", 0), cs.get("FP", 0)
    n_in, n_out = nc_s, nc_f - nc_s
    sp_out = (tn_full - tn_sl) / n_out if n_out > 0 else float("nan")

    print("\n" + "=" * 78)
    print(f" DOMAIN-SLICE EVAL  |  config={args.config.name}  subset={args.subset}")
    print("=" * 78)
    hdr = f"{'metrica':26s} {'held-out CHEIO':>16s} {'FATIA('+args.subset+')':>18s}"
    print(hdr); print("-" * len(hdr))
    for k, lab in [("n_test", "n (clean+erro)"), ("free_confound_AUROC", "free-confound AUROC ⭐"),
                   ("gate_proto_AUROC", "gate proto AUROC"), ("gate_fusao_AUROC", "gate fusao AUROC"),
                   ("especificidade", "especificidade ⭐"), ("acuracia", "acuracia"),
                   ("precisao", "precisao"), ("recall", "recall"), ("bAcc", "bAcc"),
                   ("baseline_resolucao_trivial", "confound trivial"),
                   ("coarse_F1", "coarse F1"), ("fine_F1", "fine F1")]:
        print(f"{lab:26s} {_f(full[k]):>16s} {_f(sli[k]):>18s}")
    print("-" * len(hdr))
    print(f"{'confusao (TP/TN/FP/FN)':26s} "
          f"{str([cf.get(x) for x in ('TP','TN','FP','FN')]):>16s} "
          f"{str([cs.get(x) for x in ('TP','TN','FP','FN')]):>18s}")
    print(f"\n especificidade por subpop (clean): DENTRO da fatia={_f(sli['especificidade'])} "
          f"(TN {tn_sl}/{n_in})  |  FORA={_f(sp_out)} (TN {tn_full-tn_sl}/{n_out})")
    print(" ⭐ = criterios de aceite da SPEC_COLETA_FOLDABLE.md (a fatia e' o numero honesto do dominio).")
    print("=" * 78)

    out = args.out or (Path(cfg0.paths.reports_dir) / f"domain_slice_{args.subset}.json")
    out.write_text(json.dumps({"subset": args.subset, "held_out_full": full, "slice": sli,
                               "especificidade_fora_da_fatia": sp_out}, indent=2))
    print(f" JSON: {out}")


if __name__ == "__main__":
    main()
