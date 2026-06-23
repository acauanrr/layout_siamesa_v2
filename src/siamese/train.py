"""Treino da cabeca siamesa sobre embeddings DINOv2 cacheados.

Composicao do treino (configuravel para ablacao):
  - imagens LIMPAS reais (label 0)                          [sempre]
  - erros REAIS (label 1)                                   [train.use_real_errors]
  - erros SINTETICOS injetados nas limpas (label 1)         [train.use_synthetic]  <- anti-confound

Objetivo: L = SupCon(z, y) + aux_weight * BCE(aux_logit, y).
  SupCon molda o espaco metrico (limpo compacto, erro afastado) — o nucleo "siames".
  O cabecalho auxiliar e o detector binario direto de producao.

Como o backbone e congelado e os embeddings cacheados, o treino roda em segundos.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .config import Config


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _file_sha256(p: Path) -> str:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "missing"
from .features import load_embeddings
from .model import SiameseNet
from .losses import supcon_loss, contrastive_loss
from .decision import (fit_prototypes, PrototypeDecider,
                       fit_category_prototypes, assign_category)
from .manifest import category_id, CATEGORIES


def _labels(d: dict, multiclass: bool) -> np.ndarray:
    """Rotulo de treino: id de CATEGORIA (0=clean,1..N) se multiclass, senao binario 0/1.
    Categoria de erro fora do escopo levanta (nunca vira 'clean' silenciosamente — Fase 6)."""
    if multiclass:
        return np.array([category_id(str(c)) for c in d["category"]], dtype=np.int64)
    return d["label"].astype(np.int64)


def assemble_training(emb_dir: Path, use_real_errors: bool, use_synthetic: bool,
                      multiclass: bool = True, use_reflow: bool = False):
    tr = load_embeddings(emb_dir / "train.npz")
    ytr = _labels(tr, multiclass)
    if use_real_errors:
        X, y = [tr["emb"]], [ytr]
    else:  # so as limpas reais (label binario 0)
        m = tr["label"] == 0
        X, y = [tr["emb"][m]], [ytr[m]]
    src = ["real"] * len(y[0])
    synth_path = emb_dir / "train_synth.npz"
    if use_synthetic and synth_path.exists():
        sy = load_embeddings(synth_path)
        X.append(sy["emb"]); y.append(_labels(sy, multiclass))
        src += ["synth"] * len(sy["label"])
    # REFLOW-CLEAN: variantes de layout legitimo como NEGATIVOS (label 0 / category clean).
    # Expandem o cluster limpo -> gate aprende invariancia a layout + resolucao (anti-confound
    # pelo lado limpo, anti-falso-positivo). category='clean' -> _labels devolve 0 (bin e multi).
    reflow_path = emb_dir / "train_reflow.npz"
    if use_reflow:
        if reflow_path.exists():
            rf = load_embeddings(reflow_path)
            X.append(rf["emb"]); y.append(_labels(rf, multiclass))
            src += ["reflow"] * len(rf["label"])
        else:
            print(f"  [aviso] synthetic.reflow_clean=true mas {reflow_path} ausente -> "
                  "treino SEM reflow. Rode scripts/make_synthetic.py para gera-lo.")
    return np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int64), np.array(src)


def _balanced_batches(y: torch.Tensor, batch_size: int, rng: torch.Generator, balance: bool,
                      max_oversample: int = 0):
    """Gera indices de batches balanceados por CLASSE (>=2 por classe presente, p/ a SupCon
    ter positivos). Generaliza o caso binario 50/50 para N+1 classes.

    `max_oversample` (Fase 4.4): teto de quantas vezes CADA exemplo de uma classe pode
    reaparecer por epoca. Com teto, monta um POOL por classe (cada indice repetido <= teto
    vezes, embaralhado) e consome SEM reposicao -> classes raras (orientation/distortion) nao
    sao replicadas ~90x/epoca. Sem teto (0), mantem a amostragem COM reposicao (legado)."""
    classes = torch.unique(y)
    if not balance or len(classes) <= 1:
        perm = torch.randperm(len(y), generator=rng)
        return [perm[i:i + batch_size] for i in range(0, len(y), batch_size)]
    idx_by_class = [torch.where(y == c)[0] for c in classes]
    per = max(2, batch_size // len(classes))
    n_batches = max(1, len(y) // batch_size)
    pools, cursors = [], []
    for idxs in idx_by_class:
        if max_oversample and max_oversample > 0:
            reps = idxs.repeat(max_oversample)
            pools.append(reps[torch.randperm(len(reps), generator=rng)])
        else:
            pools.append(None)
        cursors.append(0)
    batches = []
    for _ in range(n_batches):
        picks = []
        for ci, idxs in enumerate(idx_by_class):
            pool = pools[ci]
            if pool is None:                       # sem teto: com reposicao (legado)
                picks.append(idxs[torch.randint(len(idxs), (per,), generator=rng)])
            else:                                  # com teto: consome o pool sem reposicao
                c = cursors[ci]
                take = pool[c:c + per]
                cursors[ci] = c + len(take)
                if len(take) >= 2:                 # SupCon precisa de >=2 positivos da classe
                    picks.append(take)
        if picks:
            batches.append(torch.cat(picks))
    return [b for b in batches if len(b) >= 2]


def train_head(cfg: Config, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir = Path(cfg.paths.emb_dir)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = torch.Generator().manual_seed(cfg.seed)

    multiclass = cfg.train.multiclass
    num_classes = len(CATEGORIES) if multiclass else 1

    # synthetic.enabled e' a chave-mestra do anti-confound: se desligada, NENHUM sintetico
    # entra (treino nem sonda de early-stop), independentemente de train.use_synthetic.
    # (antes synthetic.enabled so afetava make_synthetic.py -> parametro logico ignorado.)
    use_synth = cfg.train.use_synthetic and cfg.synthetic.enabled
    # reflow-clean tambem e' gateado por synthetic.enabled (e' uma augmentacao sintetica de limpas)
    use_reflow = cfg.synthetic.reflow_clean and cfg.synthetic.enabled
    X, y, src = assemble_training(emb_dir, cfg.train.use_real_errors, use_synth, multiclass,
                                  use_reflow=use_reflow)
    in_dim = X.shape[1]
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    clean_idx = torch.where(yt == 0)[0]   # clean = id 0 (binario ou multi-classe)

    val = load_embeddings(emb_dir / "val.npz")
    Xva = torch.from_numpy(val["emb"].astype(np.float32)).to(device)
    yva_cat = _labels(val, True)                       # id de categoria (sempre util p/ Estagio 2)
    if multiclass:
        yva_bin = (yva_cat != 0).astype(np.int64)      # gate "tem erro?" (Estagio 1)
        err_tr_np = y != 0                             # erros de treino p/ protótipos de categoria
        err_va_np = yva_cat != 0
    else:
        yva_bin = val["label"].astype(np.int64)

    # sonda LIVRE DE CONFOUND na VAL (erros sinteticos nas limpas de val, mesma resolucao) ->
    # criterio de early-stop ESTAVEL e HONESTO (vs o gate confundido por resolucao). Sem
    # data-snooping no test. Fallback p/ o gate confundido se val_synth nao existir.
    vsp = emb_dir / "val_synth.npz"
    Xva_synth = None
    if use_synth and vsp.exists():
        Xva_synth = torch.from_numpy(load_embeddings(vsp)["emb"].astype(np.float32)).to(device)

    model = SiameseNet(in_dim, cfg.head.hidden, cfg.head.proj_dim, cfg.head.p_drop,
                       num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # NB: NAO usamos pesos de classe na cross-entropy — _balanced_batches ja apresenta as
    # classes equilibradas por batch; somar inverse-freq dobraria a correcao e faria as
    # classes raras (orientation/distortion) dominarem.
    nsrc = lambda s: int((src == s).sum())
    if multiclass:
        dist = {CATEGORIES[c]: int((y == c).sum()) for c in range(num_classes)}
        print(f"[treino] amostras={len(y)} MULTI-CLASSE ({num_classes} classes) "
              f"[real={nsrc('real')}, synth={nsrc('synth')}, reflow={nsrc('reflow')}] in_dim={in_dim}")
        print(f"         distribuicao por classe: {dist}")
    else:
        print(f"[treino] amostras={len(y)} BINARIO (limpas={int((y==0).sum())}, "
              f"erros={int((y==1).sum())}) [real={nsrc('real')}, "
              f"synth={nsrc('synth')}, reflow={nsrc('reflow')}] in_dim={in_dim}")

    sel_name = cfg.train.early_stop_metric
    best_sel, best_state, best_epoch, best_metrics = -1.0, None, -1, {}
    since_improve = 0
    sel_warned = False
    history = []

    for epoch in range(cfg.train.epochs):
        model.train()
        batches = _balanced_batches(yt, cfg.train.batch_size, rng, cfg.train.balance_batches,
                                    max_oversample=getattr(cfg.train, "max_oversample_per_class", 0))
        ep_loss = 0.0
        for bidx in batches:
            z, aux = model(Xt[bidx])
            yb = yt[bidx]
            if cfg.train.loss == "supcon":
                l_metric = supcon_loss(z, yb, cfg.train.temperature)
            else:
                # contrastiva: forma pares aleatorios dentro do batch
                perm = torch.randperm(len(bidx), generator=rng).to(device)
                ypair = (yb != yb[perm]).float()
                l_metric = contrastive_loss(z, z[perm], ypair)
            if multiclass:
                l_aux = F.cross_entropy(aux, yb)
            else:
                l_aux = F.binary_cross_entropy_with_logits(aux, yb.float())
            loss = l_metric + cfg.train.aux_weight * l_aux
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()

        # validacao: AP do GATE binario "tem erro?" (Estagio 1) -> criterio de early-stop
        # (preserva o objetivo de alta precisao; a clusterizacao por categoria e avaliada
        #  no Estagio 2 em evaluate.py). No multi-classe, p(erro)=1-softmax[clean].
        model.eval()
        with torch.no_grad():
            z_all, _ = model(Xt)
            z_all_np = z_all.cpu().numpy()
            protos = fit_prototypes(z_all_np[clean_idx.cpu().numpy()], k=cfg.decision.k_prototypes, seed=cfg.seed)
            z_va, aux_va = model(Xva)
            z_va_np = z_va.cpu().numpy()
            decider = PrototypeDecider(protos, threshold=0.0, target_precision=cfg.decision.target_precision)
            score_proto = decider.scores(z_va_np)
            ap_proto = average_precision_score(yva_bin, score_proto)
            if multiclass:
                p_err = 1.0 - torch.softmax(aux_va, dim=1)[:, 0].cpu().numpy()
            else:
                p_err = torch.sigmoid(aux_va).cpu().numpy()
            ap_aux = average_precision_score(yva_bin, p_err)

            # GATE LIVRE DE CONFOUND (val): clean-val real vs val_synth (mesma resolucao)
            sg_proto = sg_aux = float("nan")
            if Xva_synth is not None:
                z_vs, aux_vs = model(Xva_synth)
                sp_synth = decider.scores(z_vs.cpu().numpy())
                clean_va = yva_bin == 0
                y_cf = np.concatenate([np.zeros(int(clean_va.sum())), np.ones(len(sp_synth))])
                sg_proto = float(roc_auc_score(y_cf, np.concatenate([score_proto[clean_va], sp_synth])))
                if multiclass:
                    pe_synth = 1.0 - torch.softmax(aux_vs, dim=1)[:, 0].cpu().numpy()
                else:
                    pe_synth = torch.sigmoid(aux_vs).cpu().numpy()
                sg_aux = float(roc_auc_score(y_cf, np.concatenate([p_err[clean_va], pe_synth])))

        cat_f1 = float("nan")
        if multiclass and err_tr_np.sum() > 0 and err_va_np.sum() > 0:
            # Estagio 2 (clusterizacao): protótipos por categoria no train, macro-F1 na val
            cprotos, cids = fit_category_prototypes(z_all_np[err_tr_np], y[err_tr_np],
                                                    k=cfg.decision.k_prototypes, seed=cfg.seed)
            y_pred = assign_category(z_va_np[err_va_np], cprotos, cids)
            cat_f1 = float(f1_score(yva_cat[err_va_np], y_pred, average="macro", zero_division=0))

        # TABELA de criterios candidatos. O salvamento usa cfg.train.early_stop_metric (agora
        # respeitado): por padrao 'val_synth_gate' = gate LIVRE DE CONFOUND da cabeca de
        # producao (aux head). A formula legada max(proto,aux)+0.5*cat_f1 continua disponivel
        # como 'val_synth_gate_max'/'val_synth_gate+cat_f1', mas nao e' mais o default hardcoded.
        has_synth = sg_aux == sg_aux  # not NaN
        metrics = {
            "val_ap": float(max(ap_proto, ap_aux)),
            "val_ap_proto": float(ap_proto),
            "val_ap_aux": float(ap_aux),
            "val_cat_f1": cat_f1,
            "val_synth_gate": sg_aux,
            "val_synth_gate_proto": sg_proto,
            "val_synth_gate_max": (max(sg_proto, sg_aux) if has_synth else float("nan")),
        }
        metrics["val_synth_gate+cat_f1"] = (
            0.5 * sg_aux + 0.5 * cat_f1 if (has_synth and cat_f1 == cat_f1) else sg_aux)

        sel = metrics.get(sel_name, float("nan"))
        if sel != sel:  # metrica pedida ausente/NaN (ex.: val_synth sem synthetic) -> fallback honesto
            if not sel_warned:
                print(f"  [aviso] early_stop_metric='{sel_name}' indisponivel (NaN) -> "
                      f"fallback para 'val_ap' (gate confundido). Habilite synthetic p/ a metrica honesta.")
                sel_warned = True
            sel = metrics["val_ap"]

        history.append({"epoch": epoch, "loss": ep_loss / len(batches), **metrics, "sel": sel})

        if sel > best_sel + 1e-4:
            best_sel, best_epoch, best_metrics = sel, epoch, dict(metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
        if epoch % 25 == 0 or epoch == cfg.train.epochs - 1:
            print(f"  ep {epoch:3d} loss={ep_loss/len(batches):.4f} synth_gate={sg_aux:.3f} "
                  f"val_ap_aux={ap_aux:.3f} val_cat_f1={cat_f1:.3f} ({sel_name}={sel:.3f})")
        if since_improve >= cfg.train.patience:
            print(f"  early stop @ {epoch} (melhor {sel_name}={best_sel:.3f} @ {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    out = Path(cfg.paths.models_dir); out.mkdir(parents=True, exist_ok=True)
    # PROVENIENCIA (Fase 6 — rastreabilidade total): config completa, hash do commit e
    # hashes do dataset (embeddings de entrada) gravados no checkpoint.
    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "head": {"hidden": cfg.head.hidden, "proj_dim": cfg.head.proj_dim, "p_drop": cfg.head.p_drop},
        "num_classes": num_classes,
        "multiclass": multiclass,
        "categories": list(CATEGORIES) if multiclass else ["clean", "error"],
        "early_stop_metric": sel_name,
        "best_sel": best_sel, "best_epoch": best_epoch, "best_metrics": best_metrics,
        "history": history,
        "provenance": {
            "git_commit": _git_commit(),
            "config": asdict(cfg),
            "dataset_sha256": {n: _file_sha256(emb_dir / n) for n in
                               ("train.npz", "train_synth.npz", "train_reflow.npz",
                                "val.npz", "val_synth.npz", "val_reflow.npz")},
            "used_synthetic": bool(use_synth),
            "used_reflow": bool(use_reflow),
        },
        "train_cfg": {"use_real_errors": cfg.train.use_real_errors,
                      "use_synthetic": bool(use_synth), "synthetic_enabled": cfg.synthetic.enabled,
                      "loss": cfg.train.loss, "multiclass": multiclass},
    }
    torch.save(ckpt, out / "siamese_head.pt")
    print(f"[treino] melhor {sel_name}={best_sel:.3f} (ep {best_epoch}) -> {out/'siamese_head.pt'}")
    # retorno rico p/ grid_search ranquear SEM tocar o teste (anti-snooping): metricas de VAL.
    return {"best_sel": best_sel, "sel_name": sel_name, "best_epoch": best_epoch,
            "best_metrics": best_metrics, "history": history}


def load_model(ckpt_path: Path, device: str = "cpu") -> SiameseNet:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = SiameseNet(ck["in_dim"], ck["head"]["hidden"], ck["head"]["proj_dim"], ck["head"]["p_drop"],
                   num_classes=ck.get("num_classes", 1)).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m
