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

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .config import Config
from .features import load_embeddings
from .model import SiameseNet
from .losses import supcon_loss, contrastive_loss
from .decision import (fit_prototypes, PrototypeDecider,
                       fit_category_prototypes, assign_category)
from .manifest import CATEGORY_TO_ID, CATEGORIES


def _labels(d: dict, multiclass: bool) -> np.ndarray:
    """Rotulo de treino: id de CATEGORIA (0=clean,1..N) se multiclass, senao binario 0/1."""
    if multiclass:
        return np.array([CATEGORY_TO_ID.get(str(c), 0) for c in d["category"]], dtype=np.int64)
    return d["label"].astype(np.int64)


def assemble_training(emb_dir: Path, use_real_errors: bool, use_synthetic: bool,
                      multiclass: bool = True):
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
    return np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int64), np.array(src)


def _balanced_batches(y: torch.Tensor, batch_size: int, rng: torch.Generator, balance: bool):
    """Gera indices de batches balanceados por CLASSE (>=2 por classe presente, p/ a SupCon
    ter positivos). Generaliza o caso binario 50/50 para N+1 classes; oversample com
    reposicao cobre classes raras (orientation/distortion)."""
    classes = torch.unique(y)
    if not balance or len(classes) <= 1:
        perm = torch.randperm(len(y), generator=rng)
        return [perm[i:i + batch_size] for i in range(0, len(y), batch_size)]
    idx_by_class = [torch.where(y == c)[0] for c in classes]
    per = max(2, batch_size // len(classes))
    n_batches = max(1, len(y) // batch_size)
    batches = []
    for _ in range(n_batches):
        picks = [idxs[torch.randint(len(idxs), (per,), generator=rng)] for idxs in idx_by_class]
        batches.append(torch.cat(picks))
    return batches


def train_head(cfg: Config, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir = Path(cfg.paths.emb_dir)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = torch.Generator().manual_seed(cfg.seed)

    multiclass = cfg.train.multiclass
    num_classes = len(CATEGORIES) if multiclass else 1

    X, y, src = assemble_training(emb_dir, cfg.train.use_real_errors,
                                  cfg.train.use_synthetic, multiclass)
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
    if cfg.train.use_synthetic and vsp.exists():
        Xva_synth = torch.from_numpy(load_embeddings(vsp)["emb"].astype(np.float32)).to(device)

    model = SiameseNet(in_dim, cfg.head.hidden, cfg.head.proj_dim, cfg.head.p_drop,
                       num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # NB: NAO usamos pesos de classe na cross-entropy — _balanced_batches ja apresenta as
    # classes equilibradas por batch; somar inverse-freq dobraria a correcao e faria as
    # classes raras (orientation/distortion) dominarem.
    if multiclass:
        dist = {CATEGORIES[c]: int((y == c).sum()) for c in range(num_classes)}
        print(f"[treino] amostras={len(y)} MULTI-CLASSE ({num_classes} classes) "
              f"[real={int((src=='real').sum())}, synth={int((src=='synth').sum())}] in_dim={in_dim}")
        print(f"         distribuicao por classe: {dist}")
    else:
        print(f"[treino] amostras={len(y)} BINARIO (limpas={int((y==0).sum())}, "
              f"erros={int((y==1).sum())}) [real={int((src=='real').sum())}, "
              f"synth={int((src=='synth').sum())}] in_dim={in_dim}")

    best_ap, best_state, best_epoch = -1.0, None, -1
    since_improve = 0
    history = []

    for epoch in range(cfg.train.epochs):
        model.train()
        batches = _balanced_batches(yt, cfg.train.batch_size, rng, cfg.train.balance_batches)
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
            synth_gate = float("nan")
            if Xva_synth is not None:
                z_vs, aux_vs = model(Xva_synth)
                sp_synth = decider.scores(z_vs.cpu().numpy())
                clean_va = yva_bin == 0
                y_cf = np.concatenate([np.zeros(int(clean_va.sum())), np.ones(len(sp_synth))])
                sg_proto = roc_auc_score(y_cf, np.concatenate([score_proto[clean_va], sp_synth]))
                if multiclass:
                    pe_synth = 1.0 - torch.softmax(aux_vs, dim=1)[:, 0].cpu().numpy()
                else:
                    pe_synth = torch.sigmoid(aux_vs).cpu().numpy()
                sg_aux = roc_auc_score(y_cf, np.concatenate([p_err[clean_va], pe_synth]))
                synth_gate = float(max(sg_proto, sg_aux))

        gate = max(ap_proto, ap_aux)               # gate CONFUNDIDO (so p/ log)
        honest_gate = synth_gate if synth_gate == synth_gate else gate  # livre de confound (selecao)
        cat_f1 = float("nan")
        if multiclass and err_tr_np.sum() > 0 and err_va_np.sum() > 0:
            # Estagio 2 (clusterizacao): protótipos por categoria no train, macro-F1 na val
            cprotos, cids = fit_category_prototypes(z_all_np[err_tr_np], y[err_tr_np],
                                                    k=cfg.decision.k_prototypes, seed=cfg.seed)
            y_pred = assign_category(z_va_np[err_va_np], cprotos, cids)
            cat_f1 = float(f1_score(yva_cat[err_va_np], y_pred, average="macro", zero_division=0))
            # equilibra deteccao honesta (gate livre de confound) + clusterizacao
            ap = 0.5 * honest_gate + 0.5 * cat_f1
        else:
            ap = honest_gate
        history.append({"epoch": epoch, "loss": ep_loss / len(batches), "val_ap_proto": ap_proto,
                        "val_ap_aux": ap_aux, "val_synth_gate": synth_gate, "val_cat_f1": cat_f1})

        if ap > best_ap + 1e-4:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
        if epoch % 25 == 0 or epoch == cfg.train.epochs - 1:
            print(f"  ep {epoch:3d} loss={ep_loss/len(batches):.4f} synth_gate={synth_gate:.3f} "
                  f"val_ap_aux={ap_aux:.3f} val_cat_f1={cat_f1:.3f} (sel={ap:.3f})")
        if since_improve >= cfg.train.patience:
            print(f"  early stop @ {epoch} (melhor ap={best_ap:.3f} @ {best_epoch})")
            break

    model.load_state_dict(best_state)
    out = Path(cfg.paths.models_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "head": {"hidden": cfg.head.hidden, "proj_dim": cfg.head.proj_dim, "p_drop": cfg.head.p_drop},
        "num_classes": num_classes,
        "multiclass": multiclass,
        "categories": list(CATEGORIES) if multiclass else ["clean", "error"],
        "best_ap": best_ap, "best_epoch": best_epoch,
        "train_cfg": {"use_real_errors": cfg.train.use_real_errors, "use_synthetic": cfg.train.use_synthetic,
                      "loss": cfg.train.loss, "multiclass": multiclass},
    }
    torch.save(ckpt, out / "siamese_head.pt")
    print(f"[treino] melhor val_ap={best_ap:.3f} (ep {best_epoch}) -> {out/'siamese_head.pt'}")
    return {"best_ap": best_ap, "best_epoch": best_epoch, "history": history}


def load_model(ckpt_path: Path, device: str = "cpu") -> SiameseNet:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = SiameseNet(ck["in_dim"], ck["head"]["hidden"], ck["head"]["proj_dim"], ck["head"]["p_drop"],
                   num_classes=ck.get("num_classes", 1)).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m
