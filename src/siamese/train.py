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
from sklearn.metrics import average_precision_score

from .config import Config
from .features import load_embeddings
from .model import SiameseNet
from .losses import supcon_loss, contrastive_loss
from .decision import fit_prototypes, PrototypeDecider


def assemble_training(emb_dir: Path, use_real_errors: bool, use_synthetic: bool):
    tr = load_embeddings(emb_dir / "train.npz")
    if use_real_errors:
        X, y = [tr["emb"]], [tr["label"]]
    else:  # so as limpas reais
        m = tr["label"] == 0
        X, y = [tr["emb"][m]], [tr["label"][m]]
    src = ["real"] * len(y[0])
    synth_path = emb_dir / "train_synth.npz"
    if use_synthetic and synth_path.exists():
        sy = load_embeddings(synth_path)
        X.append(sy["emb"]); y.append(sy["label"])
        src += ["synth"] * len(sy["label"])
    return np.concatenate(X).astype(np.float32), np.concatenate(y).astype(np.int64), np.array(src)


def _balanced_batches(y: torch.Tensor, batch_size: int, rng: torch.Generator, balance: bool):
    """Gera indices de batches; se balance, ~50/50 por classe."""
    pos = torch.where(y == 1)[0]
    neg = torch.where(y == 0)[0]
    if not balance:
        perm = torch.randperm(len(y), generator=rng)
        return [perm[i:i + batch_size] for i in range(0, len(y), batch_size)]
    half = batch_size // 2
    n_batches = max(1, (len(pos) + len(neg)) // batch_size)
    batches = []
    for _ in range(n_batches):
        pi = pos[torch.randint(len(pos), (half,), generator=rng)]
        ni = neg[torch.randint(len(neg), (half,), generator=rng)]
        batches.append(torch.cat([pi, ni]))
    return batches


def train_head(cfg: Config, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb_dir = Path(cfg.paths.emb_dir)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = torch.Generator().manual_seed(cfg.seed)

    X, y, src = assemble_training(emb_dir, cfg.train.use_real_errors, cfg.train.use_synthetic)
    in_dim = X.shape[1]
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    clean_idx = torch.where(yt == 0)[0]

    val = load_embeddings(emb_dir / "val.npz")
    Xva = torch.from_numpy(val["emb"].astype(np.float32)).to(device)
    yva = val["label"].astype(np.int64)

    model = SiameseNet(in_dim, cfg.head.hidden, cfg.head.proj_dim, cfg.head.p_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    print(f"[treino] amostras={len(y)} (limpas={int((y==0).sum())}, erros={int((y==1).sum())} "
          f"[real={int((src=='real').sum())}, synth={int((src=='synth').sum())}]) in_dim={in_dim}")

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
            l_aux = F.binary_cross_entropy_with_logits(aux, yb.float())
            loss = l_metric + cfg.train.aux_weight * l_aux
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()

        # validacao: AP do score de prototipo (regra de decisao de producao)
        model.eval()
        with torch.no_grad():
            z_all, _ = model(Xt)
            protos = fit_prototypes(z_all[clean_idx].cpu().numpy(), k=cfg.decision.k_prototypes, seed=cfg.seed)
            z_va, aux_va = model(Xva)
            decider = PrototypeDecider(protos, threshold=0.0, target_precision=cfg.decision.target_precision)
            score_proto = decider.scores(z_va.cpu().numpy())
            ap_proto = average_precision_score(yva, score_proto)
            ap_aux = average_precision_score(yva, torch.sigmoid(aux_va).cpu().numpy())
        ap = max(ap_proto, ap_aux)
        history.append({"epoch": epoch, "loss": ep_loss / len(batches), "val_ap_proto": ap_proto, "val_ap_aux": ap_aux})

        if ap > best_ap + 1e-4:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1
        if epoch % 25 == 0 or epoch == cfg.train.epochs - 1:
            print(f"  ep {epoch:3d} loss={ep_loss/len(batches):.4f} val_ap_proto={ap_proto:.3f} val_ap_aux={ap_aux:.3f}")
        if since_improve >= cfg.train.patience:
            print(f"  early stop @ {epoch} (melhor ap={best_ap:.3f} @ {best_epoch})")
            break

    model.load_state_dict(best_state)
    out = Path(cfg.paths.models_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "head": {"hidden": cfg.head.hidden, "proj_dim": cfg.head.proj_dim, "p_drop": cfg.head.p_drop},
        "best_ap": best_ap, "best_epoch": best_epoch,
        "train_cfg": {"use_real_errors": cfg.train.use_real_errors, "use_synthetic": cfg.train.use_synthetic,
                      "loss": cfg.train.loss},
    }
    torch.save(ckpt, out / "siamese_head.pt")
    print(f"[treino] melhor val_ap={best_ap:.3f} (ep {best_epoch}) -> {out/'siamese_head.pt'}")
    return {"best_ap": best_ap, "best_epoch": best_epoch, "history": history}


def load_model(ckpt_path: Path, device: str = "cpu") -> SiameseNet:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = SiameseNet(ck["in_dim"], ck["head"]["hidden"], ck["head"]["proj_dim"], ck["head"]["p_drop"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m
