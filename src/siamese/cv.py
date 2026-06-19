"""Grouped Nested Cross-Validation + predicoes OUT-OF-FOLD (Fase 2 da auditoria).

Motivacao: o limiar de decisao era calibrado nas mesmas ~26 limpas de validacao em que o
modelo era selecionado (ressubstituicao -> ponto de operacao fragil; especificidade oscilava
0.12–0.62 entre seeds). Aqui:

  - **Outer folds** (GroupKFold por SESSAO/ticket, sem grupo cruzando fold) dao a estimativa de
    generalizacao; cada item recebe uma predicao de um modelo treinado SO nos outros folds (OOF).
  - **Inner holdout** por fold (tambem agrupado) faz o early-stop -> a parte "nested".
  - O limiar e' calibrado nas predicoes OOF (nunca in-sample) -> robusto.

Nada disto toca o TESTE externo (continua trancado por siamese.protocol). Opera sobre arrays de
embeddings ja' cacheados (rapido); a montagem do pool (com sinteticos ancorados a imagem-mae) e'
responsabilidade de quem chama (scripts/nested_cv.py)."""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .model import SiameseNet
from .losses import supcon_loss
from .decision import fit_prototypes, PrototypeDecider
from .train import _balanced_batches


def _aux_err(aux: np.ndarray, multiclass: bool) -> np.ndarray:
    """Score escalar de erro da cabeca aux (igual a evaluate._aux_err)."""
    if multiclass:
        e = np.exp(aux - aux.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return 1.0 - p[:, 0]
    return aux


# ---------- folds agrupados (deterministicos por seed) ----------
def grouped_kfold(groups, n_splits: int, seed: int):
    """Gera (train_idx, test_idx) com NENHUM grupo cruzando fold. Atribui cada grupo (maior
    primeiro) ao fold atualmente mais vazio -> folds balanceados; embaralho por seed varia a
    particao entre seeds (desempate)."""
    groups = np.asarray(groups)
    uniq = sorted(set(groups.tolist()))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    sizes = {g: int((groups == g).sum()) for g in uniq}
    load = [0] * n_splits
    fold_of: dict = {}
    for g in sorted(uniq, key=lambda g: -sizes[g]):
        f = min(range(n_splits), key=lambda i: load[i])
        fold_of[g] = f
        load[f] += sizes[g]
    idx = np.arange(len(groups))
    fold_id = np.array([fold_of[g] for g in groups])
    for f in range(n_splits):
        te = idx[fold_id == f]
        tr = idx[fold_id != f]
        if len(te) and len(tr):
            yield tr, te


def grouped_holdout(idx, groups, frac: float, seed: int):
    """Separa um inner holdout (por grupo) de ~frac dos itens de `idx`, para early-stop."""
    idx = np.asarray(idx)
    g = np.asarray(groups)[idx]
    uniq = sorted(set(g.tolist()))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    target = int(round(len(idx) * frac))
    held, n = set(), 0
    for grp in uniq:
        if n >= target:
            break
        held.add(grp)
        n += int((g == grp).sum())
    mask = np.array([x in held for x in g])
    return idx[~mask], idx[mask]


# ---------- treino da cabeca sobre arrays (early-stop opcional) ----------
def fit_head_arrays(Xtr, ytr_cat, *, in_dim, cfg, seed, num_classes, device,
                    Xiv=None, yiv_bin=None, epochs=200, patience=30):
    """Treina SiameseNet em (Xtr, ytr_cat). Se (Xiv, yiv_bin) com 2 classes, early-stop pelo AP
    do gate binario no inner-val; senao, treina `epochs` fixas. Devolve o modelo (melhor estado)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(Xtr.astype(np.float32)).to(device)
    yt = torch.from_numpy(ytr_cat.astype(np.int64)).to(device)
    clean_np = np.where(ytr_cat == 0)[0]
    use_es = Xiv is not None and yiv_bin is not None and len(np.unique(yiv_bin)) == 2
    if use_es:
        Xivt = torch.from_numpy(Xiv.astype(np.float32)).to(device)

    model = SiameseNet(in_dim, cfg.head.hidden, cfg.head.proj_dim, cfg.head.p_drop,
                       num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    max_os = getattr(cfg.train, "max_oversample_per_class", 0)
    best_ap, best_state, since = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        for bidx in _balanced_batches(yt, cfg.train.batch_size, rng, cfg.train.balance_batches, max_os):
            z, aux = model(Xt[bidx])
            yb = yt[bidx]
            l_metric = supcon_loss(z, yb, cfg.train.temperature)
            if num_classes > 1:
                l_aux = F.cross_entropy(aux, yb)
            else:
                l_aux = F.binary_cross_entropy_with_logits(aux, yb.float())
            loss = l_metric + cfg.train.aux_weight * l_aux
            opt.zero_grad(); loss.backward(); opt.step()

        if not use_es:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue
        model.eval()
        with torch.no_grad():
            z_all, _ = model(Xt)
            protos = fit_prototypes(z_all.cpu().numpy()[clean_np], k=cfg.decision.k_prototypes, seed=seed)
            z_iv, aux_iv = model(Xivt)
            dec = PrototypeDecider(protos, 0.0, cfg.decision.target_precision)
            ap_proto = average_precision_score(yiv_bin, dec.scores(z_iv.cpu().numpy()))
            p_err = _aux_err(aux_iv.cpu().numpy(), num_classes > 1)
            ap = float(max(ap_proto, average_precision_score(yiv_bin, p_err)))
        if ap > best_ap + 1e-4:
            best_ap, since = ap, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _embed(model, X, device):
    with torch.no_grad():
        z, aux = model(torch.from_numpy(X.astype(np.float32)).to(device))
    return z.cpu().numpy(), aux.cpu().numpy()


# ---------- CV out-of-fold ----------
def run_oof(X, ycat, label, groups, *, cfg, seed, n_splits, num_classes, device,
            inner_frac=0.15, epochs=200, patience=30):
    """Grouped nested CV. Para cada outer fold: treina a cabeca nos demais folds (early-stop em
    inner holdout), ajusta prototipos + fusao logistica no fold-train, e prediz no fold-test.
    Devolve as predicoes OOF (fused/proto/aux_err) alinhadas a X, cobrindo TODO o pool 1x."""
    n = len(X)
    oof_fused = np.full(n, np.nan)
    oof_proto = np.full(n, np.nan)
    oof_aux = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)
    in_dim = X.shape[1]
    for fi, (tr, te) in enumerate(grouped_kfold(groups, n_splits, seed)):
        in_tr, in_val = grouped_holdout(tr, groups, inner_frac, seed * 100 + fi)
        model = fit_head_arrays(
            X[in_tr], ycat[in_tr], in_dim=in_dim, cfg=cfg, seed=seed, num_classes=num_classes,
            device=device, Xiv=X[in_val], yiv_bin=label[in_val], epochs=epochs, patience=patience)
        # prototipos + fusao no fold-train INTEIRO (o modelo veio so do inner-train)
        z_tr, aux_tr = _embed(model, X[tr], device)
        clean_tr = np.where(ycat[tr] == 0)[0]
        dec = PrototypeDecider(fit_prototypes(z_tr[clean_tr], k=cfg.decision.k_prototypes, seed=seed),
                               0.0, cfg.decision.target_precision)
        sp_tr = dec.scores(z_tr)
        ae_tr = _aux_err(aux_tr, num_classes > 1)
        fus = LogisticRegression(max_iter=1000).fit(np.stack([sp_tr, ae_tr], 1), label[tr])
        # predicao OOF no fold-test
        z_te, aux_te = _embed(model, X[te], device)
        sp_te = dec.scores(z_te)
        ae_te = _aux_err(aux_te, num_classes > 1)
        oof_proto[te] = sp_te
        oof_aux[te] = ae_te
        oof_fused[te] = fus.predict_proba(np.stack([sp_te, ae_te], 1))[:, 1]
        covered[te] = True
    if not covered.all():
        raise RuntimeError(f"OOF incompleto: {int((~covered).sum())} itens sem predicao "
                           "(grupo nao alocado a nenhum fold?)")
    return {"fused": oof_fused, "proto": oof_proto, "aux_err": oof_aux}
