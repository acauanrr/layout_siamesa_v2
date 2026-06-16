"""Localizacao de erro por patch + mapa de calor.

Duas estrategias (a supervisionada e a padrao, por ser fiel ao "erro aprendido"):

1) SyntheticPatchLocalizer (padrao) — SUPERVISIONADO pelos erros sinteticos.
   Injetamos erros nas imagens limpas e sabemos EXATAMENTE onde os pusemos (diff de pixels).
   Treinamos um classificador leve por-patch (LogReg sobre o patch token 384-d) para
   distinguir patch-de-erro vs patch-normal. O mapa de calor = P(erro) por patch. Assim ele
   aprende a assinatura visual de faixa preta / vazio / overlay / crop e localiza o defeito.

2) PatchCoreLocalizer (alternativa) — NAO-supervisionado: distancia ao patch limpo mais
   proximo (novidade vs telas limpas). Bom p/ conteudo estranho; fraco p/ faixa preta
   (preto tambem ocorre em telas limpas).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image

from .backbone import DinoV2Backbone, load_image
from .geometry import IMAGENET_MEAN, IMAGENET_STD, pad_to_square, content_patch_mask
from .synthetic import inject
from .features import read_manifest


# ---------- rotulos de erro por patch (via diff de pixels) ----------
def _patch_error_labels(orig: Image.Image, corrupted: Image.Image, size: int, mode: str,
                        grid: int, thresh: float = 0.25) -> np.ndarray:
    a = np.asarray(orig).astype(np.int16)
    b = np.asarray(corrupted).astype(np.int16)
    diff = (np.abs(a - b).mean(axis=2) > 12).astype(np.uint8) * 255   # HxW onde mudou
    mimg = Image.fromarray(diff, mode="L")
    if mode == "pad":
        mimg = pad_to_square(mimg.convert("RGB"), (0, 0, 0)).convert("L")
    mimg = mimg.resize((size, size), Image.NEAREST)
    m = (np.asarray(mimg) > 127).astype(np.float32)
    patch = size // grid
    pooled = m.reshape(grid, patch, grid, patch).mean(axis=(1, 3))    # fracao de erro por patch
    return (pooled.reshape(-1) > thresh)


def _denorm(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _heatmap_overlay(image_path, grid_scores, score, x, out_path: Path,
                     vmin=None, vmax=None, title_extra=""):
    """vmin/vmax: escala ABSOLUTA (ex.: 0..1 p/ probabilidade) -> imagem limpa fica fria.
    Se None, usa min-max por imagem (apenas relativo)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vis = _denorm(x)
    g = grid_scores.copy()
    lo = np.nanmin(g) if vmin is None else vmin
    hi = np.nanmax(g) if vmax is None else vmax
    gnorm = np.clip((g - lo) / (hi - lo + 1e-9), 0, 1)
    heat = np.array(Image.fromarray((np.nan_to_num(gnorm) * 255).astype(np.uint8))
                    .resize((vis.shape[1], vis.shape[0]), Image.BICUBIC)) / 255.0
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    ax[0].imshow(vis); ax[0].set_title(Path(image_path).name[:34]); ax[0].axis("off")
    ax[1].imshow(vis); im = ax[1].imshow(heat, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax[1].set_title(f"erro (score={score:.3f}){title_extra}"); ax[1].axis("off")
    fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110); plt.close(fig)
    return score


class SyntheticPatchLocalizer:
    """Classificador por-patch supervisionado por erros sinteticos."""

    def __init__(self, backbone: DinoV2Backbone, clf: LogisticRegression, top_frac: float = 0.03):
        self.backbone = backbone
        self.grid = backbone.cfg.size // 14
        self.clf = clf
        self.top_frac = top_frac

    @staticmethod
    def train(backbone: DinoV2Backbone, train_csv: Path, *, n_variants: int = 4,
              max_errors: int = 2, seed: int = 0, batch_size: int = 8,
              max_neg_per_img: int = 200) -> "SyntheticPatchLocalizer":
        rows = [r for r in read_manifest(train_csv) if int(r["label"]) == 0]
        rng = random.Random(seed)
        grid = backbone.cfg.size // 14
        mode = backbone.cfg.preprocess
        X, y = [], []
        buf, labels_buf, cmask_buf = [], [], []

        def flush():
            if not buf:
                return
            xb = torch.stack(buf)
            tok = backbone.patch_tokens(xb).cpu().numpy()                 # [B,N,C]
            for k in range(len(buf)):
                lab = labels_buf[k]; cm = cmask_buf[k]
                pos = np.where(lab & cm)[0]
                neg = np.where((~lab) & cm)[0]
                if len(neg) > max_neg_per_img:
                    neg = rng_choice(neg, max_neg_per_img, rng)
                idx = np.concatenate([pos, neg])
                X.append(tok[k][idx]); y.append(lab[idx].astype(int))
            buf.clear(); labels_buf.clear(); cmask_buf.clear()

        for r in rows:
            orig = load_image(r["path"])
            for _ in range(n_variants):
                corr, _ = inject(orig, rng, n_errors=max_errors)
                xt, cmask = backbone.preprocess(corr)
                lab = _patch_error_labels(orig, corr, backbone.cfg.size, mode, grid)
                buf.append(xt); labels_buf.append(lab); cmask_buf.append(cmask.numpy().astype(bool))
                if len(buf) >= batch_size:
                    flush()
        flush()
        Xc = np.concatenate(X); yc = np.concatenate(y)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Xc, yc)
        print(f"[localizer] patches treino={len(yc)} (erro={int(yc.sum())}) "
              f"acc_treino={clf.score(Xc, yc):.3f}")
        return SyntheticPatchLocalizer(backbone, clf)

    def save(self, models_dir: Path):
        np.savez(Path(models_dir) / "patch_localizer.npz",
                 coef=self.clf.coef_.ravel(), intercept=self.clf.intercept_,
                 classes=self.clf.classes_)

    @classmethod
    def load(cls, backbone: DinoV2Backbone, models_dir: Path, **kw):
        d = np.load(Path(models_dir) / "patch_localizer.npz")
        clf = LogisticRegression()
        clf.coef_ = d["coef"].reshape(1, -1); clf.intercept_ = d["intercept"]
        clf.classes_ = d["classes"]
        return cls(backbone, clf, **kw)

    @torch.no_grad()
    def anomaly_map(self, image_path: str):
        img = load_image(image_path)
        x, cmask = self.backbone.preprocess(img)
        cmask = cmask.numpy().astype(bool)
        tok = self.backbone.patch_tokens(x.unsqueeze(0)).cpu().numpy()[0]   # [N,C]
        prob = self.clf.predict_proba(tok)[:, 1]
        prob[~cmask] = np.nan
        grid = prob.reshape(self.grid, self.grid)
        vals = prob[cmask]
        k = max(1, int(len(vals) * self.top_frac))
        score = float(np.sort(vals[~np.isnan(vals)])[-k:].mean())
        return grid, score, x

    def save_overlay(self, image_path: str, out_path: Path):
        grid, score, x = self.anomaly_map(image_path)
        # escala ABSOLUTA 0..1 (probabilidade) -> tela limpa fica fria
        return _heatmap_overlay(image_path, grid, score, x, out_path, vmin=0.0, vmax=1.0)


def rng_choice(arr, n, rng: random.Random):
    idx = list(range(len(arr)))
    rng.shuffle(idx)
    return arr[idx[:n]]


class PatchCoreLocalizer:
    """Alternativa nao-supervisionada: distancia ao patch limpo mais proximo."""

    def __init__(self, backbone: DinoV2Backbone, bank: np.ndarray, top_frac: float = 0.01):
        self.backbone = backbone
        self.grid = backbone.cfg.size // 14
        self.top_frac = top_frac
        self.bank = bank
        self.nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(normalize(bank))
        # calibracao da escala: distancia "normal" entre patches limpos (2o vizinho, exclui self)
        rng = np.random.default_rng(0)
        sample = normalize(bank[rng.choice(len(bank), min(3000, len(bank)), replace=False)])
        d = self.nn.kneighbors(sample)[0][:, 1]
        self.vmin = float(np.percentile(d, 50))
        self.vmax = float(np.percentile(d, 99.5))

    @staticmethod
    def build_bank(backbone, train_csv: Path, *, max_tokens=30000, batch_size=8, seed=0):
        rows = [r for r in read_manifest(train_csv) if int(r["label"]) == 0]
        rng = np.random.default_rng(seed)
        toks, buf, mbuf = [], [], []

        def flush():
            if not buf:
                return
            p = backbone.patch_tokens(torch.stack(buf)).cpu().numpy()
            for k in range(len(buf)):
                toks.append(p[k][mbuf[k].numpy().astype(bool)])
            buf.clear(); mbuf.clear()
        for r in rows:
            x, m = backbone.preprocess(load_image(r["path"]))
            buf.append(x); mbuf.append(m)
            if len(buf) >= batch_size:
                flush()
        flush()
        bank = np.concatenate(toks, axis=0)
        if len(bank) > max_tokens:
            bank = bank[rng.choice(len(bank), max_tokens, replace=False)]
        return bank.astype(np.float32)

    @torch.no_grad()
    def anomaly_map(self, image_path: str):
        x, cmask = self.backbone.preprocess(load_image(image_path))
        cmask = cmask.numpy().astype(bool)
        tok = self.backbone.patch_tokens(x.unsqueeze(0)).cpu().numpy()[0]
        dist = np.full(len(tok), np.nan, dtype=np.float32)
        d, _ = self.nn.kneighbors(normalize(tok[cmask]))
        dist[cmask] = d[:, 0]
        vals = dist[cmask]
        k = max(1, int(len(vals) * self.top_frac))
        score = float(np.sort(vals)[-k:].mean())
        return dist.reshape(self.grid, self.grid), score, x

    def save_overlay(self, image_path: str, out_path: Path):
        grid, score, x = self.anomaly_map(image_path)
        # escala ABSOLUTA calibrada (telas limpas ficam frias)
        return _heatmap_overlay(image_path, grid, score, x, out_path,
                                vmin=self.vmin, vmax=self.vmax)
