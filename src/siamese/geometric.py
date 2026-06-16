"""Detector geometrico dedicado a BLACK-REGION e EMPTY-SPACE (bandas/regioes uniformes).

Por que existe: o PatchCore localiza "conteudo incomum" mas NAO acha a faixa preta (preto
tambem ocorre em telas limpas). Este detector ataca diretamente os dois erros mais comuns e
visuais, com visao computacional classica (sem aprendizado), de forma interpretavel e
LIVRE DE CONFOUND (mede a barra preta real, independente de resolucao/device).

Roda na imagem ORIGINAL (nao no 518 pre-processado) para ver as barras de verdade, nao o
nosso padding cinza.

Sinais:
- baixa variancia local  -> regiao "lisa" (sem textura/conteudo)
- escuridao              -> barra preta (tela dobravel nao expandida); distinta de app
                            dark-theme porque a barra e LISA e quase pura preta (<~28)
- criterio de BANDA      -> a regiao toca uma borda e se estende ao longo dela mas e fina
                            na perpendicular -> e uma faixa/banda (preta ou vazia), e NAO o
                            fundo branco da tela inteira (que preenche as duas dimensoes)

Saida: heatmap, score de imagem e regioes estruturadas (tipo, area, borda, caixa).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class Region:
    tipo: str            # "black_region" | "empty_space"
    area_frac: float
    luminancia: float
    bordas: str          # ex.: "left+right", "top", ""
    is_band: bool
    bbox: tuple          # (x0, y0, x1, y1) em fracao 0..1


@dataclass
class GeoResult:
    score: float
    black_frac: float
    empty_frac: float
    regions: list

    def as_dict(self):
        d = asdict(self)
        d["regions"] = [asdict(r) if isinstance(r, Region) else r for r in self.regions]
        return d


class GeometricDetector:
    def __init__(self, work: int = 480, win: int = 17, flat_std: float = 7.0,
                 black_v: float = 15.0, min_black: float = 0.06, min_empty: float = 0.08,
                 band_span: float = 0.80, band_thick: float = 0.45):
        # black_v=15: so preto quase-puro (barra real ~0); exclui app dark-theme (~20+).
        # min_black=0.06: exclui barras de SISTEMA (status/navegacao ~3%), mantem barras de
        # erro (faixas laterais do dobravel nao expandido ~15-22%).
        self.work = work
        self.win = win
        self.flat_std = flat_std
        self.black_v = black_v
        self.min_black = min_black
        self.min_empty = min_empty
        self.band_span = band_span      # quanto a banda cobre ao longo da borda
        self.band_thick = band_thick    # espessura maxima (perpendicular) p/ ser "banda"

    # ---------- nucleo ----------
    def _load_gray(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        W0, H0 = img.size
        s = self.work / max(W0, H0)
        img_s = img.resize((max(1, int(W0 * s)), max(1, int(H0 * s))), Image.BILINEAR)
        arr = np.asarray(img_s).astype(np.float32)
        return img_s, arr.mean(axis=2)

    def _flat_mask(self, gray: np.ndarray):
        mean = ndimage.uniform_filter(gray, self.win)
        sq = ndimage.uniform_filter(gray * gray, self.win)
        std = np.sqrt(np.clip(sq - mean * mean, 0, None))
        return std < self.flat_std

    @staticmethod
    def _edges(ys, xs, H, W):
        e = []
        m = 2
        if xs.min() <= m: e.append("left")
        if xs.max() >= W - 1 - m: e.append("right")
        if ys.min() <= m: e.append("top")
        if ys.max() >= H - 1 - m: e.append("bottom")
        return e

    def detect(self, image_path: str):
        img_s, gray = self._load_gray(image_path)
        H, W = gray.shape
        total = H * W
        flat = self._flat_mask(gray)
        black = gray < self.black_v

        heat = np.zeros((H, W), dtype=np.float32)
        regions: list[Region] = []
        black_frac = empty_frac = 0.0

        for tipo, base, minarea, weight in [
            ("black_region", flat & black, self.min_black, 1.0),
            ("empty_space", flat & ~black, self.min_empty, 0.55),
        ]:
            lbl, n = ndimage.label(base)
            for i in range(1, n + 1):
                comp = lbl == i
                area = comp.sum() / total
                if area < minarea:
                    continue
                ys, xs = np.where(comp)
                bh = (ys.max() - ys.min() + 1) / H
                bw = (xs.max() - xs.min() + 1) / W
                edges = self._edges(ys, xs, H, W)
                # banda: toca borda, e fina numa direcao e longa na outra
                vert_band = bw <= self.band_thick and bh >= self.band_span and (("left" in edges) or ("right" in edges))
                horz_band = bh <= self.band_thick and bw >= self.band_span and (("top" in edges) or ("bottom" in edges))
                is_band = vert_band or horz_band
                # empty_space SO conta se for banda de borda (evita o fundo branco da tela)
                if tipo == "empty_space" and not is_band:
                    continue
                # black_region: conta se for banda OU bloco grande tocando >=2 bordas
                if tipo == "black_region" and not (is_band or (area > 0.06 and len(edges) >= 2)):
                    continue
                regions.append(Region(tipo, round(float(area), 4), round(float(gray[comp].mean()), 1),
                                       "+".join(edges), bool(is_band),
                                       (round(xs.min()/W, 3), round(ys.min()/H, 3),
                                        round(xs.max()/W, 3), round(ys.max()/H, 3))))
                heat[comp] = max(weight, 0.0)
                if tipo == "black_region":
                    black_frac += area
                else:
                    empty_frac += area

        score = float(min(1.0, black_frac * 1.0 + empty_frac * 0.6))
        return GeoResult(round(score, 4), round(float(black_frac), 4),
                         round(float(empty_frac), 4), regions), heat, img_s

    # ---------- visualizacao ----------
    def save_overlay(self, image_path: str, out_path: Path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        res, heat, img_s = self.detect(image_path)
        vis = np.asarray(img_s) / 255.0
        fig, ax = plt.subplots(1, 2, figsize=(9, 4.8))
        ax[0].imshow(vis); ax[0].set_title(Path(image_path).name[:34]); ax[0].axis("off")
        ax[1].imshow(vis); ax[1].imshow(heat, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        desc = ", ".join(f"{r.tipo}({r.bordas or 'bloco'})" for r in res.regions) or "nada"
        ax[1].set_title(f"score={res.score:.2f} | {desc}"[:60]); ax[1].axis("off")
        fig.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110); plt.close(fig)
        return res
