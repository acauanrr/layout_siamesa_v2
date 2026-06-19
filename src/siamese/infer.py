"""Inferencia: imagem nova -> p(erro) + decisao, usando o bundle de decisao salvo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .backbone import DinoV2Backbone, BackboneConfig, load_image
from .train import load_model
from .decision import PrototypeDecider, assign_category


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Predictor:
    def __init__(self, models_dir: str | Path, device: str | None = None):
        models_dir = Path(models_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        d = np.load(models_dir / "decision.npz", allow_pickle=False)
        self.use_patch_stats = bool(d["use_patch_stats"][0])
        self.size = int(d["size"][0])
        self.preprocess = str(d["preprocess"][0]) if "preprocess" in d else "resize"
        self.prototypes = d["prototypes"]
        self.fusion_coef = d["fusion_coef"]
        self.fusion_intercept = float(d["fusion_intercept"][0])
        self.threshold = float(d["threshold"][0])
        self.target_precision = float(d["target_precision"][0])
        # ESTAGIO 2 (multi-cluster): categorias + protótipos por categoria
        self.multiclass = bool(d["multiclass"][0]) if "multiclass" in d else False
        self.categories = [str(c) for c in d["categories"]] if "categories" in d else ["clean", "error"]
        self.cat_prototypes = d["cat_prototypes"] if "cat_prototypes" in d else None
        self.cat_proto_ids = d["cat_proto_ids"] if "cat_proto_ids" in d else None

        self.backbone = DinoV2Backbone(BackboneConfig(
            size=self.size, use_patch_stats=self.use_patch_stats,
            preprocess=self.preprocess, device=self.device))
        self.model = load_model(models_dir / "siamese_head.pt", device=self.device)
        self.decider = PrototypeDecider(self.prototypes, self.threshold, self.target_precision)

    def _aux_err(self, aux: np.ndarray) -> float:
        """Score escalar de erro (mesma definicao usada para calibrar a fusao em evaluate.py).
        aux: [1, C] (multi-classe) ou [1] (binario) -> achatamos a unica linha."""
        a = aux.ravel()
        if self.multiclass:
            e = np.exp(a - a.max())
            p = e / e.sum()
            return float(1.0 - p[0])     # 1 - P(clean)
        return float(a[0])               # logit binario

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        img = load_image(image_path)
        x, mask = self.backbone.preprocess(img)
        x = x.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0)
        emb = self.backbone(x, mask)                 # [1, D]
        z, aux = self.model(emb)
        z = z.cpu().numpy()
        aux = aux.cpu().numpy()
        aux_err = self._aux_err(aux)
        score_proto = float(self.decider.scores(z)[0])
        # ESTAGIO 1 — gate "tem erro?": fusao calibrada [score_proto, aux_err] -> p_erro,
        # comparada ao limiar fixado na VAL para a precisao-alvo.
        logit = self.fusion_coef[0] * score_proto + self.fusion_coef[1] * aux_err + self.fusion_intercept
        p_erro = float(_sigmoid(logit))
        is_err = p_erro > self.threshold
        out = {
            "file": Path(image_path).name,
            "p_erro": p_erro,
            "decisao": "ERRO" if is_err else "sem_erro",
            "limiar": self.threshold,
            "score_prototipo": score_proto,
            "aux_err": aux_err,
            "categoria": None,
            "scores_por_categoria": None,
        }
        # ESTAGIO 2 — se ERRO, atribui a categoria pelo protótipo mais proximo
        if is_err and self.multiclass and self.cat_prototypes is not None and len(self.cat_prototypes):
            cid = int(assign_category(z, self.cat_prototypes, self.cat_proto_ids)[0])
            out["categoria"] = self.categories[cid] if cid < len(self.categories) else str(cid)
            # similaridade (1 - dist cosseno) a cada protótipo de categoria, p/ transparencia
            from sklearn.preprocessing import normalize
            sims = (normalize(z) @ self.cat_prototypes.T).ravel()
            best = {}
            for c, s in zip(self.cat_proto_ids, sims):
                name = self.categories[int(c)] if int(c) < len(self.categories) else str(int(c))
                best[name] = max(best.get(name, -1.0), float(s))
            out["scores_por_categoria"] = dict(sorted(best.items(), key=lambda kv: -kv[1]))
        return out
