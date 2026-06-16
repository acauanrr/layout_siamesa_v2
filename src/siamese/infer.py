"""Inferencia: imagem nova -> p(erro) + decisao, usando o bundle de decisao salvo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .backbone import DinoV2Backbone, BackboneConfig, load_image
from .train import load_model
from .decision import PrototypeDecider


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Predictor:
    def __init__(self, models_dir: str | Path, device: str | None = None):
        models_dir = Path(models_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        d = np.load(models_dir / "decision.npz")
        self.use_patch_stats = bool(d["use_patch_stats"][0])
        self.size = int(d["size"][0])
        self.preprocess = str(d["preprocess"][0]) if "preprocess" in d else "resize"
        self.prototypes = d["prototypes"]
        self.fusion_coef = d["fusion_coef"]
        self.fusion_intercept = float(d["fusion_intercept"][0])
        self.threshold = float(d["threshold"][0])
        self.target_precision = float(d["target_precision"][0])

        self.backbone = DinoV2Backbone(BackboneConfig(
            size=self.size, use_patch_stats=self.use_patch_stats,
            preprocess=self.preprocess, device=self.device))
        self.model = load_model(models_dir / "siamese_head.pt", device=self.device)
        self.decider = PrototypeDecider(self.prototypes, self.threshold, self.target_precision)

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        img = load_image(image_path)
        x, mask = self.backbone.preprocess(img)
        x = x.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0)
        emb = self.backbone(x, mask)                 # [1, D]
        z, aux = self.model(emb)
        z = z.cpu().numpy()
        aux_logit = float(aux.cpu().numpy()[0])
        score_proto = float(self.decider.scores(z)[0])
        # fusao = LogisticRegression.predict_proba (= sigmoid do logit linear); p_erro e
        # a probabilidade fundida [score_proto, aux_logit]. O limiar salvo (self.threshold)
        # foi escolhido na VAL para a precisao-alvo, NA MESMA escala de probabilidade.
        logit = self.fusion_coef[0] * score_proto + self.fusion_coef[1] * aux_logit + self.fusion_intercept
        p_erro = float(_sigmoid(logit))
        return {
            "file": Path(image_path).name,
            "p_erro": p_erro,
            "decisao": "ERRO" if p_erro > self.threshold else "sem_erro",
            "limiar": self.threshold,
            "score_prototipo": score_proto,
            "aux_logit": aux_logit,
        }
