"""Inferencia: imagem nova -> p(erro) + decisao, usando o bundle de decisao salvo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .backbone import DinoV2Backbone, BackboneConfig, load_image
from .train import load_model
from .decision import (PrototypeDecider, KNNDecider, KNNCategoryClassifier, assign_category)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Predictor:
    def __init__(self, models_dir: str | Path, device: str | None = None,
                 route_foldable: bool = True):
        models_dir = Path(models_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        d = np.load(models_dir / "decision.npz", allow_pickle=False)
        self.use_patch_stats = bool(d["use_patch_stats"][0])
        self.size = int(d["size"][0])
        self.preprocess = str(d["preprocess"][0]) if "preprocess" in d else "resize"
        # backbone: usa o model_name SALVO no bundle (crucial p/ reg4/large — dims != do default S).
        # Bundles antigos sem a chave caem no default do BackboneConfig (S), como antes.
        self.model_name = str(d["model_name"][0]) if "model_name" in d else None
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
        # método de decisão (default prototype p/ bundles antigos sem a chave)
        self.gate_method = str(d["gate_method"][0]) if "gate_method" in d else "prototype"
        self.stage2_method = str(d["stage2_method"][0]) if "stage2_method" in d else "prototype"
        self.knn_k = int(d["knn_k"][0]) if "knn_k" in d else 5
        # ROTEAMENTO POR DOMÍNIO (docs/RELATORIO_FOLDABLE.md): o bucket near-square (foldable) usa o
        # gate de PROTÓTIPO + um limiar foldable (a fusão+limiar global dão falso-alarme lá: espec
        # 0.51 -> 0.68). route_foldable liga/desliga; NaN no bundle (val sem near-square) -> não roteia.
        self.route_foldable = bool(route_foldable)
        self.foldable_threshold = (float(d["foldable_proto_threshold"][0])
                                   if "foldable_proto_threshold" in d else float("nan"))
        self.foldable_ar_lo = float(d["foldable_ar_lo"][0]) if "foldable_ar_lo" in d else 0.85
        self.foldable_ar_hi = float(d["foldable_ar_hi"][0]) if "foldable_ar_hi" in d else 1.18

        _bcfg = BackboneConfig(size=self.size, use_patch_stats=self.use_patch_stats,
                               preprocess=self.preprocess, device=self.device)
        if self.model_name:
            _bcfg.model_name = self.model_name
        self.backbone = DinoV2Backbone(_bcfg)
        self.model = load_model(models_dir / "siamese_head.pt", device=self.device)
        # GATE: reconstrói EXATAMENTE o decisor avaliado (k-NN sobre as refs cruas, ou protótipo).
        # Crucial: a fusão+limiar foram calibrados sobre ESTE score — usar o decisor errado
        # miscalibraria a decisão silenciosamente.
        if self.gate_method == "knn":
            self.decider = KNNDecider(self.prototypes, k=self.knn_k)
            self.decider.threshold = self.threshold
        else:
            self.decider = PrototypeDecider(self.prototypes, self.threshold, self.target_precision)
        # ESTÁGIO 2: k-NN de categoria (refs de erro cruas) ou protótipo de categoria.
        self.cat_knn = (KNNCategoryClassifier(self.cat_prototypes, self.cat_proto_ids, k=self.knn_k)
                        if (self.stage2_method == "knn" and self.cat_prototypes is not None
                            and len(self.cat_prototypes)) else None)

    def _aux_err(self, aux: np.ndarray) -> float:
        """Score escalar de erro (mesma definicao usada para calibrar a fusao em evaluate.py).
        aux: [1, C] (multi-classe) ou [1] (binario) -> achatamos a unica linha."""
        a = aux.ravel()
        if self.multiclass:
            e = np.exp(a - a.max())
            p = e / e.sum()
            return float(1.0 - p[0])     # 1 - P(clean)
        return float(a[0])               # logit binario

    @staticmethod
    def _gate_decision(score_proto: float, p_erro: float, ar: float, *, route_foldable: bool,
                       foldable_threshold: float, foldable_ar_lo: float, foldable_ar_hi: float,
                       threshold: float) -> tuple[bool, str, float, bool]:
        """Roteamento por domínio (docs/RELATORIO_FOLDABLE.md). Puro/testável: near-square
        (foldable) decide pelo score de PROTÓTIPO + limiar foldable; demais pelo FUNDIDO + limiar
        global. Devolve (is_err, gate_usado, limiar, near_square)."""
        near_square = (route_foldable and not np.isnan(foldable_threshold)
                       and foldable_ar_lo <= ar <= foldable_ar_hi)
        if near_square:
            return bool(score_proto >= foldable_threshold), "foldable_prototipo", float(foldable_threshold), True
        return bool(p_erro > threshold), "global_fusao", float(threshold), False

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
        # ROTEAMENTO POR DOMÍNIO (AR NATIVO — load_image não redimensiona):
        w, h = img.size
        is_err, gate_usado, limiar, near_square = self._gate_decision(
            score_proto, p_erro, (w / h) if h else 0.0, route_foldable=self.route_foldable,
            foldable_threshold=self.foldable_threshold, foldable_ar_lo=self.foldable_ar_lo,
            foldable_ar_hi=self.foldable_ar_hi, threshold=self.threshold)
        out = {
            "file": Path(image_path).name,
            "p_erro": p_erro,
            "decisao": "ERRO" if is_err else "sem_erro",
            "gate": gate_usado,
            "near_square": bool(near_square),
            "limiar": limiar,
            "score_prototipo": score_proto,
            "aux_err": aux_err,
            "categoria": None,
            "scores_por_categoria": None,
        }
        # ESTAGIO 2 — se ERRO, atribui a categoria (k-NN de categoria ou protótipo mais próximo)
        if is_err and self.multiclass and self.cat_prototypes is not None and len(self.cat_prototypes):
            _name = lambda c: (self.categories[int(c)] if int(c) < len(self.categories) else str(int(c)))
            if self.cat_knn is not None:
                cid = int(self.cat_knn.predict(z)[0])
                sc, classes = self.cat_knn.class_scores(z)        # média top-k sim por classe
                best = {_name(c): float(s) for c, s in zip(classes, sc.ravel())}
            else:
                cid = int(assign_category(z, self.cat_prototypes, self.cat_proto_ids)[0])
                from sklearn.preprocessing import normalize
                sims = (normalize(z) @ self.cat_prototypes.T).ravel()   # sim a cada protótipo
                best = {}
                for c, s in zip(self.cat_proto_ids, sims):
                    best[_name(c)] = max(best.get(_name(c), -1.0), float(s))
            out["categoria"] = self.categories[cid] if cid < len(self.categories) else str(cid)
            out["scores_por_categoria"] = dict(sorted(best.items(), key=lambda kv: -kv[1]))
        return out
