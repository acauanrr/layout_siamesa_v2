"""Backbone DINOv2 ViT-S/14 (congelado) + pre-processamento de imagem.

Decisoes de design (ver docs/DESIGN.md):
- O backbone e CONGELADO. Com ~360 imagens, fine-tunar 22M de parametros faria overfit
  imediato. Usamos o DINOv2 como extrator de features e treinamos so uma cabeca leve.
- PRE-PROCESSAMENTO: dois modos (config backbone.preprocess), ver geometry.py:
    * "resize": resize anamorfico direto para 518x518 (nao injeta bordas; espreme aspecto).
    * "pad":    padding cinza ate quadrado + resize (preserva a geometria do erro). As
                estatisticas de patch sao calculadas SO na regiao de conteudo (mascara).
- Saida de features: token CLS (384) e, se use_patch_stats, concatena media+std dos patch
  tokens de CONTEUDO (1152). Os patch tokens crus tambem sao expostos (localizacao).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import timm

from .geometry import IMAGENET_MEAN_255, preprocess_image, content_patch_mask  # re-export
from PIL import Image

DINOV2_MODEL = "vit_small_patch14_dinov2.lvd142m"
DEFAULT_SIZE = 518  # 37x37 patches de 14px
EMBED_DIM = 384


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@dataclass
class BackboneConfig:
    model_name: str = DINOV2_MODEL
    size: int = DEFAULT_SIZE
    use_patch_stats: bool = False
    preprocess: str = "resize"                 # "resize" | "pad"
    pad_color: tuple = field(default_factory=lambda: IMAGENET_MEAN_255)
    device: str = "cuda"
    amp: bool = True


class DinoV2Backbone(nn.Module):
    """Wrapper congelado do DINOv2. Em eval(), sem grad, devolve embeddings."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(cfg.model_name, pretrained=True, num_classes=0)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.model.num_features
        self.out_dim = self.embed_dim * (3 if cfg.use_patch_stats else 1)
        self.to(cfg.device)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def preprocess(self, img: Image.Image):
        """PIL -> (tensor, mascara_de_patch). Usa o modo configurado."""
        return preprocess_image(img, self.cfg.size, self.cfg.preprocess, tuple(self.cfg.pad_color))

    @torch.no_grad()
    def _masked_stats(self, patches: torch.Tensor, mask: torch.Tensor | None):
        """patches: [B,N,C]; mask: [B,N] bool ou None -> (mean[B,C], std[B,C])."""
        if mask is None:
            return patches.mean(1), patches.std(1)
        m = mask.to(patches.device).unsqueeze(-1).float()      # [B,N,1]
        cnt = m.sum(1).clamp(min=1.0)                          # [B,1]
        mean = (patches * m).sum(1) / cnt
        var = (((patches - mean.unsqueeze(1)) ** 2) * m).sum(1) / cnt
        return mean, var.clamp(min=0).sqrt()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B,3,H,W] normalizado -> [B, out_dim] embeddings (nao normalizados).
        patch_mask: [B,N] bool dos patches de conteudo (so usado se use_patch_stats)."""
        x = x.to(self.device, non_blocking=True)
        use_amp = self.cfg.amp and self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            if not self.cfg.use_patch_stats:
                feat = self.model(x)
            else:
                tokens = self.model.forward_features(x)        # [B, prefix+N, C]
                # prefix = CLS (+ registers nos modelos reg4). Pula TODOS p/ as stats de patch
                # refletirem so os tokens espaciais (senao os 4 registers poluem mean/std).
                n_prefix = getattr(self.model, "num_prefix_tokens", 1)
                cls, patches = tokens[:, 0], tokens[:, n_prefix:]
                mean, std = self._masked_stats(patches, patch_mask)
                feat = torch.cat([cls, mean, std], dim=-1)
        return feat.float()

    @torch.no_grad()
    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W] -> [B, N, C] patch tokens crus (para localizacao PatchCore)."""
        x = x.to(self.device, non_blocking=True)
        use_amp = self.cfg.amp and self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            tokens = self.model.forward_features(x)
        n_prefix = getattr(self.model, "num_prefix_tokens", 1)
        return tokens[:, n_prefix:].float()
