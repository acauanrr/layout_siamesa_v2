"""Configuracao do projeto (dataclasses + YAML)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml


@dataclass
class Paths:
    input_dir: str = "data/input"
    splits_dir: str = "data/splits"
    emb_dir: str = "artifacts/embeddings"
    models_dir: str = "artifacts/models"
    reports_dir: str = "artifacts/reports"


@dataclass
class BackboneCfg:
    model_name: str = "vit_small_patch14_dinov2.lvd142m"
    size: int = 518
    use_patch_stats: bool = False
    preprocess: str = "resize"        # "resize" (anamorfico) | "pad" (cinza preservando aspecto)
    batch_size: int = 16


@dataclass
class SyntheticCfg:
    enabled: bool = True
    n_variants: int = 4
    max_errors_per_image: int = 2
    seed: int = 0


@dataclass
class HeadCfg:
    hidden: int = 256
    proj_dim: int = 128
    p_drop: float = 0.3


@dataclass
class TrainCfg:
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    temperature: float = 0.1
    loss: str = "supcon"          # supcon | contrastive
    aux_weight: float = 0.3       # peso do cabecalho auxiliar de classificacao
    multiclass: bool = True       # True: clusteriza por CATEGORIA (clean + 6 erros); aux=softmax,
                                  # SupCon por categoria, batches balanceados por classe.
                                  # False: detector BINARIO legado (erro/sem-erro), aux=sigmoid.
    use_real_errors: bool = True   # incluir erros REAIS no treino (alem dos sinteticos)
    use_synthetic: bool = True     # incluir erros SINTETICOS (anti-confound)
    balance_batches: bool = True
    # criterio de early-stop / salvamento do melhor checkpoint. AGORA respeitado em train.py
    # (antes era ignorado: a formula max(proto,aux)+0.5*cat_f1 era hardcoded). Opcoes:
    #   val_synth_gate        gate LIVRE DE CONFOUND na val (aux head; metrica de PRODUCAO) [PADRAO]
    #   val_synth_gate_proto  idem, via score de prototipo
    #   val_synth_gate_max    max(proto,aux) livre de confound (legado)
    #   val_synth_gate+cat_f1 0.5*gate_livre_confound + 0.5*F1_categoria (deteccao+cluster)
    #   val_ap                AP do gate CONFUNDIDO (legado; pode rastrear resolucao)
    #   val_ap_aux            AP da cabeca auxiliar (confundido)
    #   val_cat_f1            so a clusterizacao por categoria (Estagio 2)
    early_stop_metric: str = "val_synth_gate"
    max_oversample_per_class: int = 0  # teto de repeticoes/classe por epoca no sampler (0 = sem teto)
    patience: int = 40


@dataclass
class DecisionCfg:
    k_prototypes: int = 1
    objective: str = "f1"          # "f1" = ponto balanceado (padrao p/ acuracia/comparacao)
                                   # "precision" = alta precisao (usa target_precision)
    target_precision: float = 0.95


@dataclass
class Config:
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.24
    paths: Paths = field(default_factory=Paths)
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    synthetic: SyntheticCfg = field(default_factory=SyntheticCfg)
    head: HeadCfg = field(default_factory=HeadCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    decision: DecisionCfg = field(default_factory=DecisionCfg)

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return Config(
            seed=raw.get("seed", 42),
            val_frac=raw.get("val_frac", 0.15),
            test_frac=raw.get("test_frac", 0.15),
            paths=Paths(**raw.get("paths", {})),
            backbone=BackboneCfg(**raw.get("backbone", {})),
            synthetic=SyntheticCfg(**raw.get("synthetic", {})),
            head=HeadCfg(**raw.get("head", {})),
            train=TrainCfg(**raw.get("train", {})),
            decision=DecisionCfg(**raw.get("decision", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False, allow_unicode=True)
