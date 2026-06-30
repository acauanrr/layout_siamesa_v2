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
    use_patch_stats: bool = True      # VALIDADO: CLS+mean+std dos patches (1152-d), melhor p/ erro espacial
    preprocess: str = "pad"           # VALIDADO: "pad" (cinza preservando aspecto) | "resize" (anamorfico)
    batch_size: int = 16


@dataclass
class SyntheticCfg:
    enabled: bool = True
    n_variants: int = 4
    max_errors_per_image: int = 2
    seed: int = 0
    # --- REFLOW (negativos com layout legitimo; portado do projeto legado) ---------------
    # Variantes LIMPAS (label 0, category=clean) de mudanca de layout legitima (scroll,
    # dual-pane, outro aspect-ratio, reflow de espacamento). Atacam (a) o falso-positivo
    # estrutural (modelo confunde "tela diferente" com "errada") e (b) o confound de
    # resolucao PELO LADO LIMPO (ar_relayout tira a limpa da resolucao 2076x2152). Ver
    # src/siamese/reflow.py e docs/DESIGN.md. No legado foi o maior ganho (AUROC 0.62->0.80).
    reflow_clean: bool = True
    n_reflow_variants: int = 4         # variantes de reflow por imagem limpa de treino
    max_reflow_ops: int = 2            # compoe 1..N operadores de reflow por variante
    reflow_ops: dict = field(default_factory=lambda: {
        "scroll_shift": 1.0, "two_pane": 0.6, "ar_relayout": 1.0, "band_jitter": 1.0})
    p_reflow_pos: float = 0.0          # LEGADO/ablacao: fracao de erros sinteticos injetados
                                       # SOBRE um layout reflowado (impede atalho "layout mudou
                                       # => erro"). No legado deu resultado NEGATIVO -> default OFF.
    benign_augment: bool = True        # VALIDADO: round-trip de resolucao + jitter foto-metrico nas
                                       # limpas (remove o atalho de nitidez/resolucao)
    reflow_match_error_ar: bool = False  # Fase 2.4: ar_relayout mira a distribuicao de AR dos ERROS
                                       # (mediana near-square 0.96) em vez de aspecto aleatorio
                                       # U(0.5,2.0). Cobre o bucket near-square sub-representado
                                       # (onde o modelo falso-alarmava nas limpas) e casa o confound
                                       # nas resolucoes dos erros. Alvos = AR dos erros de TREINO.


@dataclass
class HeadCfg:
    hidden: int = 256
    proj_dim: int = 64                # VALIDADO (grid + 1-SE multi-seed): cabeca menor generaliza melhor
    p_drop: float = 0.3


@dataclass
class TrainCfg:
    epochs: int = 500             # teto alto; o early-stop (patience) decide quando parar
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    temperature: float = 0.1
    loss: str = "supcon"          # supcon | contrastive | triplet (batch-hard online mining)
    triplet_margin: float = 0.5   # margem da triplet loss (usado quando loss='triplet')
    aux_weight: float = 0.3       # peso do cabecalho auxiliar de classificacao
    multiclass: bool = True       # True: clusteriza por CATEGORIA (clean + 4 erros); aux=softmax,
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
    patience: int = 80


@dataclass
class DecisionCfg:
    k_prototypes: int = 3
    gate_method: str = "prototype" # decisor do GATE (Estagio 1): "prototype" (k-means do cluster
                                   # limpo) | "knn" (vizinhos mais proximos as limpas de treino —
                                   # melhor p/ manifold limpo multimodal). A fusao com a cabeca aux
                                   # e a calibracao do limiar sao identicas nos dois.
    knn_k: int = 5                 # k dos decisores k-NN (gate e/ou Estagio 2), quando ativos
    objective: str = "specificity" # VALIDADO specificity-first. "f1" = ponto balanceado (comparacao)
                                   # "precision" = alta precisao (usa target_precision)
                                   # "specificity" = ponto specificity-first (usa target_specificity)
    target_precision: float = 0.95
    target_specificity: float = 0.80   # usado quando objective=specificity: menor limiar cuja
                                       # especificidade na calibracao >= alvo (gerente quer poucos
                                       # falso-alarmes); reporta o recall obtido nesse ponto.
    # CONJUNTO DE CALIBRACAO da fusao LogReg + limiar (SEMPRE na val — nunca no teste):
    #   "confound_free" (PADRAO): limpas-val reais (0) + val_synth erros (1) + val_reflow limpas (0).
    #       Muitos mais negativos limpos + positivos LIVRES de confound -> limiar ESTAVEL (corrige
    #       o FPR 0.88 vindo de calibrar em so 26 limpas). E' a licao do legado (calibrar na val
    #       sintetica). Cai para "real_val" se as sondas sinteticas nao existirem.
    #   "real_val" (legado): so a val real (26 limpas + erros reais) — ponto de operacao instavel.
    calibrate_on: str = "confound_free"
    stage2_method: str = "knn"         # decisor CANONICO do Estagio 2 (VALIDADO jun/2026): "knn"|"prototype"
                                       # (protótipo de categoria no espaco z) | "knn" (k-NN aos
                                       # erros de treino por categoria) | "aux" (argmax da cabeca
                                       # aux). Os demais viram diagnostico/ablacao em evaluate.py.
    coarse_taxonomy: bool = True       # avalia o Estagio 2 TAMBEM na taxonomia grossa (2 super-classes
                                       # de erro por nao-colisao) — reportada como PRIMARIA por ter
                                       # poder estatistico; a fina (4 classes) vira secundaria.


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
            test_frac=raw.get("test_frac", 0.24),
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
