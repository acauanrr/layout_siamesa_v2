"""Extracao e cache de embeddings DINOv2.

FONTE DA VERDADE = data/processed/. O modelo treina/valida/testa SOBRE o que esta
materializado em processed/<split>/<fonte>/<categoria>/ — incluindo correcoes/ajustes
manuais feitos nessa pasta. `scan_processed` varre a arvore e deriva (split, fonte,
categoria, label) do CAMINHO e os metadados (form factor, orientacao, ticket/grupo) do
NOME do arquivo. `extract_rows`/`extract_processed` cacheiam os embeddings por (split, fonte).

Como o backbone e congelado, os embeddings sao FIXOS -> cachear em disco torna o treino da
cabeca siamesa quase instantaneo.

Cache: artifacts/embeddings/<split>.npz (reais) e <split>_synth.npz (sinteticos) com:
  emb   [N, D] float32   (embeddings do backbone, nao normalizados)
  label [N]    int64     (0=clean, 1=erro)
  category [N] <U...     (slug da categoria)
  group [N]    <U...     (ticket, para amostragem sem vazamento)
  path  [N]    <U...     (caminho ABSOLUTO em data/processed/)
  kind, form_factor, orientation, source [N] <U...
  (sinteticos: + applied [N] <U... e parent [N] <U...)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .backbone import DinoV2Backbone, BackboneConfig, load_image
from .geometry import preprocess_image
from .manifest import _parse_meta, _group_key

_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_SPLITS = ("train", "val", "test")
_SOURCES = ("real", "synthetic")
# (split, source) -> nome do arquivo de cache
NPZ_NAME = {
    ("train", "real"): "train.npz", ("val", "real"): "val.npz", ("test", "real"): "test.npz",
    ("train", "synthetic"): "train_synth.npz",
    ("val", "synthetic"): "val_synth.npz", ("test", "synthetic"): "test_synth.npz",
}


class _ImageManifestDataset(Dataset):
    """Nao guarda o modelo (evita enviar CUDA aos workers); so os params de preproc."""
    def __init__(self, rows: list[dict], size: int, mode: str, pad_color):
        self.rows = rows
        self.size = size
        self.mode = mode
        self.pad_color = tuple(pad_color)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        img = load_image(self.rows[i]["path"])
        x, mask = preprocess_image(img, self.size, self.mode, self.pad_color)
        return x, mask, i


def read_manifest(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def scan_processed(processed_dir: Path) -> list[dict]:
    """Varre data/processed/<split>/<fonte>/<categoria>/* (FONTE DA VERDADE) e devolve
    registros (sem embedding). O split/fonte/categoria/label vem do CAMINHO; os metadados
    de confound (form_factor, orientation, kind, group) vem do NOME do arquivo. Assim,
    mover/remover/corrigir arquivos em processed/ e' honrado sem precisar de manifesto."""
    processed_dir = Path(processed_dir)
    rows: list[dict] = []
    for split in _SPLITS:
        for source in _SOURCES:
            base = processed_dir / split / source
            if not base.is_dir():
                continue
            for catdir in sorted(base.iterdir()):
                if not catdir.is_dir():
                    continue
                category = catdir.name
                label = 0 if category == "clean" else 1
                for p in sorted(catdir.iterdir()):
                    if p.suffix.lower() not in _EXTS:
                        continue
                    rec = {
                        "path": str(p.resolve()),
                        "label": label,
                        "category": category,
                        "group": _group_key(p, source),
                        "split": split,
                        "source": source,
                        **_parse_meta(p.name, source),
                    }
                    if source == "synthetic":
                        # nome = {parent_stem}__{tipo}__v{n}.png
                        parts = p.stem.split("__")
                        rec["applied"] = parts[1] if len(parts) >= 2 else ""
                        rec["parent"] = parts[0] if parts else ""
                        rec["kind"] = "synthetic"
                    rows.append(rec)
    # group consistente com o split: telas LIMPAS reagrupadas por SESSAO de captura (timestamp),
    # nao por arquivo. Mantem a unidade de bootstrap/CI = sessao (independencia real), igual ao
    # split de build_splits. (so timestamp aqui: deterministico e sem ler pixels.)
    from .manifest import clean_session_components
    clean = [r for r in rows if r.get("category") == "clean"]
    if clean:
        names = sorted({Path(r["path"]).name for r in clean})
        gmap = clean_session_components(names)
        for r in clean:
            r["group"] = gmap[Path(r["path"]).name]
    return rows


def extract_rows(
    rows: list[dict],
    out_npz: Path,
    backbone: DinoV2Backbone,
    *,
    batch_size: int = 16,
    num_workers: int = 8,
) -> dict:
    """Embeda uma lista de registros (cada um com 'path' + metadados) e salva o .npz."""
    ds = _ImageManifestDataset(rows, backbone.cfg.size, backbone.cfg.preprocess, backbone.cfg.pad_color)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    embs = np.zeros((len(rows), backbone.out_dim), dtype=np.float32)
    for x, mask, idx in tqdm(dl, desc=f"extract {out_npz.stem}"):
        embs[idx.numpy()] = backbone(x, mask).cpu().numpy()

    def col(name):
        return np.array([r.get(name, "") for r in rows])

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    data = dict(
        emb=embs,
        label=col("label").astype(np.int64),
        category=col("category"),
        group=col("group"),
        path=col("path"),
        kind=col("kind"),
        form_factor=col("form_factor"),
        orientation=col("orientation"),
        source=col("source"),
    )
    if any(r.get("source") == "synthetic" for r in rows):
        data["applied"] = col("applied")
        data["parent"] = col("parent")
    np.savez(out_npz, **data)
    return {"n": len(rows), "dim": backbone.out_dim, "out": str(out_npz)}


def extract_processed(
    processed_dir: Path,
    out_dir: Path,
    backbone: DinoV2Backbone,
    *,
    batch_size: int = 16,
    num_workers: int = 8,
) -> dict:
    """Varre processed/ e cacheia os embeddings por (split, fonte). Reais: train/val/test.npz;
    sinteticos materializados (train): train_synth.npz. (val/test_synth = sonda livre de
    confound, gerados por make_synthetic a partir de processed/{val,test}/real/clean.)"""
    out_dir = Path(out_dir)
    rows = scan_processed(processed_dir)
    if not rows:
        raise FileNotFoundError(f"Nenhuma imagem em {processed_dir}/ — rode export_processed.py antes.")
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["split"], r["source"]), []).append(r)
    summary = {}
    for key in sorted(groups):
        name = NPZ_NAME.get(key)
        if name is None:
            continue
        info = extract_rows(groups[key], out_dir / name, backbone,
                            batch_size=batch_size, num_workers=num_workers)
        summary[f"{key[0]}/{key[1]}"] = info["n"]
    return summary


def extract_split(
    manifest_csv: Path,
    out_npz: Path,
    backbone: DinoV2Backbone,
    *,
    batch_size: int = 16,
    num_workers: int = 8,
) -> dict:
    """LEGADO: embeda a partir de um CSV de split (caminhos apontando p/ data/input/).
    Mantido para o caminho binario legado e reuso por compare_preprocess.py."""
    return extract_rows(read_manifest(manifest_csv), out_npz, backbone,
                        batch_size=batch_size, num_workers=num_workers)


def load_embeddings(npz_path: Path) -> dict:
    # Fase 0: blindagem do teste. Carregar test.npz/test_synth.npz so e' permitido com a
    # trava --final-test liberada (ver siamese.protocol). Este e' o chokepoint por onde TODO
    # modelo consome embeddings -> grid_search/ablation/visualize ficam fisicamente impedidos
    # de tocar o teste durante a selecao (anti-snooping).
    from .protocol import guard_path
    z = np.load(guard_path(npz_path), allow_pickle=False)
    return {k: z[k] for k in z.files}
