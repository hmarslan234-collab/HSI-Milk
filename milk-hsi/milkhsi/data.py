"""Loading the extracted ROI dataset and building leakage-free splits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from .labels import add_task_columns
from .preprocess import fixed_steps


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_rois(cfg: dict, with_patches: bool = False):
    """Return X (n_roi x B'), wl', meta DataFrame and optionally patches.

    X has the fixed (fit-free) steps applied: band trimming and SG smoothing.
    Patches are returned raw and memory-mapped (they can be several GB);
    apply `patch_steps` to each mini-batch.
    """
    p = Path(cfg["paths"]["processed"])
    z = np.load(p / "rois.npz")
    meta = add_task_columns(pd.read_csv(p / "rois_meta.csv"))
    raw_wl = z["wavelengths"]
    X, wl = fixed_steps(z["spectra"].astype(np.float32), raw_wl, cfg["preprocess"])
    patches = None
    if with_patches:
        pf = p / "patches.npy"
        if not pf.exists():
            raise FileNotFoundError("patches.npy missing; run 02_extract_rois.py with roi.save_patches: true")
        patches = np.load(pf, mmap_mode="r")
    return X, wl, meta, patches


def patch_steps(P: np.ndarray, raw_wl: np.ndarray, cfg: dict) -> np.ndarray:
    """Trim + SG along the spectral axis of patches (N x h x w x B)."""
    out, _ = fixed_steps(np.asarray(P, dtype=np.float32), raw_wl, cfg["preprocess"])
    return out


def raw_wavelengths(cfg: dict) -> np.ndarray:
    return np.load(Path(cfg["paths"]["processed"]) / "rois.npz")["wavelengths"]


def encode(y: pd.Series):
    classes = sorted(y.unique())
    lut = {c: i for i, c in enumerate(classes)}
    return y.map(lut).to_numpy(), classes


def make_splits(meta: pd.DataFrame, cfg: dict) -> list[dict]:
    """Outer folds for every repeat, grouped by physical sample.

    One set of folds is shared by all tasks and all models, so every model is
    scored on exactly the same test samples (needed for the paired tests) and
    the multi-task network can be compared with the single-task ones.
    Splitting is done on samples, not ROIs, so all ROIs of a dish land on the
    same side. Folds are stratified by species x level class (the finest
    label); if some stratum has too few samples, a coarser key is used.
    Saved to results/splits.json.
    """
    f = Path(cfg["paths"]["results"]) / "splits.json"
    if f.exists():
        return json.loads(f.read_text())
    f.parent.mkdir(parents=True, exist_ok=True)
    groups = meta["sample_id"].to_numpy()
    k_max = cfg["cv"]["outer_folds"]
    keys = {"species x level": meta["species"] + "|" + meta["level_class"],
            "species x adulterant": meta["species"] + "|" + meta["adulterant"],
            "species": meta["species"]}
    for key_name, key in keys.items():
        y = key.to_numpy()
        n_min = pd.Series(y).groupby(groups).first().value_counts().min()
        k = int(min(k_max, n_min))
        if k >= 2:
            break
    if k < 2:
        raise ValueError("every species needs at least 2 physical samples for grouped CV "
                         f"(smallest stratum has {n_min})")
    print(f"outer CV: {k} folds x {cfg['cv']['repeats']} repeats, stratified by {key_name}")
    splits = []
    for r in range(cfg["cv"]["repeats"]):
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg["cv"]["seed"] + r)
        for fold, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
            assert not set(groups[tr]) & set(groups[te])
            splits.append({"repeat": r, "fold": fold, "train": tr.tolist(), "test": te.tolist()})
    f.write_text(json.dumps(splits))
    return splits
