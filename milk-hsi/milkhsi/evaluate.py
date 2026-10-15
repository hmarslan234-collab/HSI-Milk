"""Nested, grouped cross-validation for the scikit-learn models."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from .metrics import aggregate_by_sample, score


def _inner_k(y, groups, k_max):
    per_class = pd.Series(y).groupby(groups).first().value_counts().min()
    return int(max(2, min(k_max, per_class)))


def expand_proba(proba_local: np.ndarray, local_classes: np.ndarray, n_classes: int) -> np.ndarray:
    full = np.zeros((proba_local.shape[0], n_classes), dtype=np.float64)
    full[:, local_classes] = proba_local
    return full


def run_sklearn(pipe, grid, X, y, groups, splits, n_classes, cfg, log=print):
    """Return (per-fold metrics DataFrame, out-of-fold probabilities for repeat 0, best params)."""
    rows, best = [], []
    oof = np.full((len(y), n_classes), np.nan)
    for sp in splits:
        tr, te = np.asarray(sp["train"]), np.asarray(sp["test"])
        local = np.unique(y[tr])
        ytr = np.searchsorted(local, y[tr])           # contiguous labels for xgboost
        k = _inner_k(ytr, groups[tr], cfg["cv"]["inner_folds"])
        inner = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg["cv"]["seed"])
        gs = GridSearchCV(clone(pipe), grid, scoring="f1_macro", cv=inner, n_jobs=1, refit=True)
        t0 = time.time()
        gs.fit(X[tr], ytr, groups=groups[tr])
        fit_s = time.time() - t0
        t0 = time.time()
        p = expand_proba(gs.predict_proba(X[te]), local, n_classes)
        inf_ms = 1000 * (time.time() - t0) / len(te)
        m_roi = score(y[te], p)
        ys, ps = aggregate_by_sample(groups[te], y[te], p)
        m_smp = score(ys, ps)
        row = {"repeat": sp["repeat"], "fold": sp["fold"], "fit_s": fit_s, "infer_ms_per_roi": inf_ms,
               **{f"roi_{k}": v for k, v in m_roi.items()},
               **{f"sample_{k}": v for k, v in m_smp.items()}}
        rows.append(row)
        best.append({"repeat": sp["repeat"], "fold": sp["fold"], **gs.best_params_})
        if sp["repeat"] == 0:
            oof[te] = p
        log(f"   rep {sp['repeat']} fold {sp['fold']}: roi acc {m_roi['accuracy']:.3f} "
            f"f1 {m_roi['f1_macro']:.3f} | sample acc {m_smp['accuracy']:.3f}  ({fit_s:.0f}s)")
    return pd.DataFrame(rows), oof, pd.DataFrame(best)
