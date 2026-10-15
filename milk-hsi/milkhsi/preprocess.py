"""Spectral preprocessing as scikit-learn transformers.

Order (Fig. 5 of the paper): band trimming -> Savitzky-Golay smoothing ->
scatter correction (SNV or MSC) -> min-max scaling.

Band trimming, SG and SNV act on each spectrum on its own, so they cannot leak
information between training and test data. MSC (reference = mean training
spectrum) and min-max (per-band min/max) are *fitted*, so they live inside the
model pipeline and are re-fitted on the training part of every fold.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


def trim_mask(wl: np.ndarray, wl_min: float, wl_max: float) -> np.ndarray:
    return (wl >= wl_min) & (wl <= wl_max)


def savgol(X: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    return savgol_filter(X, window_length=window, polyorder=poly, axis=-1)


def snv(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, keepdims=True)
    return (X - mu) / np.maximum(sd, 1e-8)


class SNV(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return snv(np.asarray(X, dtype=np.float32))


class MSC(BaseEstimator, TransformerMixin):
    """Multiplicative scatter correction against the mean training spectrum."""

    def fit(self, X, y=None):
        self.ref_ = np.asarray(X, dtype=np.float64).mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X)
        A = np.vstack([self.ref_, np.ones_like(self.ref_)]).T
        coef, *_ = np.linalg.lstsq(A, X.T, rcond=None)   # slope, offset per spectrum
        b, a = coef[0], coef[1]
        out = (X - a[:, None]) / np.where(np.abs(b) < 1e-8, 1e-8, b)[:, None]
        return out.astype(np.float32)


class BandMinMax(BaseEstimator, TransformerMixin):
    """Min-max scaling to [0, 1] per band, fitted on training data."""

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.min_ = X.min(axis=0)
        self.rng_ = np.maximum(X.max(axis=0) - self.min_, 1e-8)
        return self

    def transform(self, X):
        return ((np.asarray(X) - self.min_) / self.rng_).astype(np.float32)


def fixed_steps(X: np.ndarray, wl: np.ndarray, cfg: dict):
    """Per-spectrum steps that need no fitting: trim + SG. Works on (..., B)."""
    keep = trim_mask(wl, cfg["wl_min"], cfg["wl_max"])
    X = X[..., keep]
    X = savgol(X, cfg["sg_window"], cfg["sg_poly"])
    return X.astype(np.float32), wl[keep]


def scatter_step(name: str):
    name = (name or "none").lower()
    if name == "snv":
        return SNV()
    if name == "msc":
        return MSC()
    return None
