"""Evaluation metrics: accuracy, macro P/R/F1, ROC-AUC, ECE, plus
aggregation of ROI predictions to one prediction per physical sample."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


def expected_calibration_error(y_true_idx: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true_idx).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def safe_auc(y_true_idx: np.ndarray, proba: np.ndarray) -> float:
    """Binary AUC, or macro one-vs-rest AUC over the classes present in y_true."""
    present = np.unique(y_true_idx)
    if len(present) < 2:
        return float("nan")
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_true_idx, proba[:, 1]))
    aucs = []
    for c in present:
        yb = (y_true_idx == c).astype(int)
        aucs.append(roc_auc_score(yb, proba[:, c]))
    return float(np.mean(aucs))


def score(y_true_idx: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true_idx, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true_idx, pred),
        "precision_macro": precision_score(y_true_idx, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_idx, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_idx, pred, average="macro", zero_division=0),
        "roc_auc": safe_auc(y_true_idx, proba),
        "ece": expected_calibration_error(y_true_idx, proba),
    }


def aggregate_by_sample(groups: np.ndarray, y_true_idx: np.ndarray, proba: np.ndarray):
    """Average ROI probabilities within each physical sample."""
    df = pd.DataFrame(proba)
    df["g"] = groups
    p = df.groupby("g", sort=False).mean()
    y = pd.Series(y_true_idx, index=groups).groupby(level=0, sort=False).first().loc[p.index]
    return y.to_numpy(), p.to_numpy()
