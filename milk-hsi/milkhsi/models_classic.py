"""Lightweight (KNN, DT, RF) and gradient-boosting (XGBoost, LightGBM) models.

Each model is a scikit-learn Pipeline whose fitted steps (scatter correction,
min-max, optional PCA) are re-fitted inside every training fold. Grids follow
Table 7 of the paper and are searched with an inner grouped CV.
"""
from __future__ import annotations

import warnings

from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocess import BandMinMax, scatter_step

try:
    from xgboost import XGBClassifier
except ImportError:          # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:          # pragma: no cover
    LGBMClassifier = None


def _front(scatter: str, minmax: bool, pca: bool):
    steps = []
    s = scatter_step(scatter)
    if s is not None:
        steps.append(("scatter", s))
    if minmax:
        steps.append(("minmax", BandMinMax()))
    if pca:
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full")))
    return steps


def build(name: str, pp: dict, seed: int = 0, n_jobs: int = -1):
    """Return (pipeline, param_grid) or None if the library is missing."""
    sc, mm = pp["scatter"], pp["minmax"]
    if name == "knn":
        pipe = Pipeline(_front(sc, mm, pca=True) + [("clf", KNeighborsClassifier(metric="euclidean"))])
        grid = {"clf__n_neighbors": [5, 7, 9]}
    elif name == "dt":
        pipe = Pipeline(_front(sc, mm, pca=True) + [("clf", DecisionTreeClassifier(criterion="gini", random_state=seed,
                                                                                   class_weight="balanced"))])
        grid = {"clf__max_depth": [5, 10]}
    elif name == "rf":
        pipe = Pipeline(_front(sc, mm, pca=False) + [("clf", RandomForestClassifier(random_state=seed, n_jobs=n_jobs,
                                                                                    class_weight="balanced"))])
        grid = {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 20]}
    elif name == "xgb":
        if XGBClassifier is None:
            warnings.warn("xgboost not installed; skipping XGBoost")
            return None
        pipe = Pipeline(_front(sc, mm, pca=False) + [("clf", XGBClassifier(
            learning_rate=0.05, max_depth=6, n_estimators=400, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=seed, n_jobs=n_jobs, eval_metric="mlogloss"))])
        grid = {"clf__max_depth": [4, 6]}
    elif name == "lgbm":
        if LGBMClassifier is None:
            warnings.warn("lightgbm not installed; skipping LightGBM")
            return None
        pipe = Pipeline(_front(sc, mm, pca=False) + [("clf", LGBMClassifier(
            learning_rate=0.05, num_leaves=31, n_estimators=400, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, n_jobs=n_jobs, class_weight="balanced", verbose=-1))])
        grid = {"clf__num_leaves": [15, 31]}
    elif name == "hgb":
        # scikit-learn's histogram gradient boosting; used only when neither
        # xgboost nor lightgbm can be installed. Report it under its own name.
        pipe = Pipeline(_front(sc, mm, pca=False) + [("clf", HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=31, max_iter=300, random_state=seed, class_weight="balanced"))])
        grid = {"clf__max_leaf_nodes": [15, 31]}
    else:
        raise ValueError(name)
    return pipe, grid


CLASSICAL = ["knn", "dt", "rf", "xgb", "lgbm"]
