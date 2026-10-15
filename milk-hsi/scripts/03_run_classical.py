"""Nested grouped CV for KNN, DT, RF, XGBoost and LightGBM on every task.

    python scripts/03_run_classical.py                     # all tasks, all models
    python scripts/03_run_classical.py --tasks species --models rf knn

Writes results/folds/<task>__<model>.csv (one row per outer fold),
results/oof/<task>__<model>.npy (out-of-fold probabilities, repeat 0) and
results/params/<task>__<model>.csv (selected hyper-parameters).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from milkhsi.data import encode, load_config, load_rois, make_splits   # noqa: E402
from milkhsi.evaluate import run_sklearn                               # noqa: E402
from milkhsi.labels import task_target                                 # noqa: E402
from milkhsi.models_classic import CLASSICAL, build                    # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--models", nargs="*", default=CLASSICAL)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    cfg = load_config(a.config)
    X, wl, meta, _ = load_rois(cfg)
    groups = meta["sample_id"].to_numpy()
    res = Path(cfg["paths"]["results"])
    for d in ("folds", "oof", "params", "classes"):
        (res / d).mkdir(parents=True, exist_ok=True)
    print(f"{X.shape[0]} ROIs, {X.shape[1]} bands ({wl[0]:.0f}-{wl[-1]:.0f} nm), "
          f"{meta['sample_id'].nunique()} physical samples")

    splits = make_splits(meta, cfg)
    for task in a.tasks or cfg["tasks"]:
        y, classes = encode(task_target(meta, task))
        if len(classes) < 2:
            print(f"skip {task}: only one class present ({classes[0]})")
            continue
        (res / "classes" / f"{task}.txt").write_text("\n".join(map(str, classes)))
        for name in a.models:
            built = build(name, cfg["preprocess"], seed=cfg["cv"]["seed"], n_jobs=a.n_jobs)
            if built is None:
                continue
            print(f"== {task} / {name}")
            folds, oof, best = run_sklearn(*built, X, y, groups, splits, len(classes), cfg)
            folds.to_csv(res / "folds" / f"{task}__{name}.csv", index=False)
            np.save(res / "oof" / f"{task}__{name}.npy", oof)
            best.to_csv(res / "params" / f"{task}__{name}.csv", index=False)
            print(f"   mean sample-level acc {folds['sample_accuracy'].mean():.4f}, "
                  f"ROI-level acc {folds['roi_accuracy'].mean():.4f}")


if __name__ == "__main__":
    main()
