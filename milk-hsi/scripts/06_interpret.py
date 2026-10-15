"""Which wavelengths drive the predictions?

For each outer fold of repeat 0, a Random Forest (or LightGBM with
--model lgbm) is trained on the training samples and two importance measures
are computed on the *held-out* samples:

1. window permutation importance: the bands inside a ~13 nm window are
   shuffled together across test ROIs and the drop in macro-F1 is recorded
   (windows avoid the dilution that single-band permutation suffers when
   neighbouring bands are almost collinear);
2. mean |SHAP| per band (TreeExplainer), only if the `shap` package is installed.

Outputs: results/importance_<task>.csv and results/fig_importance_<task>.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from milkhsi.data import encode, load_config, load_rois, make_splits  # noqa: E402
from milkhsi.labels import task_target                               # noqa: E402
from milkhsi.models_classic import build                             # noqa: E402

try:
    import shap
except ImportError:
    shap = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="rf")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--window", type=int, default=5, help="bands per permutation window")
    ap.add_argument("--n-perm", type=int, default=5)
    a = ap.parse_args()
    cfg = load_config(a.config)
    X, wl, meta, _ = load_rois(cfg)
    res = Path(cfg["paths"]["results"])
    splits = [s for s in make_splits(meta, cfg) if s["repeat"] == 0]
    rng = np.random.default_rng(0)
    means = X.mean(0)

    for task in a.tasks or cfg["tasks"]:
        y, classes = encode(task_target(meta, task))
        if len(classes) < 2:
            continue
        pipe, _ = build(a.model, cfg["preprocess"], seed=0)
        params = res / "params" / f"{task}__{a.model}.csv"
        win_imp, shap_imp = [], []
        starts = list(range(0, X.shape[1], a.window))
        for i, sp in enumerate(splits):
            tr, te = np.asarray(sp["train"]), np.asarray(sp["test"])
            m = clone(pipe)
            if params.exists():
                best = pd.read_csv(params)
                best = best[(best.repeat == 0) & (best.fold == sp["fold"])].drop(columns=["repeat", "fold"])
                if len(best):
                    m.set_params(**{k: (None if pd.isna(v) else type_fix(v)) for k, v in best.iloc[0].items()})
            local = np.unique(y[tr])
            m.fit(X[tr], np.searchsorted(local, y[tr]))
            yt = np.searchsorted(local, y[te])
            base = f1_score(yt, m.predict(X[te]), average="macro")
            imp = np.zeros(len(starts))
            for j, s0 in enumerate(starts):
                drops = []
                for _ in range(a.n_perm):
                    Xp = X[te].copy()
                    perm = rng.permutation(len(te))
                    Xp[:, s0:s0 + a.window] = Xp[perm, s0:s0 + a.window]
                    drops.append(base - f1_score(yt, m.predict(Xp), average="macro"))
                imp[j] = np.mean(drops)
            win_imp.append(imp)
            if shap is not None and a.model in ("rf", "dt", "lgbm", "xgb"):
                Xt = m[:-1].transform(X[te]) if len(m) > 1 else X[te]
                sv = shap.TreeExplainer(m[-1]).shap_values(Xt[:500])
                sv = np.abs(np.asarray(sv))
                # reduce over classes and samples -> one value per band
                band_axis = [k for k, n in enumerate(sv.shape) if n == X.shape[1]][0]
                shap_imp.append(sv.mean(axis=tuple(k for k in range(sv.ndim) if k != band_axis)))
            print(f"{task} fold {i}: base macro-F1 {base:.3f}")
        win_imp = np.mean(win_imp, 0)
        centers = np.array([wl[s0:s0 + a.window].mean() for s0 in starts])
        out = pd.DataFrame({"window_center_nm": centers, "window_from_nm": [wl[s0] for s0 in starts],
                            "window_to_nm": [wl[min(s0 + a.window, len(wl)) - 1] for s0 in starts],
                            "perm_importance_f1_drop": win_imp})
        if shap_imp:
            sv = np.mean(shap_imp, 0)
            out["mean_abs_shap"] = [sv[s0:s0 + a.window].mean() for s0 in starts]
        out.to_csv(res / f"importance_{task}.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.bar(centers, win_imp, width=(wl[1] - wl[0]) * a.window * 0.9, color="#3b6ea5")
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("drop in macro-F1")
        ax2 = ax.twinx()
        ax2.plot(wl, means, color="0.4", lw=1)
        ax2.set_ylabel("mean preprocessed spectrum", color="0.4")
        ax.set_title(f"Window permutation importance ({task}, {a.model.upper()}, held-out samples)", fontsize=9)
        fig.tight_layout()
        fig.savefig(res / f"fig_importance_{task}.png", dpi=200)
        plt.close(fig)
        top = out.sort_values("perm_importance_f1_drop", ascending=False).head(5)
        print(top.to_string(index=False))


def type_fix(v):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


if __name__ == "__main__":
    main()
