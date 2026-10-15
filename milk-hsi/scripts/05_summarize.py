"""Summary tables, significance tests, calibration and confusion matrices.

Reads every results/folds/<task>__<model>.csv written by steps 03 and 04.

Outputs in results/:
    summary_<level>.csv / .md   mean +- SD over all outer folds (level = sample or roi)
    stats_<task>.csv           paired tests of every model against the reference model
    fig_calibration_<task>.png, fig_confusion_<task>__<model>.png

Significance: with k-fold CV the fold scores are not independent, so we use
the Nadeau-Bengio corrected resampled t-test (variance inflated by
n_test/n_train) together with the Wilcoxon signed-rank test, over all
repeats x folds, and Holm-correct across models.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from milkhsi.data import encode, load_config                  # noqa: E402
from milkhsi.labels import add_task_columns, task_target      # noqa: E402
from milkhsi.metrics import aggregate_by_sample               # noqa: E402

ORDER = ["knn", "dt", "rf", "xgb", "lgbm", "hgb", "cnn1d", "cnn3d", "mtl1d", "mtl3d", "kd"]
NAMES = {"knn": "KNN", "dt": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost", "lgbm": "LightGBM",
         "hgb": "HistGB (sklearn)", "cnn1d": "1D-CNN + attention", "cnn3d": "3D-CNN", "mtl1d": "MTL 1D-CNN",
         "mtl3d": "MTL 3D-CNN", "kd": "KD student (1D)"}
METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "ece"]


def corrected_ttest(diff, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test."""
    d = np.asarray(diff, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.nan
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(2 * stats.t.sf(abs(t), df=k - 1))


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]).clip(max=1)
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--reference", default="rf")
    ap.add_argument("--metric", default="sample_f1_macro")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = Path(cfg["paths"]["results"])
    files = sorted((res / "folds").glob("*__*.csv"))
    if not files:
        sys.exit("no results/folds/*.csv yet; run steps 03/04 first")
    allf = []
    for f in files:
        task, model = f.stem.split("__")
        d = pd.read_csv(f)
        d["task"], d["model"] = task, model
        allf.append(d)
    allf = pd.concat(allf, ignore_index=True)
    tasks = [t for t in cfg["tasks"] if t in allf.task.unique()]
    models = [m for m in ORDER if m in allf.model.unique()]

    for lvl in ("sample", "roi"):
        rows = []
        for t in tasks:
            for m in models:
                d = allf[(allf.task == t) & (allf.model == m)]
                if d.empty:
                    continue
                r = {"task": t, "model": NAMES[m], "folds": len(d)}
                for k in METRICS:
                    col = f"{lvl}_{k}"
                    if col in d and d[col].notna().any():
                        r[k] = f"{d[col].mean():.4f} +- {d[col].std(ddof=1):.4f}"
                for k in ("fit_s", "infer_ms_per_roi", "params"):
                    if k in d and d[k].notna().any():
                        r[k] = round(float(d[k].mean()), 4)
                rows.append(r)
        tab = pd.DataFrame(rows)
        tab.to_csv(res / f"summary_{lvl}.csv", index=False)
        try:
            md = tab.to_markdown(index=False)
        except ImportError:          # 'tabulate' not installed
            md = tab.to_string(index=False)
        (res / f"summary_{lvl}.md").write_text(md)
        print(f"\n### {lvl}-level results\n{tab.to_string(index=False)}")

    # ---- significance against the reference model
    splits = json.loads((res / "splits.json").read_text())
    n_tr = np.mean([len(s["train"]) for s in splits])
    n_te = np.mean([len(s["test"]) for s in splits])
    for t in tasks:
        ref = allf[(allf.task == t) & (allf.model == a.reference)].set_index(["repeat", "fold"])[a.metric]
        if ref.empty:
            continue
        out = []
        for m in models:
            if m == a.reference:
                continue
            cur = allf[(allf.task == t) & (allf.model == m)].set_index(["repeat", "fold"])[a.metric]
            j = ref.to_frame("ref").join(cur.to_frame("m"), how="inner").dropna()
            if len(j) < 3:
                continue
            diff = j["m"] - j["ref"]
            try:
                pw = stats.wilcoxon(diff, zero_method="zsplit").pvalue if np.any(diff != 0) else 1.0
            except ValueError:
                pw = np.nan
            out.append({"model": NAMES[m], "pairs": len(j), "mean_diff": diff.mean(),
                        "p_corrected_t": corrected_ttest(diff, n_tr, n_te), "p_wilcoxon": pw})
        if out:
            s = pd.DataFrame(out)
            s["p_corrected_t_holm"] = holm(s["p_corrected_t"])
            s["p_wilcoxon_holm"] = holm(s["p_wilcoxon"])
            s.to_csv(res / f"stats_{t}.csv", index=False)
            print(f"\n### {t}: {a.metric} vs {NAMES[a.reference]}\n{s.to_string(index=False)}")

    # ---- calibration + confusion matrices from out-of-fold predictions (repeat 0)
    meta = add_task_columns(pd.read_csv(Path(cfg["paths"]["processed"]) / "rois_meta.csv"))
    groups = meta["sample_id"].to_numpy()
    for t in tasks:
        y, classes = encode(task_target(meta, t))
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        for m in models:
            f = res / "oof" / f"{t}__{m}.npy"
            if not f.exists():
                continue
            p = np.load(f)
            ok = ~np.isnan(p).any(1)
            if ok.sum() == 0:
                continue
            conf = p[ok].max(1)
            corr = (p[ok].argmax(1) == y[ok]).astype(float)
            bins = np.linspace(0, 1, 11)
            b = np.clip(np.digitize(conf, bins) - 1, 0, 9)
            xs = [conf[b == i].mean() for i in range(10) if (b == i).sum() >= 10]
            ys = [corr[b == i].mean() for i in range(10) if (b == i).sum() >= 10]
            ax.plot(xs, ys, "o-", ms=3, lw=1, label=NAMES[m])
            # confusion matrix at sample level
            if len(classes) <= 12:
                ys_, ps_ = aggregate_by_sample(groups[ok], y[ok], p[ok])
                cm = pd.crosstab(pd.Series(ys_, name="true"), pd.Series(ps_.argmax(1), name="pred"))
                cm = cm.reindex(index=range(len(classes)), columns=range(len(classes)), fill_value=0)
                fg, axc = plt.subplots(figsize=(1.0 + 0.55 * len(classes), 0.8 + 0.5 * len(classes)))
                axc.imshow(cm.values, cmap="Blues")
                for (i, j), v in np.ndenumerate(cm.values):
                    axc.text(j, i, int(v), ha="center", va="center", fontsize=8,
                             color="white" if v > cm.values.max() / 2 else "black")
                axc.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
                axc.set_yticks(range(len(classes)), classes, fontsize=8)
                axc.set_xlabel("predicted")
                axc.set_ylabel("true")
                axc.set_title(f"{t}: {NAMES[m]} (samples)", fontsize=9)
                fg.tight_layout()
                fg.savefig(res / f"fig_confusion_{t}__{m}.png", dpi=200)
                plt.close(fg)
        ax.set_xlabel("mean predicted confidence")
        ax.set_ylabel("observed accuracy")
        ax.set_title(f"Reliability ({t}, ROI level)", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(res / f"fig_calibration_{t}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
