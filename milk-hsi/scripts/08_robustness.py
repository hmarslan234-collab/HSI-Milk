"""Robustness checks that the dataset can actually support.

1. Detection rate per adulterant and level: from the out-of-fold binary
   predictions of a model (default rf), the share of adulterated samples
   flagged as adulterated at every level. This shows how low a concentration
   is still detected (e.g. 4 % and 8 % water).
2. Unseen levels: the binary detector is trained on pure samples plus the odd
   levels (1,3,5,7,9) of every adulterant and tested on held-out pure samples
   plus the even levels (2,4,6,8,10), and the other way round. Pure samples are
   split by group so no dish is in both sets.

Outputs: results/robust_detection_by_level.csv, results/robust_unseen_levels.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from milkhsi.data import encode, load_config, load_rois  # noqa: E402
from milkhsi.metrics import aggregate_by_sample, score  # noqa: E402
from milkhsi.models_classic import build                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="rf")
    a = ap.parse_args()
    cfg = load_config(a.config)
    X, wl, meta, _ = load_rois(cfg)
    res = Path(cfg["paths"]["results"])
    groups = meta["sample_id"].to_numpy()
    y, classes = encode(meta["binary"])
    pos = classes.index("adulterated") if "adulterated" in classes else None
    if pos is None or len(classes) < 2:
        sys.exit("need both pure and adulterated samples")

    # 1. detection rate by level from OOF predictions
    f = res / "oof" / f"binary__{a.model}.npy"
    if f.exists():
        p = np.load(f)
        ok = ~np.isnan(p).any(1)
        ys, ps = aggregate_by_sample(groups[ok], y[ok], p[ok])
        smp = meta[ok].groupby("sample_id", sort=False).first()
        smp["pred_adulterated"] = ps.argmax(1) == pos
        tab = (smp.groupby(["adulterant", "level"])["pred_adulterated"].agg(["mean", "size"])
               .rename(columns={"mean": "flagged_rate", "size": "n_samples"}).reset_index())
        tab.to_csv(res / "robust_detection_by_level.csv", index=False)
        print(tab.to_string(index=False))
    else:
        print(f"{f} not found; run step 03 for the binary task first")

    # 2. train on odd levels, test on even levels (and the reverse)
    pipe, _ = build(a.model, cfg["preprocess"], seed=0)
    lv = meta["level"].to_numpy()
    pure = meta["adulterant"].to_numpy() == "pure"
    rows = []
    for name, train_lv, test_lv in [("train odd / test even", [1, 3, 5, 7, 9], [2, 4, 6, 8, 10]),
                                    ("train even / test odd", [2, 4, 6, 8, 10], [1, 3, 5, 7, 9])]:
        for rep in range(cfg["cv"]["repeats"]):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=rep)
            pi = np.where(pure)[0]
            ptr, pte = next(gss.split(pi, groups=groups[pi]))
            tr = np.r_[pi[ptr], np.where(np.isin(lv, train_lv) & ~pure)[0]]
            te = np.r_[pi[pte], np.where(np.isin(lv, test_lv) & ~pure)[0]]
            m = clone(pipe).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            ys, ps = aggregate_by_sample(groups[te], y[te], p)
            s = score(ys, ps)
            rows.append({"setting": name, "repeat": rep, **{f"sample_{k}": v for k, v in s.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(res / "robust_unseen_levels.csv", index=False)
    print(out.drop(columns="repeat").groupby("setting").agg(["mean", "std"]).round(4).T.to_string())


if __name__ == "__main__":
    main()
