"""Data figures for the paper (no model training).

    fig_rgb_pure.png          RGB composites (640/550/460 nm) of the pure samples (needs raw cubes)
    fig_mean_spectra.png      mean reflectance of pure cow / buffalo / goat dishes, +- SD across dishes
    fig_water_<species>.png   mean spectrum at every water level for one species
    fig_preprocessing.png     one ROI spectrum after each preprocessing step
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from milkhsi.data import load_config                    # noqa: E402
from milkhsi.io_envi import load_capture                # noqa: E402
from milkhsi.preprocess import BandMinMax, savgol, snv, trim_mask  # noqa: E402
from milkhsi.roi import band_index                      # noqa: E402

COL = {"cow": "#1f77b4", "buffalo": "#d62728", "goat": "#2ca02c"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-rgb", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    pr, res = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(cfg["paths"]["manifest"])
    cm = pd.read_csv(pr / "capture_means.csv").merge(man, on="capture")
    wcols = [c for c in cm.columns if c.replace(".", "", 1).isdigit()]
    wl = np.array([float(c) for c in wcols])
    keep = trim_mask(wl, cfg["preprocess"]["wl_min"], cfg["preprocess"]["wl_max"])

    # mean spectra of pure milk per species
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for sp, d in cm[cm.adulterant == "pure"].groupby("species"):
        S = d[wcols].to_numpy()[:, keep]
        mu = S.mean(0)
        ax.plot(wl[keep], mu, color=COL.get(sp), label=f"{sp} (n={len(d)})")
        if len(d) > 1:
            sd = S.std(0, ddof=1)
            ax.fill_between(wl[keep], mu - sd, mu + sd, color=COL.get(sp), alpha=0.2)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("relative reflectance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(res / "fig_mean_spectra.png", dpi=200)
    plt.close(fig)

    # water dilution series per species
    for sp in cm.species.unique():
        d = cm[(cm.species == sp) & cm.adulterant.isin(["pure", "water"])]
        if d.adulterant.nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.6))
        cmap = plt.get_cmap("viridis")
        for lv, g in d.groupby("level"):
            lab = "pure" if lv == 0 else f"water {4 * lv}%"
            ax.plot(wl[keep], g[wcols].to_numpy()[:, keep].mean(0), color=cmap(lv / 10), label=lab, lw=1)
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("relative reflectance")
        ax.set_title(f"{sp} milk, water dilution", fontsize=10)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(res / f"fig_water_{sp}.png", dpi=200)
        plt.close(fig)

    # preprocessing stages for one ROI spectrum
    z = np.load(pr / "rois.npz")
    s = z["spectra"][:200]
    w = z["wavelengths"]
    k = trim_mask(w, cfg["preprocess"]["wl_min"], cfg["preprocess"]["wl_max"])
    st1 = s[:, k]
    st2 = savgol(st1, cfg["preprocess"]["sg_window"], cfg["preprocess"]["sg_poly"])
    st3 = snv(st2)
    st4 = BandMinMax().fit_transform(st3)
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.8))
    for axx, (title, arr, ww) in zip(axs, [("calibrated reflectance", s, w), ("trimmed + SG", st2, w[k]),
                                           ("SNV", st3, w[k]), ("min-max", st4, w[k])]):
        axx.plot(ww, arr[:20].T, lw=0.6)
        axx.set_title(title, fontsize=9)
        axx.set_xlabel("nm", fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "fig_preprocessing.png", dpi=200)
    plt.close(fig)

    # RGB composites of one pure dish per species
    if not a.no_rgb:
        pure = man[man.adulterant == "pure"].groupby("species").head(1)
        fig, axs = plt.subplots(1, len(pure), figsize=(4 * len(pure), 3.6))
        axs = np.atleast_1d(axs)
        for axx, (_, r) in zip(axs, pure.iterrows()):
            cap = load_capture(Path(cfg["paths"]["raw_dir"]) / r["path"])
            rgb = np.stack([cap.reflectance[:, :, band_index(cap.wavelengths, x)] for x in (640, 550, 460)], -1)
            axx.imshow(np.clip(rgb / np.percentile(rgb, 99), 0, 1))
            axx.set_title(f"pure {r['species']}", fontsize=10)
            axx.axis("off")
        fig.tight_layout()
        fig.savefig(res / "fig_rgb_pure.png", dpi=200)
        plt.close(fig)
    print("figures written to", res)


if __name__ == "__main__":
    main()
