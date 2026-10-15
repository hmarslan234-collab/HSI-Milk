"""Calibrate every capture, segment the milk and extract ROIs.

Outputs (paths.processed):
    rois.npz            spectra (N_roi x 224, mean reflectance per ROI), wavelengths
    rois_meta.csv       one row per ROI: capture, sample_id, labels, ROI position
    patches.npy         N_roi x 7 x 7 x 224 float16 (only if roi.save_patches)
    capture_means.csv   mean reflectance spectrum of each dish (for figures)
and a QC image per capture in results/qc/ showing the mask and ROIs.

This is the only step that reads the raw cubes. It is I/O heavy, so run it on
the machine that holds the data; the files it writes are small enough to move.
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
from milkhsi.data import load_config        # noqa: E402
from milkhsi.io_envi import load_capture    # noqa: E402
from milkhsi.roi import band_index, extract  # noqa: E402


def qc_figure(refl, wl, mask, corners, size, path, title):
    rgb = np.stack([refl[:, :, band_index(wl, w)] for w in (640, 550, 460)], -1)
    rgb = np.clip(rgb / np.percentile(rgb, 99), 0, 1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(rgb)
    ax.contour(mask, levels=[0.5], colors="cyan", linewidths=0.8)
    for y, x in corners:
        ax.add_patch(plt.Rectangle((x, y), size, size, fill=False, ec="red", lw=0.6))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rc = cfg["roi"]
    man = pd.read_csv(cfg["paths"]["manifest"])
    raw = Path(cfg["paths"]["raw_dir"])
    out = Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    qc = Path(cfg["paths"]["results"]) / "qc"
    qc.mkdir(parents=True, exist_ok=True)

    n_max = len(man) * rc["n_per_capture"]
    patches = None
    specs, metas, means = [], [], []
    wl = None
    k = 0
    for i, r in man.iterrows():
        cap = load_capture(raw / r["path"])
        wl = cap.wavelengths
        s, p, corners, mask = extract(cap.reflectance, wl, rc["size"], rc["n_per_capture"],
                                      rc["rim_margin_px"], seed=rc["seed"] + i, keep_patches=rc["save_patches"],
                                      ratio_max=rc.get("ratio_max", 0.97), min_refl=rc.get("min_refl", 0.4))
        if len(s) < rc["n_per_capture"]:
            print(f"  warning: {r['capture']} gave only {len(s)} ROIs")
        if rc["save_patches"]:
            if patches is None:
                patches = np.lib.format.open_memmap(out / "patches_tmp.npy", mode="w+", dtype=np.float16,
                                                    shape=(n_max, rc["size"], rc["size"], len(wl)))
            patches[k:k + len(p)] = p
        k += len(s)
        specs.append(s)
        dish = cap.reflectance[mask]
        means.append({"capture": r["capture"], **{f"{w:.2f}": v for w, v in zip(wl, dish.mean(0))}})
        for (y, x) in corners:
            metas.append({"capture": r["capture"], "sample_id": r["sample_id"], "species": r["species"],
                          "adulterant": r["adulterant"], "level": int(r["level"]), "roi_y": int(y), "roi_x": int(x)})
        qc_figure(cap.reflectance, wl, mask, corners, rc["size"], qc / f"{r['capture']}.png",
                  f"{r['capture']}  ({len(s)} ROIs, {mask.sum()} milk px)")
        print(f"[{i + 1}/{len(man)}] {r['capture']}: {len(s)} ROIs")

    spectra = np.concatenate(specs)
    np.savez_compressed(out / "rois.npz", spectra=spectra, wavelengths=wl)
    pd.DataFrame(metas).to_csv(out / "rois_meta.csv", index=False)
    pd.DataFrame(means).to_csv(out / "capture_means.csv", index=False)
    if patches is not None:
        final = np.lib.format.open_memmap(out / "patches.npy", mode="w+", dtype=np.float16, shape=(k,) + patches.shape[1:])
        for s0 in range(0, k, 2000):
            final[s0:min(k, s0 + 2000)] = patches[s0:min(k, s0 + 2000)]
        final.flush()
        del patches, final
        (out / "patches_tmp.npy").unlink()
    print(f"done: {spectra.shape[0]} ROIs from {len(man)} captures -> {out}")


if __name__ == "__main__":
    main()
