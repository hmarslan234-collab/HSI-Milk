"""Dish segmentation and ROI extraction.

Each capture holds one petri dish on a white napkin over a dark background.
Milk is separated from the napkin with the 900/550 nm reflectance ratio
(milk scatters strongly and absorbs more in the NIR, so the ratio is lower
than for paper), the largest connected region is kept, holes are filled and
the edge is eroded so that the glass rim, meniscus and glare are excluded.

ROIs are then square windows (default 7 x 7 = 49 pixels) placed at random,
without overlap, fully inside the eroded mask. For every ROI we keep the mean
spectrum (classical models and 1D-CNN) and, optionally, the whole 7 x 7 x B
patch (3D-CNN).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def band_index(wl: np.ndarray, nm: float) -> int:
    return int(np.argmin(np.abs(wl - nm)))


def milk_mask(refl: np.ndarray, wl: np.ndarray, rim_margin_px: int = 15,
              ratio_max: float = 0.97, min_refl: float = 0.4) -> np.ndarray:
    r550 = refl[:, :, band_index(wl, 550)]
    r900 = refl[:, :, band_index(wl, 900)]
    ratio = r900 / np.maximum(r550, 1e-3)
    bright = refl[:, :, band_index(wl, 420):band_index(wl, 900)].mean(axis=2)
    m = (ratio < ratio_max) & (bright > min_refl) & (bright < 1.6)
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        raise RuntimeError("milk region not found; check thresholds in roi.milk_mask")
    sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
    m = lab == (1 + int(np.argmax(sizes)))
    m = ndi.binary_fill_holes(m)
    if rim_margin_px > 0:
        m = ndi.binary_erosion(m, iterations=rim_margin_px)
    # drop saturated / specular pixels (reflectance far above the white panel)
    m &= refl.max(axis=2) < 1.5
    return m


def sample_rois(mask: np.ndarray, size: int, n: int, rng: np.random.Generator):
    """Top-left corners of up to n non-overlapping size x size windows inside mask."""
    # a window is valid if every pixel is in the mask: the local mean of the
    # mask over the window (centred) must be 1
    frac = ndi.uniform_filter(mask.astype(np.float32), size=size, mode="constant")
    cy, cx = np.nonzero(frac > 1 - 1e-4)
    ys, xs = cy - size // 2, cx - size // 2
    keep = (ys >= 0) & (xs >= 0)
    ys, xs = ys[keep], xs[keep]
    order = rng.permutation(len(ys))
    taken = np.zeros_like(mask, dtype=bool)
    corners = []
    for i in order:
        y, x = ys[i], xs[i]
        if y + size > mask.shape[0] or x + size > mask.shape[1]:
            continue
        if taken[y:y + size, x:x + size].any():
            continue
        taken[y:y + size, x:x + size] = True
        corners.append((y, x))
        if len(corners) == n:
            break
    return corners


def extract(refl: np.ndarray, wl: np.ndarray, size: int = 7, n: int = 50,
            rim_margin_px: int = 15, seed: int = 0, keep_patches: bool = True,
            ratio_max: float = 0.97, min_refl: float = 0.4):
    mask = milk_mask(refl, wl, rim_margin_px, ratio_max=ratio_max, min_refl=min_refl)
    rng = np.random.default_rng(seed)
    corners = sample_rois(mask, size, n, rng)
    spectra = np.stack([refl[y:y + size, x:x + size].reshape(-1, refl.shape[2]).mean(0)
                        for y, x in corners]).astype(np.float32)
    patches = None
    if keep_patches:
        patches = np.stack([refl[y:y + size, x:x + size] for y, x in corners]).astype(np.float16)
    return spectra, patches, np.array(corners), mask
