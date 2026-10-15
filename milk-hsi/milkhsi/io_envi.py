"""Reading Specim FX10 (Lumo Scanner) captures and radiometric calibration.

A capture folder looks like this (as exported by Lumo):

    pure-cow-milk/
        capture/
            Simple-pure_cow_milk.hdr / .raw    <- sample scan
            White_pure_cow_milk.hdr  / .raw    <- white reference (Spectralon)
            Dark_pure_cow_milk.hdr   / .raw    <- dark reference (shutter closed)
        metadata/..., manifest.xml, properties.xml, preview .png

File names are not always consistent ("Simple-" vs "Simple_"), so the three
files are found by their prefix only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ENVI_DTYPES = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                5: np.float64, 12: np.uint16, 13: np.uint32}


def read_header(hdr_path: str | Path) -> dict:
    """Parse an ENVI .hdr file into a dict (numbers converted, lists split)."""
    text = Path(hdr_path).read_text(errors="ignore")
    out: dict = {}
    # join multi-line { ... } blocks onto one line
    for m in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{.*?\}|[^\n]*)", text, flags=re.M | re.S):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if val.startswith("{"):
            items = [v.strip() for v in val.strip("{}").replace("\n", " ").split(",") if v.strip()]
            try:
                out[key] = [float(v) for v in items]
            except ValueError:
                out[key] = items
        else:
            try:
                out[key] = int(val)
            except ValueError:
                try:
                    out[key] = float(val)
                except ValueError:
                    out[key] = val
    return out


def read_cube(hdr_path: str | Path) -> tuple[np.ndarray, dict]:
    """Return (cube[lines, samples, bands] as float32, header)."""
    hdr_path = Path(hdr_path)
    h = read_header(hdr_path)
    raw_path = hdr_path.with_suffix(".raw")
    if not raw_path.exists():
        alt = [p for p in hdr_path.parent.glob(hdr_path.stem + ".*") if p.suffix.lower() in (".raw", ".dat", ".img", "")]
        alt = [p for p in alt if p != hdr_path]
        if not alt:
            raise FileNotFoundError(f"no binary file next to {hdr_path}")
        raw_path = alt[0]
    dtype = np.dtype(_ENVI_DTYPES[int(h.get("data type", 12))])
    dtype = dtype.newbyteorder(">" if int(h.get("byte order", 0)) == 1 else "<")
    s, b = int(h["samples"]), int(h["bands"])
    data = np.fromfile(raw_path, dtype=dtype, offset=int(h.get("header offset", 0)))
    lines = data.size // (s * b)          # trust the file size over the header
    data = data[: lines * s * b]
    il = str(h.get("interleave", "bil")).lower()
    if il == "bil":
        cube = data.reshape(lines, b, s).transpose(0, 2, 1)
    elif il == "bip":
        cube = data.reshape(lines, s, b)
    elif il == "bsq":
        cube = data.reshape(b, lines, s).transpose(1, 2, 0)
    else:
        raise ValueError(f"unknown interleave {il}")
    return cube.astype(np.float32), h


@dataclass
class Capture:
    name: str
    reflectance: np.ndarray       # lines x samples x bands, relative reflectance
    wavelengths: np.ndarray       # bands (nm)
    header: dict


def _find(folder: Path, prefix: str) -> Path:
    hits = sorted(p for p in folder.rglob("*.hdr") if p.name.lower().startswith(prefix))
    if not hits:
        raise FileNotFoundError(f"no '{prefix}*.hdr' under {folder}")
    return hits[0]


def load_capture(folder: str | Path, eps: float = 1e-6) -> Capture:
    """Load one capture folder and convert it to reflectance with Eq. (1):

        R = (I_sample - I_dark) / (I_white - I_dark)

    The white and dark scans are averaged over their lines, giving one
    reference value per (pixel column, band). This is the usual flat-field
    correction for a push-broom camera and also removes the column stripes
    that are visible in the uncalibrated preview.
    """
    folder = Path(folder)
    sample_hdr = next((p for p in sorted(folder.rglob("*.hdr"))
                       if not p.name.lower().startswith(("white", "dark", "whiteref", "darkref"))), None)
    if sample_hdr is None:
        raise FileNotFoundError(f"no sample .hdr in {folder}")
    sample, h = read_cube(sample_hdr)
    white, _ = read_cube(_find(folder, "white"))
    dark, _ = read_cube(_find(folder, "dark"))
    w = white.mean(axis=0)
    d = dark.mean(axis=0)
    refl = (sample - d) / np.maximum(w - d, eps)
    wl = np.asarray(h.get("wavelength"), dtype=np.float32)
    return Capture(folder.name, refl.astype(np.float32), wl, h)
