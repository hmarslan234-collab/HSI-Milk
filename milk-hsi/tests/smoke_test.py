"""End-to-end smoke test on small SYNTHETIC captures.

It writes fake Lumo/ENVI capture folders (dish on a napkin, species- and
adulterant-dependent spectra), then runs steps 01-03 and 05-08 on them with a
fast config. It only checks that the code runs end to end; the numbers it
prints mean nothing and must never be reported.

    python tests/smoke_test.py            # sklearn parts
    python tests/smoke_test.py --deep     # also step 04 (needs torch)
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
WL = np.linspace(397.66, 1003.81, 224)
HDR = """ENVI
samples = {s}
bands = 224
lines = {l}
interleave = bil
data type = 12
header offset = 0
byte order = 0
Wavelength = {{
{w}
}}
"""


def write_cube(path: Path, cube: np.ndarray):
    lines, samples, _ = cube.shape
    path.with_suffix(".hdr").write_text(HDR.format(s=samples, l=lines, w=",\n".join(f"{x:.2f}" for x in WL)))
    cube.transpose(0, 2, 1).astype("<u2").tofile(path.with_suffix(".raw"))


def milk_spectrum(species, adulterant, level, rng):
    base = {"cow": 0.95, "buffalo": 1.0, "goat": 0.92}[species]
    s = base - 0.25 * np.exp(-((WL - 975) / 25) ** 2) - 0.05 * (WL < 450)
    s += {"cow": 0.0, "buffalo": 0.03, "goat": -0.02}[species] * np.exp(-((WL - 900) / 40) ** 2)
    f = level / 10
    if adulterant == "water":
        s -= 0.15 * f * (WL > 600)
    elif adulterant != "pure":
        c = {"salicylic": 520, "boric": 650, "glucose": 735, "formalin": 820}[adulterant]
        s -= 0.06 * f * np.exp(-((WL - c) / 20) ** 2)
    s = s * (1.05 - 0.15 * (WL - 500) / 400)          # real milk: reflectance falls towards the NIR
    return s * rng.normal(1, 0.01)


def make_capture(folder: Path, species, adulterant, level, rng, lines=80, samples=64):
    (folder / "capture").mkdir(parents=True)
    white, dark = 3000.0, 150.0
    yy, xx = np.mgrid[:lines, :samples]
    napkin = ((yy - 40) / 36) ** 2 + ((xx - 32) / 30) ** 2 < 1
    dish = ((yy - 40) / 30) ** 2 + ((xx - 32) / 25) ** 2 < 1
    refl = np.full((lines, samples, 224), 0.02)
    refl[napkin] = 0.6 + 0.4 * (WL - 400) / 600       # napkin: rising NIR, ratio > 1
    refl[dish] = milk_spectrum(species, adulterant, level, rng)
    refl *= rng.normal(1, 0.01, refl.shape)
    raw = dark + refl * (white - dark)
    tag = f"{species}_{adulterant}_{level}"
    write_cube(folder / "capture" / f"Simple-{tag}", np.clip(raw, 0, 4095))
    write_cube(folder / "capture" / f"White_{tag}", np.full((10, samples, 224), white) + rng.normal(0, 5, (10, samples, 224)))
    write_cube(folder / "capture" / f"Dark_{tag}", np.full((10, samples, 224), dark) + rng.normal(0, 2, (10, samples, 224)))


def run(script, cfg, *args):
    cmd = [sys.executable, str(ROOT / "scripts" / script), "--config", str(cfg), *args]
    print("\n$", " ".join(cmd[1:]))
    subprocess.run(cmd, check=True, cwd=cfg.parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="milkhsi_smoke_"))
    rng = np.random.default_rng(0)
    raw = tmp / "data" / "raw"
    for sp in ("cow", "buffalo", "goat"):
        for i in range(4):
            make_capture(raw / f"pure-{sp}-milk-{i}", sp, "pure", 0, rng)
        for ad in ("water", "salicylic", "boric", "glucose", "formalin"):
            for lv in (1, 4, 7, 10):
                tag = f"{4 * lv}" if ad == "water" else f"{lv}"
                make_capture(raw / f"{sp}-{ad}-{tag}", sp, ad, lv, rng)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["paths"] = {k: str(tmp / v) for k, v in cfg["paths"].items()}
    cfg["roi"].update(n_per_capture=12, rim_margin_px=3)
    cfg["cv"].update(outer_folds=3, inner_folds=2, repeats=2)
    cfg["deep"].update(epochs=3, patience=2)
    cf = tmp / "config.yaml"
    cf.write_text(yaml.safe_dump(cfg))
    try:
        run("01_build_manifest.py", cf)
        run("02_extract_rois.py", cf)
        run("03_run_classical.py", cf, "--models", "knn", "dt", "rf", "xgb", "lgbm")
        if a.deep:
            run("04_run_deep.py", cf, "--repeats", "1")
        run("05_summarize.py", cf)
        run("06_interpret.py", cf, "--tasks", "species", "binary", "--n-perm", "1")
        run("07_figures.py", cf)
        run("08_robustness.py", cf)
        print("\nSMOKE TEST PASSED (synthetic data; numbers are meaningless)")
    finally:
        if a.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
