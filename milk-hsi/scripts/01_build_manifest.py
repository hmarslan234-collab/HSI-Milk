"""Find every capture folder under paths.raw_dir and write data/manifest.csv.

A capture folder is any folder that contains a sample .hdr plus White*/Dark*
references (the Lumo export layout). Labels are guessed from the folder
name; open the CSV and correct anything wrong before step 02.

    python scripts/01_build_manifest.py            # keeps an existing manifest
    python scripts/01_build_manifest.py --force    # rebuilds it
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from milkhsi.data import load_config          # noqa: E402
from milkhsi.labels import guess_from_name    # noqa: E402


def capture_folders(root: Path):
    import os
    seen = set()
    hdrs = [Path(d) / f for d, _, fs in os.walk(root, followlinks=True) for f in fs if f.lower().endswith(".hdr")]
    for hdr in hdrs:
        n = hdr.name.lower()
        if n.startswith(("white", "dark")):
            continue
        folder = hdr.parent.parent if hdr.parent.name.lower() == "capture" else hdr.parent
        if folder not in seen:
            seen.add(folder)
            yield folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    root, out = Path(cfg["paths"]["raw_dir"]), Path(cfg["paths"]["manifest"])
    if out.exists() and not a.force:
        print(f"{out} exists; use --force to rebuild")
        return
    rows = []
    for f in sorted(capture_folders(root)):
        r = guess_from_name(f.name)
        r["path"] = str(f.relative_to(root))
        rows.append(r)
    if not rows:
        sys.exit(f"no captures found under {root}")
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} captures written to {out}")
    bad = df[(df.species == "") | (df.adulterant == "")]
    if len(bad):
        print("Could not guess labels for these, fill them in by hand:")
        print(bad[["capture"]].to_string(index=False))
    print(df.groupby(["species", "adulterant"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
