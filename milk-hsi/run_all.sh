#!/usr/bin/env bash
set -e
for s in 01_build_manifest 02_extract_rois 03_run_classical 04_run_deep 05_summarize 06_interpret 07_figures 08_robustness; do
  python scripts/$s.py
done
