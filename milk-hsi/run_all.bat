@echo off
REM Runs the whole pipeline on Windows. Stop after step 02 the first time and check results\qc\ and data\manifest.csv.
python scripts\01_build_manifest.py || exit /b 1
python scripts\02_extract_rois.py   || exit /b 1
python scripts\03_run_classical.py  || exit /b 1
python scripts\04_run_deep.py       || exit /b 1
python scripts\05_summarize.py      || exit /b 1
python scripts\06_interpret.py      || exit /b 1
python scripts\07_figures.py        || exit /b 1
python scripts\08_robustness.py     || exit /b 1
echo Done. Zip the "results" folder and data\processed\rois_meta.csv and send them back.
