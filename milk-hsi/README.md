# Milk HSI: species authentication and adulteration detection

Code for the paper *Lightweight-to-Advanced Spectral-Spatial Learning for Animal Milk Species
Authentication and Adulteration Detection Using Hyperspectral Imaging*.

It reads raw Specim FX10 captures (Lumo Scanner export), converts them to reflectance, extracts
ROIs from the milk surface, and evaluates KNN, Decision Tree, Random Forest, XGBoost, LightGBM,
a 1D-CNN with self-attention, a 3D-CNN, multi-task versions of both CNNs and a distilled student
network, all on the same grouped cross-validation folds.

## Folder layout

```
milk-hsi/
  config.yaml              all settings (paths, ROI size, preprocessing, CV, deep-learning)
  requirements.txt
  milkhsi/                 library code
    io_envi.py             ENVI reader, white/dark calibration (Eq. 1)
    roi.py                 milk segmentation in the dish, ROI sampling
    preprocess.py          band trimming, Savitzky-Golay, SNV / MSC, min-max
    labels.py              label columns and task definitions
    data.py                dataset loading, grouped outer folds shared by all models
    models_classic.py      KNN, DT, RF, XGBoost, LightGBM pipelines and grids
    models_deep.py         1D-CNN + attention, 3D-CNN, student net, multi-task heads, losses
    train_deep.py          training loop, early stopping, batching of spectra / patches
    evaluate.py            nested grouped CV for scikit-learn models
    metrics.py             accuracy, macro P/R/F1, ROC-AUC, ECE, per-sample aggregation
  scripts/
    01_build_manifest.py   list captures and guess labels -> data/manifest.csv
    02_extract_rois.py     calibration + ROI extraction -> data/processed/
    03_run_classical.py    KNN / DT / RF / XGBoost / LightGBM
    04_run_deep.py         1D-CNN, 3D-CNN, multi-task, knowledge distillation (PyTorch)
    05_summarize.py        tables, significance tests, calibration, confusion matrices
    06_interpret.py        wavelength importance (window permutation, SHAP)
    07_figures.py          spectra and RGB figures
    08_robustness.py       detection rate per level, unseen-level test
  tests/smoke_test.py      end-to-end run on small synthetic captures (checks the code only)
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

For a GPU, install PyTorch from https://pytorch.org with the CUDA option for your card first.

## Data

Put every capture folder, unchanged, under `data/raw/`:

```
data/raw/pure-cow-milk/capture/Simple-pure_cow_milk.hdr / .raw
data/raw/pure-cow-milk/capture/White_pure_cow_milk.hdr / .raw
data/raw/pure-cow-milk/capture/Dark_pure_cow_milk.hdr  / .raw
data/raw/cow-water-4/...
```

Each capture is one dish (one physical sample) with its own white and dark reference.

## Running

```
python scripts/01_build_manifest.py      # then open data/manifest.csv and check every label
python scripts/02_extract_rois.py        # look at results/qc/*.png: cyan = milk mask, red = ROIs
python scripts/03_run_classical.py
python scripts/04_run_deep.py            # GPU recommended
python scripts/05_summarize.py
python scripts/06_interpret.py
python scripts/07_figures.py
python scripts/08_robustness.py
```

Quick check of the deep code before a long run:

```
python scripts/04_run_deep.py --models cnn1d --tasks species --repeats 1
```

### The manifest

`data/manifest.csv` has one row per capture: `capture, species, adulterant, level, sample_id, path`.

* `adulterant` is one of `pure, water, salicylic, boric, glucose, formalin`.
* `level` is 0 for pure and 1-10 for the ten concentrations (water 4 %, 8 %, ..., 40 % map to 1-10;
  the chemicals 1-10 mL per 100 mL map to 1-10).
* `sample_id` identifies the physical sample. If one dish was scanned twice, give both scans the
  same `sample_id`; the CV keeps them together.

Labels are guessed from folder names, so check them by hand.

## How evaluation avoids leakage

* All ROIs of one dish share a `sample_id`, and outer folds are built on samples
  (`StratifiedGroupKFold`), never on ROIs.
* One set of folds (`results/splits.json`) is used by every model and task, so paired tests compare
  like with like.
* Hyper-parameters are chosen with an inner grouped CV on the training part only.
* Fitted preprocessing (MSC reference, min-max ranges, PCA) is inside each model pipeline and is
  re-fitted per fold. Trimming, SG and SNV act on one spectrum at a time and cannot leak.
* Scores are reported at ROI level and at sample level (ROI probabilities averaged per dish).
  The sample-level numbers are the honest ones, because ROIs of one dish are not independent.
* The binary task is very unbalanced (30 pure vs 1,500 adulterated dishes in the full design).
  Always read balanced accuracy and macro-F1 there, not plain accuracy.

## Outputs

All in `results/`: `folds/<task>__<model>.csv` (one row per outer fold), `oof/` (out-of-fold
probabilities), `params/` (selected hyper-parameters), `summary_sample.csv`, `summary_roi.csv`,
`stats_<task>.csv`, `importance_<task>.csv`, `robust_*.csv` and the figures.

## Settings that need checking on new data

* `roi.ratio_max`, `roi.min_refl`: thresholds of the milk mask. Inspect `results/qc/` after step 02.
* `roi.rim_margin_px`: how far ROIs stay from the glass rim.
* `preprocess.wl_min / wl_max`: the 420-950 nm window. The strong water band near 970 nm lies
  outside it; widen `wl_max` to about 990 nm if you want the models to use it.
