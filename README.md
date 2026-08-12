# Freight-Load-Rate-Prediction

End-to-end regression pipeline that predicts freight `posted_rate` (USD) for
truckload shipments, built entirely with **pandas / numpy / matplotlib**
(no scikit-learn — see [Environment Constraint](#environment-constraint)).

Final deliverable: [`validation_predictions.csv`](validation_predictions.csv)
— predicted rates for all 12,000 loads in `validation.csv`.

## Overview

| | |
|---|---|
| **Task** | Regression — predict `posted_rate` per load |
| **Training data** | 48,000 labeled loads (`train-test.csv`), Jan–Oct 2025 |
| **Prediction target** | 12,000 unlabeled loads (`validation.csv`), Nov–Dec 2025 |
| **Best model** | Bagged regression trees (25-tree ensemble, from scratch) |
| **Holdout performance** | MAE \$135 · RMSE \$610 · R² 0.838 · MAPE 6.6% |

## Environment Constraint

The task's `requirement.txt` specifies only:

```
matplotlib>=3.8,<4
numpy>=1.26,<3
pandas>=2.0,<3
```

No scikit-learn. Rather than defaulting to sklearn, this project treats
`requirement.txt` as the authoritative environment spec and implements every
model — Ridge regression, a CART decision tree, and a bagged-tree ensemble —
from scratch in numpy. See [`features_models.py`](features_models.py).

## Project Structure

```
.
├── train-test.csv                       # labeled development data (input)
├── validation.csv                       # unlabeled loads to score (input)
├── validation-predictions-template.csv  # output schema (input)
├── december-chart-inputs.csv            # bonus scenario file (input)
│
├── features_models.py                   # feature engineering + Ridge/Tree/BaggedTrees (from scratch)
├── 01_eda_prep.py                       # exploratory data analysis
├── 02_train_eval.py                     # time-based holdout, model comparison, error analysis
├── 03_final_predict.py                  # refit on full data, generate final predictions
│
├── validation_predictions.csv           # ★ required deliverable
├── december_chart_predictions.csv       # bonus deliverable
├── model_comparison.csv                 # holdout metrics per model
├── linear_coefficients.csv              # standardized linear-regression coefficients
├── permutation_importance.csv           # feature importance (bagged trees)
│
├── eda_overview.png                     # EDA plots
├── model_eval.png                       # predicted-vs-actual, residuals, model comparison
├── december_chart.png                   # December rate chart
│
└── REPORT.md                            # full write-up: EDA, decisions, results, limitations
```

## Setup

```bash
pip install -r requirement.txt
```

## Usage

Run the pipeline in order from the project directory:

```bash
python 01_eda_prep.py       # inspects data, prints summary stats, saves eda_overview.png
python 02_train_eval.py     # time-based holdout comparison of baseline / linear / tree / bagged-tree models
python 03_final_predict.py  # refits the selected model on all training data, writes validation_predictions.csv
```

Each script is self-contained and writes its outputs (CSVs, PNGs) to the
working directory.

## Methodology Summary

- **Target:** `posted_rate` (continuous, USD) — a regression problem.
- **Key EDA finding:** `distance` is the dominant driver (r = 0.91 with
  `posted_rate`); `equipment` type shifts price (Reefer > Flatbed > Dry Van);
  `market_index` / `quote_signal` add weak but non-zero signal.
- **Data quality fixes:** negative `weight` values (~0.6% of rows) are a
  sign-flip entry error, corrected with `abs()`; missing `weight` /
  `market_index` are imputed with training-fold medians.
- **Generalization fix:** `validation.csv` contains 8 pickup/delivery cities
  never seen in training. Raw city names are dropped as features in favor of
  their (lat, lon) coordinates, which generalize to unseen locations.
- **Validation strategy:** time-based holdout (train on Jan 1–Sep 15,
  validate on Sep 16–Oct 31) rather than a random split, since the real
  validation set is strictly future relative to training — this avoids
  overstating generalization performance.
- **Models compared:** mean baseline, rate-per-mile-by-equipment baseline,
  Ridge linear regression, single CART tree, bagged trees (final choice).

Full reasoning — preprocessing decisions, feature importance, error
analysis, and limitations — is documented in [`REPORT.md`](REPORT.md).

## Results

| Model | MAE ($) | RMSE ($) | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| Baseline: mean | 1,173.9 | 1,515.7 | -0.00 | 82.9 |
| Baseline: rate-per-mile × equipment | 263.3 | 666.4 | 0.807 | 11.1 |
| Linear Regression (Ridge) | 162.7 | 616.9 | 0.834 | 8.7 |
| Regression Tree (single) | 149.0 | 615.3 | 0.835 | 6.9 |
| **Bagged Trees (n=25)** | **135.0** | **610.0** | **0.838** | **6.6** |

*(computed on the Sep 16–Oct 31 2025 out-of-time holdout; see `model_comparison.csv`)*

## Known Limitations

- Rare, extreme-surge loads (rate-per-mile far above the typical 1.7–3.2
  range) are systematically under-predicted — none of the available
  features explain them.
- `december-chart-inputs.csv` lacks daily market signals, so its predicted
  rate is flat across December by design, not a modeling defect — see
  `REPORT.md §12` for the full explanation.




