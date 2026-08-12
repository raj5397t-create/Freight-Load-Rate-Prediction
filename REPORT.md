# Freight Load Rate Prediction — End-to-End ML Project

## 1. Problem Statement

**Business objective:** A freight brokerage needs to predict what a load will cost
(`posted_rate`, in USD) before it is quoted, using load characteristics known at
quote time (lane, distance, equipment type, weight, and two market signals).

**ML objective:** Given load-level features, predict the numeric `posted_rate`
for each load in `validation.csv`, and write the results to
`validation_predictions.csv` in the exact format of the provided template.

**Problem type: Regression.** `posted_rate` is a continuous, strictly positive
dollar amount (min $57, median $2,031, max $25,533) with no discrete classes —
this is a standard tabular regression problem, not classification or
time-series forecasting in the strict sense (there is no autocorrelated
sequence to forecast — each load is an independent pricing event, though the
split does need to respect time, see §4).

**Unit of prediction:** one row = one load (one truck movement, uniquely
identified by `load_id`) between a pickup and delivery city, on a given date,
requiring a given equipment type and weight. One prediction = the dollar rate
that load should be priced at.

## 2. Note on requirement.txt vs. the master prompt

The master prompt suggests scikit-learn, seaborn, etc. `requirement.txt`
(the actual project constraint) lists only `pandas`, `numpy`, and
`matplotlib`. **I followed `requirement.txt`** as the authoritative
environment spec: all models (linear regression, decision tree, bagged
trees/ensemble) are implemented from scratch with numpy, and all plots use
matplotlib only. This is flagged per Rule 9 (handle ambiguity explicitly).

## 3. Data Overview (from actual inspection, see `01_eda_prep.py`)

| File | Rows | Cols | Role |
|---|---|---|---|
| train-test.csv | 48,000 | 14 | labeled development data (`posted_rate` present) |
| validation.csv | 12,000 | 13 | unlabeled — needs predictions |
| validation-predictions-template.csv | 12,000 | 2 | `load_id`, empty `predicted_rate` |
| december-chart-inputs.csv | 31 | 7 | bonus scenario: one fixed lane, every day in Dec 2025 |

Key findings:
- **No duplicate rows or `load_id`s.**
- **Missing values:** `weight` (~0.6%) and `market_index` (~0.8%) in both
  train and validation, missing roughly proportionally across equipment types
  (looks like MCAR — missing completely at random — not a systematic gap).
- **Data quality bug:** ~0.6% of `weight` values are **negative**
  (e.g. -47,500 lb). `abs(negative weight)` has almost the exact same
  distribution as the valid positive weights (median ~31.8k vs ~31.5k) →
  this is a sign-flip data-entry error, not a different population, so we
  take `abs(weight)` rather than treating it as invalid/missing.
- **`distance` is the dominant driver:** correlation with `posted_rate` = 0.91.
  It is also ~99.95% correlated with the great-circle (haversine) distance
  computed from lat/lon, confirming it's a real, trustworthy road-distance
  feature (not corrupted).
- **`equipment` matters:** mean rate-per-mile is Dry Van $2.12 < Flatbed
  $2.29 < Reefer $2.38.
- **`market_index` and `quote_signal`** have very weak linear correlation
  with `posted_rate` (|r| < 0.07) individually — they still carry a small
  amount of independent signal (confirmed later via permutation importance)
  but are not dominant drivers.
- **Critical leakage/generalization issue:** `validation.csv` contains **8
  cities never seen in `train-test.csv`** (Allentown, Charlotte, Chicago,
  Jackson, Knoxville, Laredo, Norfolk, San Diego), both as pickup and
  delivery. A model using pickup/delivery **city name** as a categorical
  (one-hot / target-encoded) feature would fail or silently degrade on ~time
  of these loads. **Decision: drop raw city names as features entirely,
  and use their (lat, lon) coordinates instead** — continuous features
  generalize to unseen locations, categorical dummies do not.
- **Right-skewed target** with a small number (~0.5%) of extreme outliers
  (rate-per-mile up to 14 vs. a median of ~2.1) — handled via winsorizing
  the *training* target only (never the evaluation target — see §6/§10).
- **No true leakage columns**: no feature is a deterministic function of
  the target, and `load_id` carries no signal (sequential ID, unrelated to
  price).

## 4. Train / Validation Strategy

`validation.csv` is entirely **out-of-time**: train-test.csv covers
Jan 1 – Oct 31, 2025, while validation.csv covers Nov 1 – Dec 31, 2025 (the
future). A random split of train-test.csv would let the model "see" rows
from every month during training and therefore **overstate** how well it
will generalize to genuinely future, unseen months.

**Decision:** internal holdout is **time-based**: train on Jan 1 – Sep 15
(40,850 rows), validate on Sep 16 – Oct 31 (7,150 rows, the most recent ~6
weeks). All imputation statistics (medians) and outlier bounds are fit on
the training fold only, then applied unchanged to the holdout fold — and
later, the same fitted logic (refit on 100% of train-test.csv) is applied
unchanged to `validation.csv`. This prevents any information from "the
future" leaking backward into training.

## 5. Preprocessing & Feature Engineering

| Step | Decision | Why |
|---|---|---|
| `weight` | `abs()`, then missing → median of training fold | sign-flip bug; MCAR missingness |
| `market_index` | missing → median of training fold; added `market_missing` flag | MCAR; flag lets the model use "was imputed" as a weak signal if useful |
| `pickup`/`delivery` city names | **dropped** | 8 unseen cities in validation — categorical dummies can't generalize |
| lat/lon (4 cols) | kept as continuous features | generalizes to new locations; captures regional pricing patterns |
| `distance` | kept as-is | strongest single predictor (r=0.91), already validated against haversine distance |
| `equipment` | one-hot (`eq_Reefer`, `eq_Flatbed`; Dry Van = baseline) | only 3 categories, clean effect on price |
| `date` | converted to `month_sin`/`month_cos` (cyclical encoding) | captures the mild seasonal drift observed in the EDA (rate-per-mile trends up through mid-year); day-of-week was tested and found to have negligible effect (<1.2% swing, noise-level) so it was **not** added, to avoid overfitting to noise |
| `quote_signal` | kept numeric, unscaled input to tree models / standardized for linear model | weak but non-zero independent signal |
| target `posted_rate` | **winsorized at [0.5%, 99.5%] on the training fold only** before fitting | caps the leverage of a handful of extreme-outlier loads on the fitted model, without ever altering the values we evaluate/predict against |
| `load_id` | excluded | identifier only, no predictive meaning |

No scaling was needed for the tree-based models (scale-invariant); the
linear model standardizes internally.

## 6. Baselines

1. **Global mean** — MAE $1,174, R² ≈ 0 (by construction). Confirms the
   problem is not trivial.
2. **Rate-per-mile × equipment** (`mean(posted_rate/distance | equipment) ×
   distance`) — a strong, realistic industry-style baseline. MAE $263,
   R² 0.81. Any modeled approach must beat this convincingly to be worth
   the added complexity.

## 7. Candidate Models (from scratch, numpy only)

- **Ridge Linear Regression** (closed-form normal equations, standardized
  features) — interpretable, fast, a reasonable choice given `distance` is
  near-linear with `posted_rate`.
- **Single CART Regression Tree** — captures non-linear interactions
  (e.g., equipment × distance × region) the linear model can't.
- **Bagged Trees (25-tree ensemble, bootstrap + feature subsampling)** — a
  small from-scratch random forest; reduces the variance of a single tree.

## 8. Results — Internal Time-Based Holdout (Sep 16–Oct 31 2025)

| Model | MAE ($) | RMSE ($) | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| Baseline: mean | 1,173.9 | 1,515.7 | -0.00 | 82.9 |
| Baseline: rate-per-mile × equipment | 263.3 | 666.4 | 0.807 | 11.1 |
| Linear Regression (Ridge) | 162.7 | 616.9 | 0.834 | 8.7 |
| Regression Tree (single) | 149.0 | 615.3 | 0.835 | 6.9 |
| **Bagged Trees (n=25)** | **135.0** | **610.0** | **0.838** | **6.6** |

**Why RMSE barely moves while MAE improves a lot:** a small number of
extreme, high-value loads (rate-per-mile far above the typical 1.7–3.2
range) dominate the squared-error metric for every model — see error
analysis below. MAE and MAPE (driven by the *typical* load) improve
substantially model-to-model; RMSE (driven by the *worst* loads) barely
moves, because none of the available features explain those extreme cases.

**Model selected: Bagged Trees.** It has the best MAE/RMSE/R²/MAPE on the
out-of-time holdout, handles the non-linear equipment×distance×region
interactions the linear model misses, and — being scale-invariant and
built from an ensemble of shallow trees — is reasonably stable and not
excessively complex (25 trees, depth 8, no deep-learning-scale
infrastructure needed).

## 9. Feature Importance (permutation importance, bagged trees, holdout)

| Feature | MAE increase when shuffled ($) |
|---|---:|
| distance | 1,354.7 |
| eq_Reefer | 46.3 |
| pickup_lon | 28.9 |
| delivery_lon | 21.4 |
| quote_signal | 20.6 |
| eq_Flatbed | 10.0 |
| weight_clean | 4.8 |
| pickup_lat / delivery_lat | <1 each |
| month_sin/cos, market_index, missing-flags | ~0 |

`distance` overwhelmingly dominates, consistent with the correlation
analysis. `equipment` and rough longitude (an East–West regional price
gradient) contribute modestly. `market_index` and the month features add
almost nothing once distance and equipment are accounted for — this is
**importance, not causation**: it says the model doesn't need these columns
to reduce its error on this holdout, not that market conditions don't
affect real-world rates.

Linear-regression standardized coefficients tell the same story in a
different form (`distance` dominates; `eq_Reefer`/`eq_Flatbed` push rates
up relative to Dry Van, lat/lon and month/quote effects are much smaller,
directionally consistent with the EDA) — see `linear_coefficients.csv`.

## 10. Error Analysis

- The 10 worst-predicted holdout loads are all cases where `posted_rate`
  is far above what distance/equipment predict — e.g. a $17-18k rate on a
  route where the model (correctly, based on the training pattern) expects
  $3-6k. These look like genuine outlier pricing events (spot-market
  surges, expedited loads, etc.) that the available features cannot
  explain — not a coding bug (verified against the raw rows).
- **MAE by equipment:** Dry Van $119.5 < Reefer $152.2 < Flatbed $159.0 —
  the two less-common, higher-variance equipment types are modestly harder
  to predict, consistent with having ~3x fewer training examples than Dry
  Van.
- **MAE by distance bucket:** grows from $53 (short-haul, <300 mi) to $269
  (long-haul, 2500-4000 mi) — expected, since absolute dollar amounts (and
  their natural variance) scale with distance; MAPE is the fairer metric
  across distance bands.
- **Implication:** the model is well-calibrated for typical loads but will
  systematically under-predict rare, extreme-surge loads. In production,
  pairing the point prediction with a prediction interval (e.g., from the
  spread across the 25 bagged trees) would flag these as high-uncertainty
  cases rather than presenting a single number with false confidence.

## 11. Final Validation Predictions

The bagged-trees model was **refit on 100% of `train-test.csv`** (48,000
rows; same imputation/winsorization logic, refit — not reused — on the
full data) and applied to all 12,000 rows of `validation.csv`.
`validation_predictions.csv` follows the exact template format
(`load_id`, `predicted_rate`), with predictions floored at $1 (rates
cannot be ≤ 0). Sanity check: the predicted rate-per-mile distribution on
validation (mean 2.20, and Dry Van $2.15 < Flatbed $2.20 < Reefer $2.31)
closely matches the training-data pattern — no evidence of systematic
miscalibration.

## 12. December Chart (bonus file)

`december-chart-inputs.csv` asks for a day-by-day predicted rate for one
fixed lane (Lexington→Fort Wayne, Dry Van, 32,000 lb) across all of
December. Two honest limitations, both handled explicitly rather than
papered over:

1. **No lat/lon in that file** — recovered via a lookup table built from
   every city's known coordinates in train-test.csv/validation.csv (both
   cities were present).
2. **No `market_index`/`quote_signal` in that file** — these are imputed at
   their training-median values, since no scenario-specific values were
   supplied.

**Result: the predicted rate is flat ($849.99) across every day in
December.** This is not a bug — with lane, equipment, weight, and month all
held constant, and `market_index`/`quote_signal` imputed to constant
medians, **every input feature the model has is identical on every row**,
so it produces an identical prediction. The EDA (§3) confirmed day-of-week
effects are noise-level (~1%) in the training data, so no feature was
engineered to fabricate day-to-day movement. A genuinely date-varying
December forecast would require actual daily market_index/quote_signal
values (or a real day-of-week/holiday effect, which the data doesn't
support) — this is a data limitation, not a modeling one.

## 13. Limitations & Future Improvements

- **Extreme-outlier loads** (surge pricing) are not explained by the
  available features; a quantile-regression or prediction-interval
  approach (e.g., using the spread across the bagged trees) would better
  communicate uncertainty than a single point estimate.
- **`market_index`/`quote_signal` semantics are unknown** — they were used
  as opaque numeric signals since their real-world meaning isn't
  documented; understanding what they represent could unlock more feature
  engineering (e.g., interactions with distance/equipment).
- **No lane-pair (pickup×delivery) interaction feature** was engineered
  beyond raw coordinates, specifically to keep the model generalizable to
  the unseen validation cities; a production system with a stable, closed
  set of lanes could add lane-level historical-average features for a
  likely accuracy boost.
- **Deployment:** the fitted model (imputation stats + 25 trees) should be
  serialized (e.g., pickled) and served behind the same feature-engineering
  pipeline (`features_models.py`) used here, with monitoring for feature
  drift (e.g., new cities, new equipment types, market_index/quote_signal
  distribution shifts) since both validation.csv and the December file
  already show the kind of drift (new cities) a production system must
  handle gracefully.

## 14. Files in this project

- `01_eda_prep.py`, `02_train_eval.py`, `03_final_predict.py` — full
  executable pipeline (run in order)
- `features_models.py` — feature engineering + from-scratch Ridge/Tree/Bagged-Trees implementations
- `eda_overview.png`, `model_eval.png`, `december_chart.png` — plots
- `model_comparison.csv`, `linear_coefficients.csv`, `permutation_importance.csv` — result tables
- **`validation_predictions.csv`** — the required deliverable (12,000 rows)
- `december_chart_predictions.csv` — bonus deliverable
