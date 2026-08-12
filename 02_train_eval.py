import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features_models import (fit_imputers, engineer_features, to_matrix,
                              RidgeRegressionNumpy, RegressionTreeNumpy, BaggedTreesNumpy,
                              regression_metrics, FEATURE_COLS)

np.random.seed(42)

train = pd.read_csv("train-test.csv", parse_dates=["date"])

# ---------------------------------------------------------------------
# TIME-BASED SPLIT
# Deployment validation.csv is strictly FUTURE (Nov-Dec) relative to
# train-test.csv (Jan-Oct). We mimic this with an internal time-based
# holdout: train on Jan 1 - Sep 15, validate on Sep 16 - Oct 31.
# A random split would overestimate performance because it would not
# test the model's ability to extrapolate forward in time / to new
# lanes, which is exactly the situation validation.csv presents.
# ---------------------------------------------------------------------
cutoff = pd.Timestamp('2025-09-16')
dev_train = train[train['date'] < cutoff].reset_index(drop=True)
dev_val   = train[train['date'] >= cutoff].reset_index(drop=True)
print(f"dev_train rows: {len(dev_train)}  ({dev_train['date'].min().date()} -> {dev_train['date'].max().date()})")
print(f"dev_val   rows: {len(dev_val)}  ({dev_val['date'].min().date()} -> {dev_val['date'].max().date()})")

# Fit imputation stats on TRAIN ONLY (avoid leakage)
stats = fit_imputers(dev_train)
print("Imputation stats (fit on training fold only):", stats)

dev_train_fe = engineer_features(dev_train, stats)
dev_val_fe   = engineer_features(dev_val, stats)

# Winsorize target for TRAINING rows only (cap extreme outliers that look
# like data-entry errors / rare extreme loads) - never touch val/test targets,
# we always evaluate against the real observed value.
lo, hi = dev_train_fe['posted_rate'].quantile([0.005, 0.995])
y_train_raw = dev_train_fe['posted_rate'].to_numpy()
y_train_w = np.clip(y_train_raw, lo, hi)
print(f"Winsorizing training target at [{lo:.1f}, {hi:.1f}] "
      f"affects {(np.abs(y_train_raw-y_train_w)>1e-6).sum()} / {len(y_train_raw)} training rows")

X_train = to_matrix(dev_train_fe)
X_val   = to_matrix(dev_val_fe)
y_val   = dev_val_fe['posted_rate'].to_numpy()

results = {}

# ---------------- Baseline 1: global mean ----------------
pred_mean = np.full_like(y_val, y_train_w.mean())
results['Baseline: mean'] = regression_metrics(y_val, pred_mean)

# ---------------- Baseline 2: mean rate-per-mile x distance x equipment ----------------
rpm = (dev_train_fe['posted_rate'] / dev_train_fe['distance'])
eq_rpm = rpm.groupby(dev_train_fe['equipment']).mean()
overall_rpm = rpm.mean()
pred_rpm = dev_val_fe['equipment'].map(eq_rpm).fillna(overall_rpm).to_numpy() * dev_val_fe['distance'].to_numpy()
results['Baseline: rate-per-mile x equipment'] = regression_metrics(y_val, pred_rpm)

# ---------------- Model A: Ridge Linear Regression ----------------
lin = RidgeRegressionNumpy(alpha=5.0)
lin.fit(X_train, y_train_w)
pred_lin = lin.predict(X_val)
results['Linear Regression (Ridge)'] = regression_metrics(y_val, pred_lin)

# ---------------- Model B: single Regression Tree ----------------
tree = RegressionTreeNumpy(max_depth=8, min_samples_leaf=60, min_samples_split=120)
tree.fit(X_train, y_train_w)
pred_tree = tree.predict(X_val)
results['Regression Tree (single)'] = regression_metrics(y_val, pred_tree)

# ---------------- Model C: Bagged Trees (small random forest) ----------------
bag = BaggedTreesNumpy(n_estimators=15, max_depth=8, min_samples_leaf=60,
                        min_samples_split=120, max_features=10, random_state=42)
bag.fit(X_train, y_train_w)
pred_bag = bag.predict(X_val)
results['Bagged Trees (n=15)'] = regression_metrics(y_val, pred_bag)

print("\n" + "="*70)
print("INTERNAL TIME-BASED HOLDOUT RESULTS")
print("="*70)
res_df = pd.DataFrame(results).T[['MAE','RMSE','R2','MAPE']]
print(res_df.round(3))
res_df.to_csv('model_comparison.csv')

# Linear regression coefficients (standardized) for interpretability
coef_df = pd.DataFrame({'feature': FEATURE_COLS, 'std_coef': lin.coef_}).sort_values('std_coef', key=abs, ascending=False)
print("\nRidge regression standardized coefficients:\n", coef_df)
coef_df.to_csv('linear_coefficients.csv', index=False)

# ---------------- Permutation importance for the bagged-trees model ----------------
baseline_mae = regression_metrics(y_val, pred_bag)['MAE']
importances = []
rng = np.random.RandomState(0)
for i, feat in enumerate(FEATURE_COLS):
    Xp = X_val.copy()
    rng.shuffle(Xp[:, i])
    pred_p = bag.predict(Xp)
    mae_p = regression_metrics(y_val, pred_p)['MAE']
    importances.append(mae_p - baseline_mae)
imp_df = pd.DataFrame({'feature': FEATURE_COLS, 'mae_increase': importances}).sort_values('mae_increase', ascending=False)
print("\nPermutation importance (bagged trees, increase in MAE when shuffled):\n", imp_df)
imp_df.to_csv('permutation_importance.csv', index=False)

# ---------------- Error analysis ----------------
err = y_val - pred_bag
dev_val_fe['abs_err'] = np.abs(err)
dev_val_fe['pred'] = pred_bag
print("\nWorst 10 predictions (bagged trees):")
worst = dev_val_fe.sort_values('abs_err', ascending=False).head(10)
print(worst[['load_id','pickup','delivery','distance','equipment','posted_rate','pred','abs_err']])

print("\nMAE by equipment:")
print(dev_val_fe.groupby('equipment')['abs_err'].mean())

dev_val_fe['distance_bin'] = pd.cut(dev_val_fe['distance'], bins=[0,300,700,1500,2500,4000])
print("\nMAE by distance bin:")
print(dev_val_fe.groupby('distance_bin', observed=True)['abs_err'].mean())

# ---------------- Plots: predicted vs actual + residuals ----------------
fig, axes = plt.subplots(1, 3, figsize=(15,4.5))
axes[0].scatter(y_val, pred_bag, s=4, alpha=0.2, color='#3B6FA0')
lims = [0, max(y_val.max(), pred_bag.max())]
axes[0].plot(lims, lims, 'r--', lw=1)
axes[0].set_xlabel('Actual posted_rate'); axes[0].set_ylabel('Predicted')
axes[0].set_title('Bagged Trees: Predicted vs Actual')

axes[1].scatter(pred_bag, err, s=4, alpha=0.2, color='#3B6FA0')
axes[1].axhline(0, color='r', ls='--', lw=1)
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residual (actual-pred)')
axes[1].set_title('Residuals vs Predicted')

model_names = list(results.keys())
mae_vals = [results[m]['MAE'] for m in model_names]
axes[2].barh(model_names, mae_vals, color='#3B6FA0')
axes[2].set_xlabel('MAE ($)'); axes[2].set_title('Model comparison (holdout MAE)')

plt.tight_layout()
plt.savefig('model_eval.png', dpi=130)
print("\nSaved model_eval.png")
