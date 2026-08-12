import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features_models import (fit_imputers, engineer_features, to_matrix,
                              BaggedTreesNumpy, RidgeRegressionNumpy, regression_metrics, FEATURE_COLS)

np.random.seed(42)
t0 = time.time()

train = pd.read_csv("train-test.csv", parse_dates=["date"])
val   = pd.read_csv("validation.csv", parse_dates=["date"])
tmpl  = pd.read_csv("validation-predictions-template.csv")
chart = pd.read_csv("december-chart-inputs.csv", parse_dates=["date"])

# ---------------- Fit imputers + winsorization bounds on FULL training data ----------------
stats = fit_imputers(train)
train_fe = engineer_features(train, stats)

lo, hi = train_fe['posted_rate'].quantile([0.005, 0.995])
y_full = np.clip(train_fe['posted_rate'].to_numpy(), lo, hi)
X_full = to_matrix(train_fe)

print(f"Training final model on all {len(train_fe)} rows (target winsorized to [{lo:.1f}, {hi:.1f}])")

final_model = BaggedTreesNumpy(n_estimators=25, max_depth=8, min_samples_leaf=60,
                                min_samples_split=120, max_features=10, random_state=42)
final_model.fit(X_full, y_full)
print(f"Trained in {time.time()-t0:.1f}s")

# also refit linear model on full data, for the coefficient table in the report
lin_full = RidgeRegressionNumpy(alpha=5.0).fit(X_full, y_full)

# ---------------- Predict on validation.csv ----------------
val_fe = engineer_features(val, stats)
X_val = to_matrix(val_fe)
val_pred = final_model.predict(X_val)
val_pred = np.clip(val_pred, 1, None)  # rates can't be <= 0

out = tmpl.copy()
assert (out['load_id'] == val['load_id']).all()
out['predicted_rate'] = np.round(val_pred, 2)
out.to_csv('validation_predictions.csv', index=False)
print("\nSaved validation_predictions.csv:", out.shape)
print(out.head())
print("\npredicted_rate summary:\n", out['predicted_rate'].describe())

# sanity: compare predicted rate-per-mile distribution to training rate-per-mile
val_fe['pred'] = val_pred
val_fe['pred_rpm'] = val_fe['pred'] / val_fe['distance']
print("\nValidation predicted rate-per-mile summary (sanity check vs train ~2.0-2.4):\n",
      val_fe['pred_rpm'].describe())
print("\nValidation predicted rate-per-mile BY equipment:\n",
      val_fe.groupby('equipment')['pred_rpm'].mean())

# ---------------- December chart: build a city -> lat/lon lookup from all known cities ----------------
city_geo = pd.concat([
    train[['pickup','pickup_lat','pickup_lon']].rename(columns={'pickup':'city','pickup_lat':'lat','pickup_lon':'lon'}),
    train[['delivery','delivery_lat','delivery_lon']].rename(columns={'delivery':'city','delivery_lat':'lat','delivery_lon':'lon'}),
    val[['pickup','pickup_lat','pickup_lon']].rename(columns={'pickup':'city','pickup_lat':'lat','pickup_lon':'lon'}),
    val[['delivery','delivery_lat','delivery_lon']].rename(columns={'delivery':'city','delivery_lat':'lat','delivery_lon':'lon'}),
]).drop_duplicates(subset='city').set_index('city')

missing_cities = set(chart['pickup']).union(chart['delivery']) - set(city_geo.index)
print("\nDecember-chart cities missing coordinate lookup:", missing_cities)

chart_fe = chart.copy()
chart_fe['pickup_lat'] = chart_fe['pickup'].map(city_geo['lat'])
chart_fe['pickup_lon'] = chart_fe['pickup'].map(city_geo['lon'])
chart_fe['delivery_lat'] = chart_fe['delivery'].map(city_geo['lat'])
chart_fe['delivery_lon'] = chart_fe['delivery'].map(city_geo['lon'])
# market_index / quote_signal are not provided for this scenario file at all,
# so we impute them at the training median (documented assumption -- see report)
chart_fe['market_index'] = stats['market_median']
# quote_signal has no fitted imputer (never missing in training) -> use training median directly
chart_fe['quote_signal'] = train['quote_signal'].median()

chart_full_fe = engineer_features(chart_fe, stats)
X_chart = to_matrix(chart_full_fe)
chart_pred = final_model.predict(X_chart)
chart_out = chart.copy()
chart_out['predicted_rate'] = np.round(chart_pred, 2)
chart_out.to_csv('december_chart_predictions.csv', index=False)
print("\nSaved december_chart_predictions.csv")
print(chart_out[['date','predicted_rate']])

fig, ax = plt.subplots(figsize=(9,4.5))
ax.plot(chart_out['date'], chart_out['predicted_rate'], marker='o', ms=3, color='#3B6FA0')
ax.set_title('Predicted rate — Lexington to Fort Wayne, Dry Van, 32,000 lb — December 2025')
ax.set_xlabel('Date'); ax.set_ylabel('Predicted rate ($)')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('december_chart.png', dpi=130)
print("Saved december_chart.png")
print(f"\nTotal runtime: {time.time()-t0:.1f}s")
