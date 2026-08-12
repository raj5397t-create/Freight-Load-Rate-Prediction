"""
Freight Load Rate Prediction — EDA & Preprocessing
Only uses numpy / pandas / matplotlib, per requirement.txt
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

train = pd.read_csv("train-test.csv", parse_dates=["date"])
val   = pd.read_csv("validation.csv", parse_dates=["date"])
tmpl  = pd.read_csv("validation-predictions-template.csv")
chart = pd.read_csv("december-chart-inputs.csv", parse_dates=["date"])

print("="*70)
print("SHAPES")
print("="*70)
print("train-test.csv :", train.shape)
print("validation.csv :", val.shape)
print("template       :", tmpl.shape)
print("chart inputs   :", chart.shape)

print("\n" + "="*70)
print("TRAIN DTYPES")
print("="*70)
print(train.dtypes)

print("\n" + "="*70)
print("MISSING VALUES (train / val)")
print("="*70)
print(pd.DataFrame({"train": train.isnull().sum(), "val": val.reindex(columns=train.columns.drop('posted_rate')).isnull().sum() if False else val.isnull().sum().reindex(train.columns.drop('posted_rate'))}))

print("\nDuplicate rows (train):", train.duplicated().sum())
print("Duplicate load_id (train):", train['load_id'].duplicated().sum())
print("Duplicate load_id (val):", val['load_id'].duplicated().sum())

print("\n" + "="*70)
print("CARDINALITY")
print("="*70)
for c in ["pickup","delivery","equipment"]:
    print(c, "-> train unique:", train[c].nunique(), "| val unique:", val[c].nunique())

new_pickup = set(val['pickup']) - set(train['pickup'])
new_deliv  = set(val['delivery']) - set(train['delivery'])
print("\nCities present in validation but NEVER seen in train (pickup):", sorted(new_pickup))
print("Cities present in validation but NEVER seen in train (delivery):", sorted(new_deliv))

print("\n" + "="*70)
print("DATE RANGES")
print("="*70)
print("train:", train['date'].min().date(), "->", train['date'].max().date())
print("val  :", val['date'].min().date(),   "->", val['date'].max().date())
print("chart:", chart['date'].min().date(), "->", chart['date'].max().date())

print("\n" + "="*70)
print("WEIGHT SANITY CHECK (negative values)")
print("="*70)
print("train negative weight rows:", (train['weight']<0).sum(), "/", train['weight'].notnull().sum())
print("val   negative weight rows:", (val['weight']<0).sum(), "/", val['weight'].notnull().sum())
print("abs(negative) describe (train):\n", train.loc[train['weight']<0,'weight'].abs().describe())
print("positive describe (train):\n", train.loc[train['weight']>=0,'weight'].describe())

print("\n" + "="*70)
print("TARGET DISTRIBUTION: posted_rate")
print("="*70)
print(train['posted_rate'].describe())
train['rate_per_mile'] = train['posted_rate'] / train['distance']
print("\nrate_per_mile describe:\n", train['rate_per_mile'].describe())
print("\nrate_per_mile quantiles (tails):")
print(train['rate_per_mile'].quantile([0.0005,0.001,0.005,0.01,0.99,0.995,0.999,0.9995]))

print("\n" + "="*70)
print("EQUIPMENT EFFECT ON rate_per_mile")
print("="*70)
print(train.groupby('equipment')['rate_per_mile'].agg(['count','mean','median','std']))

print("\n" + "="*70)
print("CORRELATIONS WITH posted_rate")
print("="*70)
num_cols = ['distance','weight','market_index','quote_signal','posted_rate']
print(train[num_cols].corr()['posted_rate'])

# haversine sanity check for distance column
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2-lat1); dlambda = np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlambda/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
train['hav_dist'] = haversine(train.pickup_lat, train.pickup_lon, train.delivery_lat, train.delivery_lon)
print("\ncorr(distance, haversine great-circle distance):", train['distance'].corr(train['hav_dist']))

# ---------------- PLOTS ----------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

axes[0,0].hist(train['posted_rate'], bins=80, color='#3B6FA0')
axes[0,0].set_title('posted_rate distribution')

axes[0,1].hist(train['rate_per_mile'].clip(upper=6), bins=80, color='#3B6FA0')
axes[0,1].set_title('rate_per_mile distribution (clipped at 6 for viz)')

axes[0,2].scatter(train['distance'], train['posted_rate'], s=3, alpha=0.15, color='#3B6FA0')
axes[0,2].set_xlabel('distance'); axes[0,2].set_ylabel('posted_rate')
axes[0,2].set_title('posted_rate vs distance')

train.boxplot(column='rate_per_mile', by='equipment', ax=axes[1,0])
axes[1,0].set_ylim(0, 5)
axes[1,0].set_title('rate_per_mile by equipment'); plt.suptitle('')

monthly = train.set_index('date')['rate_per_mile'].resample('W').mean()
axes[1,1].plot(monthly.index, monthly.values, color='#3B6FA0')
axes[1,1].set_title('weekly mean rate_per_mile over time (train period)')
axes[1,1].tick_params(axis='x', rotation=45)

corr = train[num_cols].corr()
im = axes[1,2].imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
axes[1,2].set_xticks(range(len(num_cols))); axes[1,2].set_xticklabels(num_cols, rotation=45, ha='right')
axes[1,2].set_yticks(range(len(num_cols))); axes[1,2].set_yticklabels(num_cols)
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        axes[1,2].text(j, i, f"{corr.values[i,j]:.2f}", ha='center', va='center', fontsize=8)
axes[1,2].set_title('correlation matrix')

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=130)
print("\nSaved eda_overview.png")
