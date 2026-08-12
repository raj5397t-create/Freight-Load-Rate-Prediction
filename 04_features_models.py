"""
Reusable feature engineering + from-scratch models (numpy/pandas only,
per requirement.txt which does not include scikit-learn).
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42

FEATURE_COLS = [
    'distance', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
    'weight_clean', 'market_index_clean', 'quote_signal',
    'eq_Reefer', 'eq_Flatbed', 'month_sin', 'month_cos',
    'weight_missing', 'market_missing'
]

def fit_imputers(df):
    """Compute imputation statistics from TRAINING data only (no leakage)."""
    weight_abs = df['weight'].abs()
    stats = {
        'weight_median': weight_abs.median(),
        'market_median': df['market_index'].median(),
    }
    return stats

def engineer_features(df, stats):
    df = df.copy()
    df['weight_missing'] = df['weight'].isnull().astype(float)
    df['weight_clean'] = df['weight'].abs()
    df['weight_clean'] = df['weight_clean'].fillna(stats['weight_median'])

    df['market_missing'] = df['market_index'].isnull().astype(float)
    df['market_index_clean'] = df['market_index'].fillna(stats['market_median'])

    df['eq_Reefer'] = (df['equipment'] == 'Reefer').astype(float)
    df['eq_Flatbed'] = (df['equipment'] == 'Flatbed').astype(float)

    month = df['date'].dt.month.astype(float)
    df['month_sin'] = np.sin(2*np.pi*month/12)
    df['month_cos'] = np.cos(2*np.pi*month/12)

    return df

def to_matrix(df):
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    return X


# ---------------------------------------------------------------------
# Model 1: Ridge Linear Regression (closed-form, numpy only)
# ---------------------------------------------------------------------
class RidgeRegressionNumpy:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.mean_ = None
        self.std_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_
        Xb = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        n_features = Xb.shape[1]
        reg = self.alpha * np.eye(n_features)
        reg[0, 0] = 0.0  # do not regularize intercept
        theta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        return self

    def predict(self, X):
        Xs = (X - self.mean_) / self.std_
        return self.intercept_ + Xs @ self.coef_


# ---------------------------------------------------------------------
# Model 2: CART Regression Tree (from scratch, numpy only)
# ---------------------------------------------------------------------
class TreeNode:
    __slots__ = ['feature', 'threshold', 'left', 'right', 'value']
    def __init__(self, value=None):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = value

class RegressionTreeNumpy:
    def __init__(self, max_depth=6, min_samples_leaf=100, min_samples_split=200,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.RandomState(random_state)
        self.root = None

    def _best_split(self, X, y, feat_idx):
        n = len(y)
        best_gain, best_feat, best_thr = -np.inf, None, None
        parent_sse = np.sum((y - y.mean())**2)

        for f in feat_idx:
            col = X[:, f]
            order = np.argsort(col)
            col_sorted = col[order]
            y_sorted = y[order]

            # candidate split points: skip duplicate values
            distinct = np.where(np.diff(col_sorted) > 1e-12)[0]
            if len(distinct) == 0:
                continue

            cs = np.cumsum(y_sorted)
            cs2 = np.cumsum(y_sorted**2)
            total_sum, total_sq = cs[-1], cs2[-1]

            idxs = distinct  # index i means split after position i (0-indexed)
            left_n = idxs + 1
            right_n = n - left_n
            valid = (left_n >= self.min_samples_leaf) & (right_n >= self.min_samples_leaf)
            if not np.any(valid):
                continue
            idxs = idxs[valid]; left_n = left_n[valid]; right_n = right_n[valid]

            left_sum = cs[idxs]; left_sq = cs2[idxs]
            right_sum = total_sum - left_sum; right_sq = total_sq - left_sq

            left_sse = left_sq - (left_sum**2)/left_n
            right_sse = right_sq - (right_sum**2)/right_n
            gain = parent_sse - (left_sse + right_sse)

            best_local = np.argmax(gain)
            if gain[best_local] > best_gain:
                best_gain = gain[best_local]
                best_feat = f
                pos = idxs[best_local]
                best_thr = (col_sorted[pos] + col_sorted[pos+1]) / 2.0

        return best_feat, best_thr, best_gain

    def _build(self, X, y, depth):
        node = TreeNode(value=y.mean())
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return node

        n_feats = X.shape[1]
        if self.max_features is not None:
            feat_idx = self.rng.choice(n_feats, size=min(self.max_features, n_feats), replace=False)
        else:
            feat_idx = np.arange(n_feats)

        feat, thr, gain = self._best_split(X, y, feat_idx)
        if feat is None or gain <= 1e-9:
            return node

        mask = X[:, feat] <= thr
        node.feature = feat
        node.threshold = thr
        node.left = self._build(X[mask], y[mask], depth+1)
        node.right = self._build(X[~mask], y[~mask], depth+1)
        return node

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _predict_row(self, row, node):
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return np.array([self._predict_row(row, self.root) for row in X])


class BaggedTreesNumpy:
    """Small random-forest-style ensemble of RegressionTreeNumpy."""
    def __init__(self, n_estimators=15, max_depth=7, min_samples_leaf=80,
                 min_samples_split=160, max_features=8, random_state=42):
        self.n_estimators = n_estimators
        self.trees = []
        self.params = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                            min_samples_split=min_samples_split, max_features=max_features)
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]
        self.trees = []
        for i in range(self.n_estimators):
            boot_idx = rng.randint(0, n, size=n)
            tree = RegressionTreeNumpy(random_state=self.random_state + i, **self.params)
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.column_stack([t.predict(X) for t in self.trees])
        return preds.mean(axis=1)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def regression_metrics(y_true, y_pred):
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res/ss_tot
    mape = np.mean(np.abs(err) / y_true) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
