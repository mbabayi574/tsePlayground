"""
Feature Selection Pipeline for TSE Stock Prediction.

Runs three independent feature‐ranking methods and intersects their
top‑K selections to produce a robust, compact feature set:

    1. **SHAP importance** – TreeExplainer on an XGBoost multi-class model.
    2. **Mutual Information** – sklearn's `mutual_info_classif` with
       discrete target (`signal_class`).
    3. **Recursive Feature Elimination (RFECV)** – wrapped around
       LightGBM with time-series–aware cross-validation.

The union/intersection logic is configurable: by default a feature must
appear in at least 2 of 3 methods to survive.  A correlation filter is
applied first to drop one member of any pair with |r| > 0.92.

Outputs
-------
    data/selected_features.json   – ordered list of selected feature names
    data/feature_selection_report.csv – per-feature scores from all methods
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[WARN] shap not installed – SHAP ranking will be skipped.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] lightgbm not installed – RFECV will use XGBoost instead.")

from symbols import SYMBOL_NAMES

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────
CORR_THRESHOLD = 0.92        # drop one of a correlated pair above this
MIN_METHODS    = 2           # feature must be selected by at least N methods
TOP_K_MI       = 35          # keep top-K from Mutual Information
TOP_K_SHAP     = 35          # keep top-K from SHAP
OUTPUT_JSON    = "data/selected_features.json"
OUTPUT_CSV     = "data/feature_selection_report.csv"

NON_FEATURE_COLS = {
    "date", "timestamp", "open", "high", "low", "close",
    "adjClose", "value", "volume", "count", "yesterday",
    "future_return", "signal", "signal_class", "symbol",
}


# ── Helpers ───────────────────────────────────────────────────────────

def load_all_data():
    """Load all processed CSVs and tag them with their symbol name."""
    dfs = []
    for symbol in SYMBOL_NAMES:
        path = f"data/processed/{symbol}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["symbol"] = symbol
            if "date" in df.columns:
                df = df.sort_values("date").reset_index(drop=True)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def discover_features(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in NON_FEATURE_COLS]


def clean_X(df, features):
    X = df[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X.columns:
        lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
        if lo != hi:
            X[col] = X[col].clip(lo, hi)
    X.fillna(0, inplace=True)
    return X


# ── Step 1: Correlation Filter ────────────────────────────────────────

def correlation_filter(X, threshold=CORR_THRESHOLD):
    """
    Remove one member of each highly-correlated pair.
    Keeps the feature that has higher mean absolute correlation with all
    others (i.e. drops the more redundant one).
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        if correlated:
            # Drop whichever has higher avg correlation overall
            avg_corr_col = corr[col].mean()
            for c in correlated:
                avg_corr_c = corr[c].mean()
                drop_me = col if avg_corr_col > avg_corr_c else c
                to_drop.add(drop_me)
    kept = [c for c in X.columns if c not in to_drop]
    print(f"\n── Correlation filter (|r| > {threshold}) ──")
    print(f"   Dropped {len(to_drop)} features: {sorted(to_drop)}")
    print(f"   Remaining: {len(kept)} features")
    return kept, to_drop


# ── Step 2: Mutual Information ────────────────────────────────────────

def mutual_information_ranking(X, y, top_k=TOP_K_MI):
    print(f"\n── Mutual Information (top {top_k}) ──")
    mi_scores = mutual_info_classif(X, y, discrete_features=False,
                                     n_neighbors=5, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    selected = mi_series.head(top_k).index.tolist()
    print(f"   Top 10: {selected[:10]}")
    return mi_series, selected


# ── Step 3: SHAP Importance ──────────────────────────────────────────

def shap_importance_ranking(X, y, top_k=TOP_K_SHAP):
    print(f"\n── SHAP Importance (top {top_k}) ──")
    if not HAS_SHAP:
        print("   [SKIP] shap not available")
        return pd.Series(dtype=float), []

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
        gamma=2, reg_alpha=0.5, reg_lambda=2.0,
        objective="multi:softprob", num_class=5,
        eval_metric="mlogloss", random_state=42,
        verbosity=0, tree_method="hist",
    )
    # Use last 80% for train (time-aware)
    n = len(X)
    split = int(n * 0.8)
    model.fit(X.iloc[:split], y.iloc[:split])

    explainer = shap.TreeExplainer(model)
    # Sample up to 2000 rows for speed
    sample_idx = np.random.RandomState(42).choice(
        range(split, n), size=min(2000, n - split), replace=False
    )
    shap_values = explainer.shap_values(X.iloc[sample_idx])

    # Compute mean absolute SHAP value per feature across all samples & classes
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray):
        n_feats = X.shape[1]
        feat_axes = [i for i, dim in enumerate(shap_values.shape) if dim == n_feats]
        if feat_axes:
            feat_axis = feat_axes[0]
            other_axes = tuple(i for i in range(shap_values.ndim) if i != feat_axis)
            mean_abs = np.abs(shap_values).mean(axis=other_axes)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    shap_series = pd.Series(mean_abs, index=X.columns).sort_values(ascending=False)
    selected = shap_series.head(top_k).index.tolist()
    print(f"   Top 10: {selected[:10]}")
    return shap_series, selected


# ── Step 4: Recursive Feature Elimination ────────────────────────────

def rfe_ranking(X, y):
    print(f"\n── Recursive Feature Elimination (RFECV) ──")
    if HAS_LGBM:
        estimator = LGBMClassifier(
            n_estimators=200, max_depth=5, num_leaves=20,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
            min_child_samples=30, reg_alpha=0.5, reg_lambda=2.0,
            objective="multiclass", num_class=5,
            random_state=42, verbosity=-1,
        )
    else:
        estimator = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            objective="multi:softprob", num_class=5,
            random_state=42, verbosity=0, tree_method="hist",
        )

    cv = TimeSeriesSplit(n_splits=3)
    selector = RFECV(
        estimator, step=3, cv=cv,
        scoring="f1_macro", min_features_to_select=15,
        n_jobs=-1, verbose=0,
    )
    selector.fit(X, y)

    rfe_mask = selector.support_
    rfe_ranking_vals = selector.ranking_
    selected = X.columns[rfe_mask].tolist()
    rfe_series = pd.Series(rfe_ranking_vals, index=X.columns)
    print(f"   Selected {len(selected)} features")
    print(f"   Top 10: {selected[:10]}")
    return rfe_series, selected


# ── Step 5: Combine ──────────────────────────────────────────────────

def combine_selections(features, mi_selected, shap_selected, rfe_selected,
                        mi_scores, shap_scores, rfe_ranks,
                        min_methods=MIN_METHODS):
    """
    Keep features selected by at least `min_methods` of the 3 methods.
    Rank the final set by average normalised score.
    """
    print(f"\n── Combining (min {min_methods} of 3 methods agree) ──")

    # Build vote counts
    votes = {}
    for f in features:
        count = 0
        if f in mi_selected:
            count += 1
        if f in shap_selected:
            count += 1
        if f in rfe_selected:
            count += 1
        votes[f] = count

    selected = [f for f, v in votes.items() if v >= min_methods]

    # Build report dataframe
    report = pd.DataFrame(index=features)
    report["mi_score"] = mi_scores.reindex(features).values if len(mi_scores) else 0
    report["shap_score"] = shap_scores.reindex(features).values if len(shap_scores) else 0
    report["rfe_rank"] = rfe_ranks.reindex(features).values if len(rfe_ranks) else 0
    report["in_mi"] = report.index.isin(mi_selected)
    report["in_shap"] = report.index.isin(shap_selected)
    report["in_rfe"] = report.index.isin(rfe_selected)
    report["vote_count"] = [votes[f] for f in features]
    report["selected"] = report.index.isin(selected)

    # Normalise scores to [0, 1] for ranking
    for col in ["mi_score", "shap_score"]:
        mx = report[col].max()
        if mx > 0:
            report[f"{col}_norm"] = report[col] / mx
        else:
            report[f"{col}_norm"] = 0
    # RFE rank: lower is better → invert
    if report["rfe_rank"].max() > 0:
        report["rfe_score_norm"] = 1 - (report["rfe_rank"] - 1) / max(report["rfe_rank"].max() - 1, 1)
    else:
        report["rfe_score_norm"] = 0

    report["avg_norm_score"] = (
        report["mi_score_norm"] + report["shap_score_norm"] + report["rfe_score_norm"]
    ) / 3

    report = report.sort_values("avg_norm_score", ascending=False)

    # Order final selected list by avg score
    selected = [f for f in report.index if f in selected]

    print(f"   Final: {len(selected)} features selected")
    return selected, report


# ── Main ─────────────────────────────────────────────────────────────

def run_feature_selection():
    print("=" * 60)
    print("Feature Selection Pipeline")
    print("=" * 60)

    # Load data
    df = load_all_data()
    print(f"Loaded {len(df):,} rows from {df['symbol'].nunique()} symbols")

    all_features = discover_features(df)
    print(f"Starting features: {len(all_features)}")

    X = clean_X(df, all_features)
    y = df["signal_class"].astype(int)

    # Drop rows with NaN target
    valid = y.notna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    # Step 1: Correlation filter
    features_after_corr, dropped_corr = correlation_filter(X)
    X_filtered = X[features_after_corr]

    # Step 2: Mutual Information
    mi_scores, mi_selected = mutual_information_ranking(X_filtered, y)

    # Step 3: SHAP
    shap_scores, shap_selected = shap_importance_ranking(X_filtered, y)

    # Step 4: RFE
    rfe_ranks, rfe_selected = rfe_ranking(X_filtered, y)

    # Step 5: Combine
    selected, report = combine_selections(
        features_after_corr,
        mi_selected, shap_selected, rfe_selected,
        mi_scores, shap_scores, rfe_ranks,
    )

    # Save outputs
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\n✓ Saved {len(selected)} selected features → {OUTPUT_JSON}")

    report.to_csv(OUTPUT_CSV)
    print(f"✓ Saved full report → {OUTPUT_CSV}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {len(all_features)} → {len(features_after_corr)} (corr filter) "
          f"→ {len(selected)} (consensus)")
    print(f"{'=' * 60}")
    print("\nSelected features (ordered by importance):")
    for i, feat in enumerate(selected, 1):
        row = report.loc[feat]
        mi_flag = "MI" if row["in_mi"] else "  "
        sh_flag = "SH" if row["in_shap"] else "  "
        rf_flag = "RF" if row["in_rfe"] else "  "
        print(f"  {i:2d}. {feat:<30s}  [{mi_flag}] [{sh_flag}] [{rf_flag}]  "
              f"avg={row['avg_norm_score']:.3f}")

    return selected, report


# ── Lookback Lag Optimization via Random Forest (Guideline B) ─────────────

from sklearn.ensemble import RandomForestRegressor

def optimize_lookback_lags_rf(series, max_lags: int = 30, min_lags: int = 5):
    """
    Optimizes lookback lag length for time-series / residual sequences using a Random Forest regressor.
    
    Treats past lag steps (t-1, t-2, ..., t-max_lags) as candidate features to predict target at t.
    Analyzes resulting feature importances to establish statistically optimal sequence length.
    
    Args:
        series (pd.Series or np.ndarray): Input time series (e.g., prices or ARIMA residuals).
        max_lags (int): Maximum candidate lookback window to evaluate.
        min_lags (int): Minimum enforced lookback sequence length.
        
    Returns:
        tuple: (optimal_lag_length, pd.Series of lag importances)
    """
    series_arr = np.array(series).flatten()
    if len(series_arr) <= max_lags + 10:
        return max(min_lags, len(series_arr) // 4), pd.Series()

    # Build lag candidate feature matrix X and target y
    X_lags, y_target = [], []
    for t in range(max_lags, len(series_arr)):
        X_lags.append(series_arr[t - max_lags : t][::-1])  # lag_1, lag_2, ..., lag_max_lags
        y_target.append(series_arr[t])

    X_lags = np.array(X_lags)
    y_target = np.array(y_target)

    # Train Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_lags, y_target)

    importances = rf.feature_importances_
    lag_names = [f"lag_{i}" for i in range(1, max_lags + 1)]
    imp_series = pd.Series(importances, index=lag_names)

    # Determine cutoff: Find smallest lag step window covering >= 85% of cumulative importance
    cum_imp = np.cumsum(importances)
    cutoff_idx = np.searchsorted(cum_imp, 0.85 * cum_imp[-1]) + 1
    optimal_lags = int(np.clip(cutoff_idx, min_lags, max_lags))

    print(f"[RF Lag Optimization] Selected optimal lookback sequence length = {optimal_lags} (from max {max_lags})")
    return optimal_lags, imp_series


if __name__ == "__main__":
    run_feature_selection()

