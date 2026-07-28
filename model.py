"""
Enhanced classifier for TSE buy-signal prediction.

Key improvements over the baseline:
  1. Purged time-aware split per symbol (gap to prevent target leakage).
  2. Automated feature-list discovery from processed CSVs.
  3. Robust pre-processing (inf/NaN handling, winsorisation).
  4. Hyperparameter-tuned XGBoost + LightGBM with early stopping.
  5. Exponential sample weighting (recency bias).
  6. Precision-recall-aware threshold tuning.
  7. Purged walk-forward cross-validation.
  8. Feature importance and per-symbol analysis.
"""

import pandas as pd
import numpy as np
import os
import warnings

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from symbols import SYMBOL_NAMES

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Columns that must NOT be used as features ──────────────────────────
NON_FEATURE_COLS = {
    "date", "timestamp", "open", "high", "low", "close",
    "adjClose", "value", "volume", "count", "yesterday",
    "future_return", "signal", "symbol",
}


# ── Data Loading ───────────────────────────────────────────────────────

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
    """
    Automatically discover feature columns:
    everything numeric that is NOT in the exclude list.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in NON_FEATURE_COLS]
    return features


# ── Purged Time-Aware Split ────────────────────────────────────────────

def purged_time_split(df, test_ratio=0.20, purge_days=10):
    """
    Split each symbol's data chronologically with a purge gap.
    The gap prevents the target's forward-looking window from
    bleeding information across the split boundary.
    """
    train_dfs, test_dfs = [], []
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        split_idx = int(n * (1 - test_ratio))
        train_end = max(0, split_idx - purge_days)
        test_start = split_idx
        train_dfs.append(grp.iloc[:train_end])
        test_dfs.append(grp.iloc[test_start:])
    train = pd.concat(train_dfs, ignore_index=True)
    test  = pd.concat(test_dfs, ignore_index=True)
    return train, test


# ── Pre-processing ─────────────────────────────────────────────────────

def clean_Xy(df, features):
    """Replace inf, winsorise extreme outliers, fill NaN."""
    X = df[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X.columns:
        lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
        if lo != hi:
            X[col] = X[col].clip(lo, hi)
    X.fillna(0, inplace=True)
    y = df["signal"].astype(int)
    return X, y


def compute_sample_weights(df, half_life_days=500):
    """
    Exponential decay weights favouring recent observations.
    Each symbol's data is weighted independently so that a symbol
    with fewer rows doesn't get dominated.
    """
    weights = np.ones(len(df))
    if "date" not in df.columns:
        return weights
    for symbol, grp in df.groupby("symbol"):
        idx = grp.index
        n = len(grp)
        # Position 0 = oldest, n-1 = newest
        positions = np.arange(n, dtype=float)
        decay = np.exp(-np.log(2) * (n - 1 - positions) / half_life_days)
        weights[idx] = decay
    return weights


# ── Threshold Tuning ───────────────────────────────────────────────────

def find_best_threshold(y_true, y_proba, min_precision=0.35):
    """
    Walk the precision-recall curve and pick the threshold that
    maximises F1 while keeping precision ≥ min_precision.
    Falls back to best-F1 threshold if nothing satisfies the constraint.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    # Constrained search
    mask = precisions[:-1] >= min_precision
    if mask.any():
        constrained_f1 = f1_scores[:-1].copy()
        constrained_f1[~mask] = -1
        best_idx = np.argmax(constrained_f1)
        return thresholds[best_idx]

    # Fallback: best F1 overall
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx]


# ── Walk-Forward Cross-Validation ──────────────────────────────────────

def walk_forward_cv(df, features, n_splits=4, purge_days=10):
    """
    Expanding-window walk-forward validation per symbol.
    """
    results = []
    symbols = df["symbol"].unique()

    for fold in range(1, n_splits + 1):
        train_dfs, test_dfs = [], []
        frac_end_train = fold / (n_splits + 1)
        frac_end_test  = (fold + 1) / (n_splits + 1)

        for symbol in symbols:
            grp = df[df["symbol"] == symbol].sort_values("date").reset_index(drop=True)
            n = len(grp)
            tr_end = int(n * frac_end_train)
            te_start = min(tr_end + purge_days, n)
            te_end = int(n * frac_end_test)
            if te_end <= te_start or tr_end < 50:
                continue
            train_dfs.append(grp.iloc[:tr_end])
            test_dfs.append(grp.iloc[te_start:te_end])

        if not train_dfs or not test_dfs:
            continue

        train = pd.concat(train_dfs, ignore_index=True)
        test  = pd.concat(test_dfs, ignore_index=True)

        X_tr, y_tr = clean_Xy(train, features)
        X_te, y_te = clean_Xy(test, features)

        if y_te.nunique() < 2 or len(X_tr) < 100:
            continue

        cr = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1.0)
        sw = compute_sample_weights(train)

        model = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
            gamma=2, reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=cr, eval_metric="aucpr",
            random_state=42, verbosity=0, tree_method="hist",
        )
        model.fit(X_tr, y_tr, sample_weight=sw,
                  eval_set=[(X_te, y_te)], verbose=False)
        proba = model.predict_proba(X_te)[:, 1]

        thr = find_best_threshold(y_te, proba, min_precision=0.35)
        preds = (proba >= thr).astype(int)

        p = precision_score(y_te, preds, zero_division=0)
        r = recall_score(y_te, preds, zero_division=0)
        f = f1_score(y_te, preds, zero_division=0)
        a = roc_auc_score(y_te, proba)
        ap = average_precision_score(y_te, proba)

        results.append({
            "fold": fold, "precision": p, "recall": r,
            "f1": f, "roc_auc": a, "pr_auc": ap, "threshold": thr,
            "train_size": len(X_tr), "test_size": len(X_te),
        })
        print(f"  Fold {fold}: Prec={p:.3f}  Rec={r:.3f}  F1={f:.3f}  "
              f"AUC={a:.3f}  PR-AUC={ap:.3f}  thr={thr:.3f}  "
              f"(train={len(X_tr):,}  test={len(X_te):,})")

    if results:
        rdf = pd.DataFrame(results)
        print(f"\n  Mean:   Prec={rdf['precision'].mean():.3f}  "
              f"Rec={rdf['recall'].mean():.3f}  "
              f"F1={rdf['f1'].mean():.3f}  "
              f"AUC={rdf['roc_auc'].mean():.3f}  "
              f"PR-AUC={rdf['pr_auc'].mean():.3f}")
    return results


# ── Training Pipeline ──────────────────────────────────────────────────

def train_models(df):
    features = discover_features(df)
    print(f"Feature count: {len(features)}")

    train_df, test_df = purged_time_split(df, purge_days=10)
    X_train, y_train = clean_Xy(train_df, features)
    X_test, y_test   = clean_Xy(test_df, features)
    sample_weights   = compute_sample_weights(train_df)

    class_ratio = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
    print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"Class ratio (neg/pos): {class_ratio:.1f}")
    print(f"Train signal rate: {y_train.mean():.3f}  |  "
          f"Test signal rate: {y_test.mean():.3f}")

    models = {}

    # ── XGBoost ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("XGBoost (regularised + sample-weighted)")
    print("=" * 60)
    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=10,
        gamma=2,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=class_ratio,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        verbosity=0,
        tree_method="hist",
    )
    xgb.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    _report("XGBoost", y_test, xgb_proba)
    models["xgb"] = xgb

    # ── LightGBM ───────────────────────────────────────────────────────
    if HAS_LGBM:
        print("\n" + "=" * 60)
        print("LightGBM (regularised + sample-weighted)")
        print("=" * 60)
        lgbm = LGBMClassifier(
            n_estimators=800,
            max_depth=5,
            num_leaves=20,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_samples=30,
            reg_alpha=0.5,
            reg_lambda=2.0,
            scale_pos_weight=class_ratio,
            random_state=42,
            verbosity=-1,
        )
        lgbm.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_X=X_test, eval_y=y_test,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
        _report("LightGBM", y_test, lgbm_proba)
        models["lgbm"] = lgbm

        # ── Ensemble ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Ensemble (XGBoost + LightGBM average)")
        print("=" * 60)
        ens_proba = 0.5 * xgb_proba + 0.5 * lgbm_proba
        _report("Ensemble", y_test, ens_proba)

    # ── Feature Importance ─────────────────────────────────────────────
    print("\n── Top 25 Features (XGBoost gain) " + "─" * 30)
    imp = pd.Series(xgb.feature_importances_, index=features)
    imp = imp.sort_values(ascending=False).head(25)
    max_imp = imp.max() if imp.max() > 0 else 1
    for feat, score in imp.items():
        bar = "█" * int(score / max_imp * 30)
        print(f"  {feat:<30s}  {score:7.4f}  {bar}")

    # ── Walk-Forward Cross-Validation ──────────────────────────────────
    print("\n── Walk-Forward Cross-Validation (4 folds) " + "─" * 18)
    walk_forward_cv(df, features, n_splits=4, purge_days=10)

    # ── Per-Symbol Breakdown (test set) ────────────────────────────────
    if "symbol" in test_df.columns:
        print("\n── Per-Symbol Test Results (XGBoost, tuned threshold) " + "─" * 8)
        thr = find_best_threshold(y_test, xgb_proba, min_precision=0.35)
        xgb_preds = (xgb_proba >= thr).astype(int)
        test_df = test_df.copy()
        test_df["pred"] = xgb_preds
        test_df["proba"] = xgb_proba
        test_df["y"] = y_test.values

        rows = []
        for sym, grp in test_df.groupby("symbol"):
            if len(grp) < 10:
                continue
            n = len(grp)
            pos = grp["y"].sum()
            pred_pos = grp["pred"].sum()
            if pred_pos > 0 and pos > 0:
                p = precision_score(grp["y"], grp["pred"], zero_division=0)
                r = recall_score(grp["y"], grp["pred"], zero_division=0)
                f = f1_score(grp["y"], grp["pred"], zero_division=0)
            else:
                p = r = f = 0.0
            rows.append({"symbol": sym, "rows": n, "actual_pos": pos,
                         "pred_pos": pred_pos, "prec": p, "rec": r, "f1": f})

        sdf = pd.DataFrame(rows).sort_values("f1", ascending=False)
        for _, row in sdf.head(15).iterrows():
            print(f"  {row['symbol']:<16s}  rows={row['rows']:4d}  "
                  f"pos={row['actual_pos']:3.0f}  pred={row['pred_pos']:3.0f}  "
                  f"P={row['prec']:.2f}  R={row['rec']:.2f}  F1={row['f1']:.2f}")

    return models


def _report(name, y_true, y_proba):
    """Print classification metrics at default and tuned thresholds."""
    auc = roc_auc_score(y_true, y_proba) if y_true.nunique() > 1 else 0
    ap  = average_precision_score(y_true, y_proba) if y_true.nunique() > 1 else 0

    # Default threshold
    preds_default = (y_proba >= 0.5).astype(int)
    print(f"\n  [{name}] @ threshold=0.50:")
    print(classification_report(y_true, preds_default, digits=3, zero_division=0))
    print(f"  ROC-AUC: {auc:.3f}  |  PR-AUC: {ap:.3f}")

    # Tuned threshold targeting precision ≥ 0.35 while maximising F1
    thr = find_best_threshold(y_true, y_proba, min_precision=0.35)
    preds_tuned = (y_proba >= thr).astype(int)
    p = precision_score(y_true, preds_tuned, zero_division=0)
    r = recall_score(y_true, preds_tuned, zero_division=0)
    f = f1_score(y_true, preds_tuned, zero_division=0)
    print(f"\n  [{name}] @ tuned threshold={thr:.3f}  (target prec≥0.35):")
    print(classification_report(y_true, preds_tuned, digits=3, zero_division=0))
    print(f"  Precision: {p:.3f}  |  Recall: {r:.3f}  |  F1: {f:.3f}")


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_all_data()
    print(f"Total rows: {len(df):,}")
    models = train_models(df)
