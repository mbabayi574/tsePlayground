"""
Multi-class classifier for TSE stock signal prediction.

Predicts a 5-class ordinal signal:
  0 = Strong Sell  (bottom 20 % of 5-day forward returns)
  1 = Sell          (20 – 40 %)
  2 = Neutral       (40 – 60 %)
  3 = Buy           (60 – 80 %)
  4 = Strong Buy    (top 20 %)

Key features:
  1. Purged time-aware split per symbol (gap to prevent target leakage).
  2. Automated feature-list discovery from processed CSVs.
  3. Robust pre-processing (inf/NaN handling, winsorisation).
  4. Hyperparameter-tuned XGBoost + LightGBM with early stopping.
  5. Exponential sample weighting (recency bias).
  6. Macro-averaged precision / recall / F1 and per-class reports.
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
    accuracy_score,
    confusion_matrix,
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

# ── Signal class labels ────────────────────────────────────────────────
SIGNAL_LABELS = {0: "Strong Sell", 1: "Sell", 2: "Neutral", 3: "Buy", 4: "Strong Buy"}
NUM_CLASSES = len(SIGNAL_LABELS)

# ── Columns that must NOT be used as features ──────────────────────────
NON_FEATURE_COLS = {
    "date", "timestamp", "open", "high", "low", "close",
    "adjClose", "value", "volume", "count", "yesterday",
    "future_return", "signal", "signal_class", "symbol",
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

def clean_Xy(df, features, target_col="signal_class"):
    """Replace inf, winsorise extreme outliers, fill NaN."""
    X = df[features].copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in X.columns:
        lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
        if lo != hi:
            X[col] = X[col].clip(lo, hi)
    X.fillna(0, inplace=True)
    y = df[target_col].astype(int)
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

        sw = compute_sample_weights(train)

        model = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
            gamma=2, reg_alpha=0.5, reg_lambda=2.0,
            objective="multi:softprob", num_class=NUM_CLASSES,
            eval_metric="mlogloss",
            random_state=42, verbosity=0, tree_method="hist",
        )
        model.fit(X_tr, y_tr, sample_weight=sw,
                  eval_set=[(X_te, y_te)], verbose=False)
        preds = model.predict(X_te)

        acc = accuracy_score(y_te, preds)
        p = precision_score(y_te, preds, average="macro", zero_division=0)
        r = recall_score(y_te, preds, average="macro", zero_division=0)
        f = f1_score(y_te, preds, average="macro", zero_division=0)

        results.append({
            "fold": fold, "accuracy": acc, "precision": p, "recall": r,
            "f1": f, "train_size": len(X_tr), "test_size": len(X_te),
        })
        print(f"  Fold {fold}: Acc={acc:.3f}  Prec={p:.3f}  Rec={r:.3f}  "
              f"F1={f:.3f}  (train={len(X_tr):,}  test={len(X_te):,})")

    if results:
        rdf = pd.DataFrame(results)
        print(f"\n  Mean:   Acc={rdf['accuracy'].mean():.3f}  "
              f"Prec={rdf['precision'].mean():.3f}  "
              f"Rec={rdf['recall'].mean():.3f}  "
              f"F1={rdf['f1'].mean():.3f}")
    return results


# ── Training Pipeline ──────────────────────────────────────────────────

def train_models(df):
    features = discover_features(df)
    print(f"Feature count: {len(features)}")

    # ── Validate target column ─────────────────────────────────────────
    if "signal_class" not in df.columns:
        print("[ERROR] Column 'signal_class' not found. "
              "Re-run features.py to generate the multi-class target.")
        return {}

    # Drop rows with NaN signal_class
    df = df.dropna(subset=["signal_class"]).copy()
    df["signal_class"] = df["signal_class"].astype(int)

    train_df, test_df = purged_time_split(df, purge_days=10)
    X_train, y_train = clean_Xy(train_df, features)
    X_test, y_test   = clean_Xy(test_df, features)
    sample_weights   = compute_sample_weights(train_df)

    print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"Class distribution (train):")
    for cls_id, label in SIGNAL_LABELS.items():
        count = (y_train == cls_id).sum()
        pct = count / len(y_train) * 100
        print(f"  {cls_id} ({label:12s}): {count:6,} ({pct:5.1f}%)")

    models = {}

    # ── XGBoost ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("XGBoost (multi-class, regularised + sample-weighted)")
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
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
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
    xgb_preds = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)
    _report("XGBoost", y_test, xgb_preds)
    models["xgb"] = xgb

    # ── LightGBM ───────────────────────────────────────────────────────
    if HAS_LGBM:
        print("\n" + "=" * 60)
        print("LightGBM (multi-class, regularised + sample-weighted)")
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
            objective="multiclass",
            num_class=NUM_CLASSES,
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
        lgbm_preds = lgbm.predict(X_test)
        lgbm_proba = lgbm.predict_proba(X_test)
        _report("LightGBM", y_test, lgbm_preds)
        models["lgbm"] = lgbm

        # ── Ensemble ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Ensemble (XGBoost + LightGBM average)")
        print("=" * 60)
        ens_proba = 0.5 * xgb_proba + 0.5 * lgbm_proba
        ens_preds = ens_proba.argmax(axis=1)
        _report("Ensemble", y_test, ens_preds)

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
        print("\n── Per-Symbol Test Results (XGBoost) " + "─" * 24)
        test_df = test_df.copy()
        test_df["pred"] = xgb_preds
        test_df["y"] = y_test.values

        rows = []
        for sym, grp in test_df.groupby("symbol"):
            if len(grp) < 10:
                continue
            n = len(grp)
            acc = accuracy_score(grp["y"], grp["pred"])
            f = f1_score(grp["y"], grp["pred"], average="macro", zero_division=0)
            # Distribution of predictions
            dist = {SIGNAL_LABELS[i]: (grp["pred"] == i).sum() for i in range(NUM_CLASSES)}
            rows.append({"symbol": sym, "rows": n, "accuracy": acc, "macro_f1": f, **dist})

        sdf = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
        for _, row in sdf.head(15).iterrows():
            dist_str = "  ".join(f"{SIGNAL_LABELS[i][:2]}={int(row.get(SIGNAL_LABELS[i], 0)):3d}"
                                 for i in range(NUM_CLASSES))
            print(f"  {row['symbol']:<16s}  rows={row['rows']:4d}  "
                  f"Acc={row['accuracy']:.2f}  F1={row['macro_f1']:.2f}  "
                  f"[{dist_str}]")

    return models


def _report(name, y_true, y_pred):
    """Print multi-class classification metrics."""
    target_names = [SIGNAL_LABELS[i] for i in range(NUM_CLASSES)]

    acc = accuracy_score(y_true, y_pred)
    p_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    r_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n  [{name}] Multi-class results:")
    print(classification_report(
        y_true, y_pred, target_names=target_names,
        digits=3, zero_division=0,
    ))
    print(f"  Accuracy:     {acc:.3f}")
    print(f"  Macro    P={p_macro:.3f}  R={r_macro:.3f}  F1={f_macro:.3f}")
    print(f"  Weighted F1:  {f_weighted:.3f}")

    # ── Confusion Matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    print(f"\n  Confusion Matrix:")
    header = "            " + "  ".join(f"{SIGNAL_LABELS[i][:6]:>6s}" for i in range(NUM_CLASSES))
    print(f"  {header}")
    for i in range(NUM_CLASSES):
        row = "  ".join(f"{cm[i, j]:6d}" for j in range(NUM_CLASSES))
        print(f"  {SIGNAL_LABELS[i]:<12s}{row}")


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_all_data()
    print(f"Total rows: {len(df):,}")
    models = train_models(df)
