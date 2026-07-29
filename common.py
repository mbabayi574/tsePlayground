import os
import pandas as pd

def load_stock(eng_name):
    path = f"data/raw/{eng_name}.csv"
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

# Configurable Parameters
pred_len = 5 # Hardcoded for a full trading week (Saturday to Wednesday)
lookback = 448

def preprocess_tsetmc_data(df: pd.DataFrame, lookback: int = lookback, pred_len: int = pred_len):
    """
    Cleans and prepares TSETMC CSV data for time-series forecasting.
    """
    # 1. Validation
    required_columns = {
        "date", "open", "high", "low",
        "adjClose", "value", "volume", "count", "yesterday", "close"
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Malformed CSV. Missing columns: {missing}")

    # 2. Rename columns to standard lowercase formats
    column_mapping = {
        "date": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "value": "amount", # map value to amount for compatibility
        "adjClose": "adjClose",
        "count": "count",
        "yesterday": "yesterday"
    }

    data = df.rename(columns=column_mapping).copy()

    # 3. Type Conversion
    numeric_columns = ["open", "high", "low", "close", "volume", "amount", "adjClose", "count", "yesterday"]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    # 4. Remove Invalid Rows & Duplicates
    data = data.dropna(subset=numeric_columns + ["timestamp"])
    data = data.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 5. Handle TSE-Specific Quirks
    # Replace invalid OPEN=0 with yesterday's close price
    mask_zero_open = data["open"] <= 0
    data.loc[mask_zero_open, "open"] = data.loc[mask_zero_open, "yesterday"]

    # Filter out suspended sessions (must have volume, value, and valid prices)
    active_data = data[
        (data["volume"] > 0) &
        (data["amount"] > 0) &
        (data["close"] > 0) &
        (data["high"] > 0) &
        (data["low"] > 0)
    ].reset_index(drop=True)

    if len(active_data) < lookback:
        raise ValueError(f"Only {len(active_data)} active trading days found (need {lookback}).")

    # 6. Build the Lookback History
    history = active_data.tail(lookback)

    x_df = history[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
    x_timestamp = history["timestamp"].reset_index(drop=True)

    # 7. Generate Future Dates using TSE Calendar (Saturday - Wednesday)
    iran_workdays = "Sat Sun Mon Tue Wed"
    y_timestamp = pd.bdate_range(
        start=x_timestamp.iloc[-1] + pd.Timedelta(days=1),
        periods=pred_len,
        freq="C",
        weekmask=iran_workdays
    )

    print(f"Prepared samples : {len(x_df)}")
    print(f"Date range       : {x_timestamp.iloc[0].date()} -> {x_timestamp.iloc[-1].date()}")
    print(f"Prediction days  : {len(y_timestamp)}")

    return x_df, x_timestamp, pd.Series(y_timestamp)


# ── Standardized Evaluation Metrics ───────────────────────────────────

import numpy as np

def compute_mae(y_true, y_pred):
    """Mean Absolute Error (MAE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_rmse(y_true, y_pred):
    """Root Mean Squared Error (RMSE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_mape(y_true, y_pred, epsilon=1e-8):
    """Mean Absolute Percentage Error (MAPE) in %."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def evaluate_forecast_metrics(y_true, y_pred):
    """
    Computes standardized time-series forecast evaluation metrics:
    MAE, RMSE, and MAPE.
    """
    return {
        "MAE": compute_mae(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAPE": compute_mape(y_true, y_pred),
    }


# ── Time-Series Cleaning & Preprocessing Helpers ──────────────────────

def clean_timeseries_data(data, iqr_threshold: float = 3.0):
    """
    Cleans time-series inputs:
    1. Forward-fill and backward-fill missing values.
    2. Winsorize extreme outliers using Interquartile Range (IQR) bounds.
    
    Accepts pd.Series, pd.DataFrame, or np.ndarray.
    """
    if isinstance(data, np.ndarray):
        s = pd.Series(data)
    else:
        s = data.copy()
        
    if isinstance(s, pd.DataFrame):
        # Apply to numeric columns
        for col in s.select_dtypes(include=[np.number]).columns:
            s[col] = s[col].ffill().bfill()
            q25, q75 = s[col].quantile(0.25), s[col].quantile(0.75)
            iqr = q75 - q25
            if iqr > 0:
                lower = q25 - iqr_threshold * iqr
                upper = q75 + iqr_threshold * iqr
                s[col] = s[col].clip(lower, upper)
        return s
    else:
        s = pd.Series(s).ffill().bfill()
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        iqr = q75 - q25
        if iqr > 0:
            lower = q25 - iqr_threshold * iqr
            upper = q75 + iqr_threshold * iqr
            s = s.clip(lower, upper)
        return s.values if isinstance(data, np.ndarray) else s


# ── Chronological Helper Splits ────────────────────────────────────────

def chronological_split(data, test_ratio: float = 0.20, purge_days: int = 0):
    """
    Chronologically splits a time series or dataframe into train and test sets
    with an optional purge gap to prevent temporal target leakage.
    """
    n = len(data)
    split_idx = int(n * (1 - test_ratio))
    train_end = max(0, split_idx - purge_days)
    test_start = split_idx
    
    if isinstance(data, (pd.DataFrame, pd.Series)):
        train = data.iloc[:train_end].copy()
        test = data.iloc[test_start:].copy()
    else:
        train = data[:train_end]
        test = data[test_start:]
        
    return train, test

def generate_walk_forward_splits(data, n_splits: int = 4, min_train_size: int = 100, test_size: int = 20):
    """
    Generates expanding-window walk-forward validation splits (train, test indices)
    for leakage-free time-series evaluation.
    """
    n = len(data)
    splits = []
    step = (n - min_train_size - test_size) // n_splits if n_splits > 1 else (n - min_train_size - test_size)
    step = max(1, step)
    
    for i in range(n_splits):
        train_end = min_train_size + i * step
        test_end = min(train_end + test_size, n)
        if train_end >= n or train_end >= test_end:
            break
        splits.append((list(range(0, train_end)), list(range(train_end, test_end))))
        
    return splits


