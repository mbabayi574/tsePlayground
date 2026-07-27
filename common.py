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

