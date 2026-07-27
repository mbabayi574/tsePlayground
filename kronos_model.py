import os
import pandas as pd
import matplotlib.pyplot as plt
from symbols import SYMBOL_NAMES

from model import Kronos, KronosTokenizer, KronosPredictor

# Configurable Parameters
pred_len = 5 # Hardcoded for a full trading week (Saturday to Wednesday)
lookback = 448
T_Value = 0.7606669382978648
top_p_Value = 0.79024065770806
sample_count_value = 27

def load_stock(eng_name):
    path = f"data/raw/{eng_name}.csv"
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

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

def process(df):
    x_df, x_timestamp, y_timestamp = preprocess_tsetmc_data(df)

    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

    # Initialize the predictor
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # Generate predictions
    # Fix: ensure timestamps are passed as Series to avoid '.dt' AttributeError
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=pd.Series(x_timestamp),
        y_timestamp=pd.Series(y_timestamp),
        pred_len=pred_len,
        T= T_Value,          # Temperature for sampling
        top_p= top_p_Value,      # Nucleus sampling probability
        sample_count= sample_count_value  # Number of forecast paths to generate and average
    )

    kline_df = x_df.copy()
    kline_df["timestamp"] = x_timestamp

    print("Forecasted Data Head:")
    print(pred_df.head())

    def plot_prediction(hist_df, pred_df):
        # Align indices for plotting
        # The prediction starts right after the historical data
        hist_close = hist_df['close'].reset_index(drop=True)
        hist_volume = hist_df['volume'].reset_index(drop=True)

        pred_close = pred_df['close'].reset_index(drop=True)
        pred_volume = pred_df['volume'].reset_index(drop=True)

        # Offset prediction index to follow history
        pred_close.index = range(len(hist_close), len(hist_close) + len(pred_close))
        pred_volume.index = range(len(hist_volume), len(hist_volume) + len(pred_volume))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Close Price Plot
        ax1.plot(hist_close, label='Historical Close', color='blue', linewidth=1.5)
        ax1.plot(pred_close, label='Forecasted Close', color='red', linewidth=1.5, linestyle='--')
        ax1.set_ylabel('Close Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Volume Plot
        ax2.plot(hist_volume, label='Historical Volume', color='blue', linewidth=1.5)
        ax2.plot(pred_volume, label='Forecasted Volume', color='red', linewidth=1.5, linestyle='--')
        ax2.set_ylabel('Volume')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # visualize using the prepared historical slice
    plot_prediction(kline_df, pred_df)

for eng_name in SYMBOL_NAMES:
    print(f"Processing {eng_name}...")
    df = load_stock(eng_name)
    if df is None:
        continue
    try:
        process(df)
    except Exception as e:
        print(f"[ERROR] Processing {eng_name} failed: {e}")