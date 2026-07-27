import pandas as pd
import matplotlib.pyplot as plt
from symbols import SYMBOL_NAMES

from model import Kronos, KronosTokenizer, KronosPredictor

from common import load_stock, preprocess_tsetmc_data

T_Value = 0.7606669382978648
top_p_Value = 0.79024065770806
sample_count_value = 27


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
        pred_len= 5,
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