import pytse_client as tse
import yfinance as yf
import os
import pandas as pd

from symbols import SYMBOLS, MACRO_SYMBOLS

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/raw/macro", exist_ok=True)


def download_macro_data():
    print("\n--- Downloading Macro Indicators ---")
    for name, info in MACRO_SYMBOLS.items():
        source = info["source"]
        ticker = info["ticker"]
        display_name = info["name"]
        csv_path = f"data/raw/macro/{name}.csv"
        print(f"Downloading macro indicator {display_name} ({name}) via {source}...")
        try:
            if source == "pytse_client":
                res = tse.download_financial_indexes(symbols=[ticker], write_to_csv=False)
                if ticker in res:
                    df = res[ticker]
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
                    df.to_csv(csv_path, index=False)
                    print(f"[OK] Saved {display_name} -> {csv_path} ({len(df)} rows)")
            elif source == "yfinance":
                yf_df = yf.Ticker(ticker).history(period="max")
                if not yf_df.empty:
                    yf_df = yf_df.reset_index()
                    yf_df["date"] = pd.to_datetime(yf_df["Date"]).dt.strftime("%Y-%m-%d")
                    yf_df = yf_df.rename(columns={
                        "Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"
                    })
                    yf_df = yf_df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
                    yf_df[["date", "close", "open", "high", "low", "volume"]].to_csv(csv_path, index=False)
                    print(f"[OK] Saved {display_name} -> {csv_path} ({len(yf_df)} rows)")
        except Exception as e:
            print(f"[ERROR] Failed downloading {display_name}: {e}")


def download_stock_data():
    print("\n--- Downloading Stock Data ---")
    for eng_name, fa_symbol in SYMBOLS.items():
        print(f"Downloading {eng_name}...")
        try:
            tse.download(
                symbols=fa_symbol,
                write_to_csv=True,
                base_path="data/raw"
            )
            fa_path = f"data/raw/{fa_symbol}.csv"
            eng_path = f"data/raw/{eng_name}.csv"
            if os.path.exists(fa_path):
                os.rename(fa_path, eng_path)
            print(f"[OK] {eng_name} saved")
        except Exception as e:
            print(f"[ERROR] {eng_name}: {e}")


if __name__ == "__main__":
    download_macro_data()
    download_stock_data()