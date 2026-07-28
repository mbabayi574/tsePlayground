import pandas as pd
import numpy as np
import ta
import os
from symbols import SYMBOL_NAMES


def load_stock(eng_name):
    path = f"data/raw/{eng_name}.csv"
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def add_features(df):
    """
    Build a rich, scale-invariant feature set suitable for pooled
    multi-stock classification.  Every feature is either a ratio,
    percentage, oscillator, or z-score so that the model can generalise
    across symbols with vastly different price levels.
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    volume = df["volume"]
    op    = df["open"]

    # ── Momentum / Oscillators ──────────────────────────────────────────
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_7"]  = ta.momentum.RSIIndicator(close, window=7).rsi()

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close).williams_r()

    df["roc_10"] = ta.momentum.ROCIndicator(close, window=10).roc()
    df["roc_20"] = ta.momentum.ROCIndicator(close, window=20).roc()

    # ── Trend ───────────────────────────────────────────────────────────
    macd = ta.trend.MACD(close)
    df["macd_diff"] = macd.macd_diff()                # histogram (already diff)
    # Normalise MACD line by price so it's scale-invariant
    df["macd_norm"] = macd.macd() / close

    adx = ta.trend.ADXIndicator(high, low, close)
    df["adx"] = adx.adx()
    df["di_plus"]  = adx.adx_pos()
    df["di_minus"] = adx.adx_neg()
    df["di_diff"]  = df["di_plus"] - df["di_minus"]

    cci = ta.trend.CCIIndicator(high, low, close, window=20)
    df["cci"] = cci.cci()

    # Price position relative to SMAs (ratio, not raw price)
    sma_10 = ta.trend.SMAIndicator(close, window=10).sma_indicator()
    sma_20 = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    ema_12 = ta.trend.EMAIndicator(close, window=12).ema_indicator()
    ema_26 = ta.trend.EMAIndicator(close, window=26).ema_indicator()

    df["price_to_sma10"] = close / sma_10
    df["price_to_sma20"] = close / sma_20
    df["price_to_sma50"] = close / sma_50
    df["price_to_ema12"] = close / ema_12
    df["price_to_ema26"] = close / ema_26
    df["sma10_to_sma50"] = sma_10 / sma_50       # short-vs-long trend

    # ── Volatility ──────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["bb_pband"]  = bb.bollinger_pband()        # %B  (0-1 position)
    df["bb_wband"]  = bb.bollinger_wband()        # bandwidth normalised

    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    df["atr_pct"] = atr.average_true_range() / close   # ATR as % of price

    # Intraday range as % of close
    df["daily_range_pct"] = (high - low) / close

    # Historical volatility (rolling std of returns)
    returns = close.pct_change()
    df["volatility_20"] = returns.rolling(20).std()
    df["volatility_10"] = returns.rolling(10).std()
    df["volatility_5"]  = returns.rolling(5).std()

    # Volatility ratio (short-term vol vs long-term — captures regime changes)
    df["vol_ratio_5_20"] = df["volatility_5"] / df["volatility_20"].replace(0, np.nan)

    # ── Volume ──────────────────────────────────────────────────────────
    df["volume_ratio_20"] = volume / volume.rolling(20).mean()
    df["volume_ratio_5"]  = volume / volume.rolling(5).mean()

    obv_raw = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    # OBV slope (5-day % change of OBV) — scale-invariant
    df["obv_slope"] = obv_raw.pct_change(5)

    mfi = ta.volume.MFIIndicator(high, low, close, volume, window=14)
    df["mfi"] = mfi.money_flow_index()

    # Value per trade (average trade size proxy)
    if "value" in df.columns:
        df["avg_trade_value"] = df["value"] / df["count"].replace(0, np.nan)
        # Normalise by its own rolling mean
        df["avg_trade_value_ratio"] = (
            df["avg_trade_value"] / df["avg_trade_value"].rolling(20).mean()
        )
    if "count" in df.columns:
        df["trade_count_ratio"] = df["count"] / df["count"].rolling(20).mean()

    # ── Price-action / Candlestick Ratios ───────────────────────────────
    hl_range = (high - low).replace(0, np.nan)
    df["body_ratio"] = (close - op).abs() / hl_range
    df["upper_shadow"] = (high - pd.concat([close, op], axis=1).max(axis=1)) / hl_range
    df["lower_shadow"] = (pd.concat([close, op], axis=1).min(axis=1) - low) / hl_range
    df["gap_pct"] = (op - close.shift(1)) / close.shift(1)

    # ── Returns / Momentum at multiple horizons ────────────────────────
    for d in [1, 2, 3, 5, 10, 20]:
        df[f"ret_{d}d"] = close.pct_change(d)

    # ── Return z-scores (mean-reversion / breakout signals) ─────────────
    ret_20_mean = returns.rolling(20).mean()
    ret_20_std  = returns.rolling(20).std().replace(0, np.nan)
    df["ret_zscore"] = (returns - ret_20_mean) / ret_20_std

    # ── 52-week range position (scale-invariant) ────────────────────────
    high_52w = close.rolling(252, min_periods=60).max()
    low_52w  = close.rolling(252, min_periods=60).min()
    df["pct_from_52w_high"] = (close - high_52w) / high_52w
    df["pct_from_52w_low"]  = (close - low_52w) / low_52w

    # ── Streak features ────────────────────────────────────────────────
    up = (returns > 0).astype(int)
    df["up_streak"] = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
    df["up_streak"] = df["up_streak"] * up  # zero for down days
    down = (returns < 0).astype(int)
    df["down_streak"] = down.groupby((down != down.shift()).cumsum()).cumcount() + 1
    df["down_streak"] = df["down_streak"] * down

    # ── Lag features (give the model access to recent history) ──────────
    lag_cols = ["rsi_14", "macd_diff", "volume_ratio_20", "ret_1d", "adx"]
    for col in lag_cols:
        if col in df.columns:
            for lag in [1, 2, 3, 5]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def create_target(df, days=5, threshold=0.05):
    """
    Binary target: 1 if forward return over `days` exceeds `threshold`.
    Uses adjClose when available for split/dividend-adjusted returns.
    """
    price_col = "adjClose" if "adjClose" in df.columns else "close"
    df["future_return"] = df[price_col].shift(-days) / df[price_col] - 1
    df["signal"] = (df["future_return"] > threshold).astype(int)
    return df


def process_all():
    os.makedirs("data/processed", exist_ok=True)

    for eng_name in SYMBOL_NAMES:
        print(f"Processing {eng_name}...")
        df = load_stock(eng_name)
        if df is None:
            continue
        df = add_features(df)
        df = create_target(df)
        df.dropna(inplace=True)

        # Replace any remaining inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        df.to_csv(f"data/processed/{eng_name}.csv", index=False)
        print(f"[OK] {eng_name} processed — {len(df)} rows")


if __name__ == "__main__":
    process_all()