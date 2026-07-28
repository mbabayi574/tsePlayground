import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

SYMBOLS = [
    "Foolad",
    "Khodro",
    "Shasta",
    "Vbmelat",
    "Fars",
    "Zob",
    "Kegel",
    "Shabdar",
    "Vasandogh",
    "Simorgh",
    "Faofogh"
]

NON_FEATURE_COLS = {
    "date", "timestamp", "open", "high", "low", "close",
    "adjClose", "value", "volume", "count", "yesterday",
    "future_return", "signal", "symbol",
}


def discover_features(df):
    """
    Automatically discover feature columns:
    everything numeric that is NOT in the exclude list.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in NON_FEATURE_COLS]


@st.cache_data
def load_data():
    dfs = []
    for symbol in SYMBOLS:
        path = os.path.join(BASE_DIR, "data", "processed", f"{symbol}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["symbol"] = symbol
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


@st.cache_resource
def train_model(df):
    features = discover_features(df)
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["signal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    class_ratio = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0)
    model = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=class_ratio
    )
    model.fit(X_train, y_train)
    return model, features


st.title("📈 Iran Stock Signal Detector")
st.caption("ML-based buy signal detection for Tehran Stock Exchange")

df = load_data()
if df.empty:
    st.error("No processed data found. Please run features.py first.")
    st.stop()

model, features = train_model(df)

# Latest Signals
st.subheader("🔔 Latest Signals")
latest = []
for symbol in SYMBOLS:
    path = os.path.join(BASE_DIR, "data", "processed", f"{symbol}.csv")
    if os.path.exists(path):
        sdf = pd.read_csv(path)
        if not sdf.empty:
            last_row = sdf[features].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)
            prob = model.predict_proba(last_row)[0][1]
            latest.append({"Symbol": symbol, "Buy Probability": round(float(prob), 2)})

signals_df = pd.DataFrame(latest).sort_values("Buy Probability", ascending=False)
st.dataframe(signals_df)

# Price Chart
st.subheader("📊 Price Chart")
selected = st.selectbox("Select Stock", SYMBOLS)
selected_path = os.path.join(BASE_DIR, "data", "processed", f"{selected}.csv")

if os.path.exists(selected_path):
    sdf = pd.read_csv(selected_path)

    fig, ax = plt.subplots(figsize=(12, 4))
    if "close" in sdf.columns:
        ax.plot(sdf["close"].values, label="Close Price")
    ax.set_title(f"{selected} - Close Price")
    ax.legend()
    st.pyplot(fig)

    # RSI
    st.subheader("📉 RSI")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    rsi_col = "rsi_14" if "rsi_14" in sdf.columns else ("rsi" if "rsi" in sdf.columns else None)
    if rsi_col and rsi_col in sdf.columns:
        ax2.plot(sdf[rsi_col].values, color="orange", label=f"RSI ({rsi_col})")
        ax2.axhline(70, color="red", linestyle="--")
        ax2.axhline(30, color="green", linestyle="--")
        ax2.legend()
    st.pyplot(fig2)