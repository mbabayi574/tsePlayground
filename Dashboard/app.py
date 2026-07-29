import json
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
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

# ── Signal class definitions ───────────────────────────────────────────
SIGNAL_LABELS = {0: "Strong Sell", 1: "Sell", 2: "Neutral", 3: "Buy", 4: "Strong Buy"}
SIGNAL_COLORS = {
    "Strong Sell": "#d32f2f",
    "Sell":        "#ef5350",
    "Neutral":     "#78909c",
    "Buy":         "#66bb6a",
    "Strong Buy":  "#2e7d32",
}
SIGNAL_ICONS = {
    "Strong Sell": ":material/trending_down:",
    "Sell":        ":material/arrow_downward:",
    "Neutral":     ":material/remove:",
    "Buy":         ":material/arrow_upward:",
    "Strong Buy":  ":material/trending_up:",
}
NUM_CLASSES = len(SIGNAL_LABELS)

NON_FEATURE_COLS = {
    "date", "timestamp", "open", "high", "low", "close",
    "adjClose", "value", "volume", "count", "yesterday",
    "future_return", "signal", "signal_class", "symbol",
}


def discover_features(df):
    """
    Automatically discover feature columns:
    everything numeric that is NOT in the exclude list.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in NON_FEATURE_COLS]


def load_selected_features(df):
    """
    Load curated feature list from feature_selection.py output.
    Falls back to discover_features() if the file doesn't exist.
    """
    path = os.path.join(BASE_DIR, "data", "selected_features.json")
    if os.path.exists(path):
        with open(path) as f:
            selected = json.load(f)
        available = set(df.select_dtypes(include=[np.number]).columns)
        valid = [f for f in selected if f in available]
        if len(valid) >= 10:
            return valid, "curated"
    return discover_features(df), "auto"


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
    """Train a 5-class XGBoost model on pooled data."""
    features, feature_mode = load_selected_features(df)
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["signal_class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=10,
        gamma=2,
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model, features, feature_mode


# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSE Signal Detector",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)

st.title(":material/candlestick_chart: Iran stock signal detector")
st.caption("Multi-class ML signal prediction for Tehran Stock Exchange")

df = load_data()

if df.empty:
    st.error("No processed data found. Please run features.py first.")
    st.stop()

if "signal_class" not in df.columns:
    st.error(
        "Column `signal_class` not found in processed data. "
        "Re-run `features.py` to generate the multi-class target."
    )
    st.stop()

# Drop NaN signal_class rows before training
df = df.dropna(subset=["signal_class"]).copy()
df["signal_class"] = df["signal_class"].astype(int)

model, features, feature_mode = train_model(df)

# Show feature mode in sidebar
with st.sidebar:
    st.markdown("### :material/tune: Model info")
    mode_label = "Curated (SHAP+MI+RFE)" if feature_mode == "curated" else "Auto-discovery"
    st.metric("Feature mode", mode_label)
    st.metric("Feature count", len(features))
    with st.expander("Feature list"):
        st.write(features)


@st.cache_data
def load_macro_summary():
    macro_info = [
        ("USD_IRR", "USD / IRR", "Rial"),
        ("TSE_Index", "TSE Index", "Pts"),
        ("Gold_USD", "Gold / USD", "$"),
        ("Silver_USD", "Silver / USD", "$"),
        ("Oil_USD", "Oil / USD", "$"),
        ("BTC_USD", "BTC / USD", "$"),
    ]
    summary = []
    for key, label, unit in macro_info:
        path = os.path.join(BASE_DIR, "data", "raw", "macro", f"{key}.csv")
        if os.path.exists(path):
            mdf = pd.read_csv(path)
            if len(mdf) >= 2:
                last_val = mdf["close"].iloc[-1]
                prev_val = mdf["close"].iloc[-2]
                chg_pct = (last_val - prev_val) / prev_val * 100
                spark = mdf["close"].tail(20).tolist()
                summary.append({
                    "Key": key,
                    "Label": label,
                    "Value": last_val,
                    "Change %": chg_pct,
                    "Unit": unit,
                    "Spark": spark,
                })
    return summary


# ── Macro Market Context ───────────────────────────────────────────────
st.subheader(":material/globe: Macro market indicators")
macro_summary = load_macro_summary()
if macro_summary:
    with st.container(horizontal=True):
        for item in macro_summary:
            fmt_val = f"{item['Value']:,.0f}" if item['Value'] > 100 else f"{item['Value']:,.2f}"
            st.metric(
                label=item["Label"],
                value=f"{fmt_val} {item['Unit']}",
                delta=f"{item['Change %']:+.2f}%",
                border=True,
                chart_data=item["Spark"],
                chart_type="line",
            )

# ── Latest Signals ─────────────────────────────────────────────────────
st.subheader(":material/notifications_active: Latest signals")


latest = []
for symbol in SYMBOLS:
    path = os.path.join(BASE_DIR, "data", "processed", f"{symbol}.csv")
    if os.path.exists(path):
        sdf = pd.read_csv(path)
        if not sdf.empty:
            last_row = sdf[features].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)
            proba = model.predict_proba(last_row)[0]
            pred_class = int(proba.argmax())
            confidence = float(proba[pred_class])
            latest.append({
                "Symbol": symbol,
                "Signal": SIGNAL_LABELS[pred_class],
                "Confidence": round(confidence, 2),
                "Strong Sell %": round(float(proba[0]) * 100, 1),
                "Sell %": round(float(proba[1]) * 100, 1),
                "Neutral %": round(float(proba[2]) * 100, 1),
                "Buy %": round(float(proba[3]) * 100, 1),
                "Strong Buy %": round(float(proba[4]) * 100, 1),
            })

signals_df = pd.DataFrame(latest).sort_values("Confidence", ascending=False)

# Show signal KPIs
buy_count = signals_df["Signal"].isin(["Buy", "Strong Buy"]).sum()
sell_count = signals_df["Signal"].isin(["Sell", "Strong Sell"]).sum()
neutral_count = (signals_df["Signal"] == "Neutral").sum()

with st.container(horizontal=True):
    st.metric(":material/trending_up: Buy signals", buy_count, border=True)
    st.metric(":material/remove: Neutral", neutral_count, border=True)
    st.metric(":material/trending_down: Sell signals", sell_count, border=True)

# Signals table
st.dataframe(
    signals_df,
    column_config={
        "Confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0, max_value=1, format="%.0%%",
        ),
    },
    hide_index=True,
)

# ── Per-Stock Detail ───────────────────────────────────────────────────
st.subheader(":material/bar_chart: Stock detail")
selected = st.selectbox("Select stock", SYMBOLS, label_visibility="collapsed")
selected_path = os.path.join(BASE_DIR, "data", "processed", f"{selected}.csv")

if os.path.exists(selected_path):
    sdf = pd.read_csv(selected_path)

    # Price chart
    if "close" in sdf.columns:
        with st.container(border=True):
            st.markdown(f"**{selected} — Close price**")
            chart_data = pd.DataFrame({"Day": range(len(sdf)), "Close": sdf["close"].values})
            st.line_chart(chart_data, x="Day", y="Close")

    # Probability breakdown for latest row
    if not sdf.empty:
        last_row = sdf[features].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0)
        proba = model.predict_proba(last_row)[0]
        pred_class = int(proba.argmax())

        with st.container(border=True):
            st.markdown(f"**Latest prediction: {SIGNAL_ICONS[SIGNAL_LABELS[pred_class]]} {SIGNAL_LABELS[pred_class]}**")

            prob_df = pd.DataFrame({
                "Signal": [SIGNAL_LABELS[i] for i in range(NUM_CLASSES)],
                "Probability": [float(proba[i]) for i in range(NUM_CLASSES)],
                "Color": [SIGNAL_COLORS[SIGNAL_LABELS[i]] for i in range(NUM_CLASSES)],
            })

            chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("Signal:N", sort=list(SIGNAL_LABELS.values()), title=None),
                    y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
                    color=alt.Color(
                        "Signal:N",
                        scale=alt.Scale(
                            domain=list(SIGNAL_COLORS.keys()),
                            range=list(SIGNAL_COLORS.values()),
                        ),
                        legend=None,
                    ),
                    tooltip=["Signal", alt.Tooltip("Probability:Q", format=".1%")],
                )
                .properties(height=260)
            )
            st.altair_chart(chart)

    # RSI
    rsi_col = "rsi_14" if "rsi_14" in sdf.columns else ("rsi" if "rsi" in sdf.columns else None)
    if rsi_col:
        with st.container(border=True):
            st.markdown("**RSI (14)**")
            rsi_data = pd.DataFrame({"Day": range(len(sdf)), "RSI": sdf[rsi_col].values})
            st.line_chart(rsi_data, x="Day", y="RSI")