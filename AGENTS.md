# AGENTS.md

ML-based buy signal detection and time-series forecasting pipeline for Tehran Stock Exchange (TSE) equities.

## Environment & Tooling
- **Virtual Environment**: Use `.venv/bin/python` or `.venv/bin/streamlit`.
- **Code Graph**: Repository is indexed by CodeGraph (`.codegraph/`). Reach for `codegraph_explore` MCP tool or `codegraph explore "<query>"` before grep/find when investigating symbol relationships or call hierarchies.

## Execution Order
Data pipeline components must be executed in sequence:
1. **Scraping**: `.venv/bin/python scraper.py`
   Downloads raw historical TSE data via `pytse-client` into `data/raw/{eng_name}.csv` (mapped from Persian symbols in `symbols.py`).
2. **Feature Engineering**: `.venv/bin/python features.py`
   Computes 15+ technical indicators (RSI, MACD, Bollinger Bands, SMA, momentum, volume ratios) and 5-day return targets (>5%), writing to `data/processed/{eng_name}.csv`.
3. **ML Classifier Training**: `.venv/bin/python model.py`
   Trains Random Forest and XGBoost classifiers on concatenated dataset from `data/processed/*.csv`.
4. **Kronos Deep Forecasting**: `.venv/bin/python kronos_model.py`
   Generates 5-day price/volume forecasts using Kronos foundation models (`NeoQuasar/Kronos-base` via HuggingFace).
5. **Dashboard**: `.venv/bin/streamlit run Dashboard/app.py`
   Launches interactive Streamlit app from repo root (requires `data/processed/*.csv`).

## Key Architectural & Domain Quirks
- **Import Shadowing (`model` directory vs `model.py`)**: Root script `model.py` trains tree classifiers. `kronos_model.py` imports `from model import Kronos...`, which resolves to the package directory `model/__init__.py`.
- **Persian/English Symbol Mapping**: `symbols.py` maps Persian tickers (e.g. `"فولاد"`) to English names (e.g. `"Foolad"`).
- **TSE Trading Calendar**: Iranian trading week runs **Saturday through Wednesday** (`common.py` sets `weekmask="Sat Sun Mon Tue Wed"`). Default lookback is `448` days with a `5`-day forecast horizon.
- **Data Cleaning Rules (`common.py`)**:
  - Drops zero-volume/inactive trading sessions.
  - Replaces `open == 0` values with `yesterday` close price.
  - Raises `ValueError` if active trading history is under `448` rows.
- **Dashboard Symbol Scope**: `Dashboard/app.py` hardcodes a subset of 11 major symbols for signal predictions rather than importing `SYMBOL_NAMES` from `symbols.py`.
