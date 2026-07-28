# AGENTS.md

Instruction guide for AI agents working in `tsePlayground`.

## Environment & Commands

Always run Python commands using the workspace virtual environment at `.venv/bin/python`.

- **Download Raw Data**: `.venv/bin/python scraper.py` (Fetches daily TSE data via `pytse-client` into `data/raw/`)
- **Feature Engineering**: `.venv/bin/python features.py` (Transforms `data/raw/*.csv` into `data/processed/*.csv`)
- **Train Classifier Models**: `.venv/bin/python model.py` (Trains Random Forest and XGBoost baseline classifiers)
- **Launch Streamlit App**: `.venv/bin/streamlit run Dashboard/app.py`

## Architecture & Data Flow

1. **Symbol Mapping (`symbols.py`)**: Maps English identifiers (e.g. `Atlas`, `Foolad`) to Persian TSE symbol names (`اطلس`, `فولاد`).
2. **Scraper (`scraper.py`)**: Downloads CSVs using Persian symbol names into `data/raw/` and renames them to English keys (`data/raw/{EngName}.csv`).
3. **Feature Engineering (`features.py`)**: Reads `data/raw/`, adds 15 technical indicators (RSI, MACD, Bollinger Bands, SMAs, volume ratio, momentum) via `ta`, builds binary target `signal` (1 if 5-day return > 5%), and writes to `data/processed/`.
4. **Dashboard (`Dashboard/app.py`)**: Note that `app.py` maintains a hardcoded subset list of 11 core symbols (`SYMBOLS`) rather than reading all symbols from `symbols.py`. Trains XGBoost model on launch and predicts buy probability for the latest trading day.
5. **Kronos Transformer Model (`model/`)**: PyTorch time-series foundation model implementation (`kronos.py`, `module.py`) utilizing Binary Spherical Quantization (`BSQuantizer`). Imports rely on `sys.path.append("../")`.

## Market & Data Quirks

- **TSE Trading Calendar**: Tehran Stock Exchange operates **Saturday through Wednesday**. Business day calculations (`common.py`) use `weekmask="Sat Sun Mon Tue Wed"`.
- **Session Data Sanitization**: `common.py` handles TSE market anomalies by replacing invalid `open == 0` values with yesterday's close and dropping suspended trading sessions (`volume == 0` or `amount == 0`).
