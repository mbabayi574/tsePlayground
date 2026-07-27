import pandas as pd
import optuna
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error


from model import Kronos, KronosTokenizer, KronosPredictor


# Configurable Parameters
pred_len = 5 # Hardcoded for a full trading week (Saturday to Wednesday)
lookback = 448

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


def generate_ensemble_forecast(df, predictor, study, top_k=4, pred_len=5):
    """
    Generates an ensemble forecast by averaging predictions from the top K Optuna trials.
    """
    print(f"\n--- Generating Ensemble Forecast ({top_k} Models) ---")

    # 1. Get completed trials and sort them by RMSE (lowest to highest)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value)

    # Select the top_k trials (e.g., Predictions A, B, C, D)
    top_trials = completed_trials[:top_k]
    all_preds = []

    # 2. Generate a prediction for each of the top parameter settings
    for i, trial in enumerate(top_trials):
        print(f"Running Model {i+1}/{top_k} (Trial {trial.number}) | RMSE: {trial.value:.4f}")
        params = trial.params

        # Preprocess the entire dataset up to the present using this trial's lookback
        x_df, x_timestamp, y_timestamp = preprocess_tsetmc_data(
            df,
            lookback=params['lookback'],
            pred_len=pred_len
        )

        # Predict
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=pd.Series(x_timestamp),
            y_timestamp=pd.Series(y_timestamp),
            pred_len=pred_len,
            T=params['T'],
            top_p=params['top_p'],
            sample_count=params['sample_count']
        )
        all_preds.append(pred_df)

    # 3. Average the predictions to improve robustness
    ensemble_pred = all_preds[0].copy()
    numeric_cols = ensemble_pred.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Stack all values for this column across the K predictions and compute the mean
        stacked_values = np.column_stack([p[col] for p in all_preds])
        ensemble_pred[col] = stacked_values.mean(axis=1)

    print("\n--- Ensemble Forecast Complete ---")
    return ensemble_pred, top_trials

df = load_stock("Faofogh")

tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")

model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

# Initialize the predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)

# Define the Objective Function for Bayesian Optimization
def objective(trial):
    # 1. Set sequence parameters
    pred_len = 5 # Hardcoded for a full trading week

    # Note: Kronos Predictor was initialized with max_context=512.
    # lookback + pred_len should not exceed this context window.
    lookback = trial.suggest_int('lookback', 128, 500, step=32)

    # 2. Suggest sampling parameters
    T = trial.suggest_float('T', 0.50, 0.80)
    top_p = trial.suggest_float('top_p', 0.60, 0.95)
    sample_count = trial.suggest_int('sample_count', 10, 50)

    # 3. Split data based on the fixed pred_len
    df_train_raw = df.iloc[:-pred_len].copy()
    df_val_raw = df.iloc[-pred_len:].copy()

    try:
        # 4. Preprocess data dynamically using this trial's lookback and the fixed pred_len
        x_df_train, x_timestamp_train, y_timestamp_val = preprocess_tsetmc_data(
            df_train_raw,
            lookback=lookback,
            pred_len=pred_len
        )

        true_future_close = pd.to_numeric(df_val_raw['close'], errors='coerce').values

        # 5. Generate predictions
        pred_df = predictor.predict(
            df=x_df_train,
            x_timestamp=pd.Series(x_timestamp_train),
            y_timestamp=pd.Series(y_timestamp_val),
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count
        )

        # 6. Calculate Errors
        mae_score = mean_absolute_error(true_future_close, pred_df['close'])
        rmse_score = np.sqrt(mean_squared_error(true_future_close, pred_df['close']))

        # Log the MAE as a custom attribute
        trial.set_user_attr("MAE", mae_score)

        return rmse_score

    except Exception as e:
        # If the model fails (e.g., lookback exceeds available active days), prune the trial
        print(f"Error testing parameters: {e}")
        raise optuna.TrialPruned()

# Run the Bayesian Optimization
print("--- Starting Bayesian Optimization ---")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("\n--- Optimization Complete ---")
best_trial = study.best_trial

print("Best Parameters:")
for key, value in best_trial.params.items():
    print(f"  {key}: {value}")

print(f"\nLowest RMSE: {best_trial.value:.4f}")
print(f"Associated MAE: {best_trial.user_attrs['MAE']:.4f}")

# Generate the ensemble prediction using the top 4 models (A, B, C, D)
ensemble_pred_df, top_used_trials = generate_ensemble_forecast(
    df=df,
    predictor=predictor,
    study=study,
    top_k=4,
    pred_len=pred_len
)

print("\nEnsemble Forecasted Data Head:")
print(ensemble_pred_df.head())

# Visualize the new ensemble prediction
# (Using the base dataset sliced by the best trial's lookback for visualization continuity)
best_lookback = top_used_trials[0].params['lookback']
hist_slice = df.tail(best_lookback).copy()

# Map column names back to match the plotting function expectations
column_mapping = {
    "date": "timestamp",
    "open": "open", "high": "high", "low": "low",
    "close": "close", "volume": "volume"
}
hist_slice = hist_slice.rename(columns=column_mapping)
hist_slice["close"] = pd.to_numeric(hist_slice["close"], errors="coerce")
hist_slice["volume"] = pd.to_numeric(hist_slice["volume"], errors="coerce")

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


plot_prediction(hist_slice, ensemble_pred_df)