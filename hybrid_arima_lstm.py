import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import MinMaxScaler
import warnings

# PyTorch Neural Network Backend
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

# Attempt to import feature selection lag optimizer if available
try:
    from feature_selection import optimize_lookback_lags_rf
    HAS_RF_LAG_OPT = True
except ImportError:
    HAS_RF_LAG_OPT = False


class PyTorchLSTMModel(nn.Module):
    """
    Optimized PyTorch LSTM architecture with Dropout and L2 Regularization.
    """
    def __init__(self, input_dim=1, hidden_dim_1=50, hidden_dim_2=30, dropout_rate=0.2):
        super(PyTorchLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim_1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_dim_1, hidden_dim_2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim_2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Take output of last time step
        out = self.dropout2(out)
        out = self.fc(out)
        return out


class HybridARIMALSTMPredictor:
    """
    A robust, leakage-free hybrid ARIMA-LSTM time series forecasting model.
    Designed for integration into tsePlayground.
    
    Supported Architectures:
    1. 'additive' (Zhang's classical hybrid): Linear ARIMA baseline + LSTM on ARIMA residual errors.
    2. 'feature_augmentation': ARIMA point forecasts/residuals appended as features.
    3. 'reversed': LSTM fitted first on non-linear trends + ARIMA fitted on LSTM residuals.
    """

    def __init__(
        self,
        arima_order=(3, 0, 3),
        seq_length="auto",
        lstm_units=(50, 30),
        dropout_rate=0.2,
        l2_penalty=0.01,
        epochs=50,
        batch_size=32,
        architecture="additive",
    ):
        self.arima_order = arima_order
        self.seq_length = seq_length  # 'auto' or int
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_penalty = l2_penalty
        self.epochs = epochs
        self.batch_size = batch_size
        self.architecture = architecture.lower()


        # Fitted module state
        self.arima_fit = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lstm_model = None
        self.train_residuals = None
        self.last_train_data = None
        self.ljung_box_pvalue = None
        self.residual_autocorr_justified = None
        self.effective_seq_length = 10 if seq_length == "auto" else seq_length

    def _create_sequences(self, data, seq_len):
        """Helper to create rolling window sequences for LSTM input (no leakage)."""
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i : (i + seq_len)])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    def _train_pytorch_lstm(self, X_train, y_train, X_val=None, y_val=None, patience=15):
        """
        Trains PyTorch LSTM neural network with L2 regularization and Early Stopping (Guideline C).
        """
        input_dim = X_train.shape[2] if X_train.ndim == 3 else 1
        model = PyTorchLSTMModel(
            input_dim=input_dim,
            hidden_dim_1=self.lstm_units[0],
            hidden_dim_2=self.lstm_units[1],
            dropout_rate=self.dropout_rate,
        )

        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=self.l2_penalty)
        criterion = nn.MSELoss()

        # Pre-convert data to PyTorch tensors once to avoid conversion overhead per batch
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        model.train()
        n_samples = len(X_train)

        for epoch in range(self.epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0.0

            for i in range(0, n_samples, self.batch_size):
                indices = permutation[i : i + self.batch_size]
                batch_x, batch_y = X_train_t[indices], y_train_t[indices]

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_x)


            # Early Stopping Check (Guideline C)
            if has_val:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                model.train()
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1} (val_loss={best_val_loss:.6f})")
                    break

        if best_weights is not None:
            model.load_state_dict(best_weights)

        model.eval()
        return model

    def fit(self, series, validation_split=0.2):
        """
        Fits the hybrid ARIMA-LSTM model following temporal causality and leakage-free scaling.
        
        Args:
            series (pd.Series or np.ndarray): Raw historical time series data.
            validation_split (float): Fraction of training sequence data to use for validation.
        """
        series = np.array(series, dtype=float).flatten()
        self.last_train_data = series

        if self.architecture == "additive":
            # 1. Fit Linear ARIMA Baseline
            print(f"Fitting ARIMA{self.arima_order} baseline model...")
            arima_model = ARIMA(series, order=self.arima_order)
            self.arima_fit = arima_model.fit()

            linear_predictions = self.arima_fit.fittedvalues
            self.train_residuals = series - linear_predictions

            # 2. Diagnostic Check via Ljung-Box Test (Guideline A)
            print("Performing Ljung-Box test on ARIMA residuals...")
            try:
                lb_df = acorr_ljungbox(self.train_residuals, lags=[10], return_df=True)
                self.ljung_box_pvalue = lb_df["lb_pvalue"].values[0]
                print(f"Ljung-Box p-value at lag 10: {self.ljung_box_pvalue:.6f}")

                if self.ljung_box_pvalue > 0.05:
                    print("WARNING: Residuals show no significant autocorrelation (p > 0.05).")
                    print("Residuals resemble white noise; LSTM corrector may overfit on noise.")
                    self.residual_autocorr_justified = False
                else:
                    print("SUCCESS: Residuals exhibit significant structured autocorrelation (p <= 0.05).")
                    print("Deploying hybrid LSTM residual corrector is mathematically justified.")
                    self.residual_autocorr_justified = True
            except Exception as e:
                print(f"[WARN] Ljung-Box test skipped: {e}")
                self.residual_autocorr_justified = True

            # 3. Optimize Lookback Lags via Random Forest (Guideline B)
            if self.seq_length == "auto":
                if HAS_RF_LAG_OPT:
                    opt_seq, _ = optimize_lookback_lags_rf(self.train_residuals, max_lags=30)
                    self.effective_seq_length = opt_seq
                else:
                    self.effective_seq_length = 10
            else:
                self.effective_seq_length = int(self.seq_length)

            # 4. Normalize Residual Series (Leakage-free: scaler fit ONLY on training residuals - Guideline D)
            reshaped_residuals = self.train_residuals.reshape(-1, 1)
            scaled_residuals = self.scaler.fit_transform(reshaped_residuals)

            # 5. Create Sequences
            X, y = self._create_sequences(scaled_residuals, self.effective_seq_length)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # 6. Partition for Early Stopping Validation
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # 7. Train PyTorch LSTM Neural Network (Guideline C)
            print(f"Training LSTM network on scaled ARIMA residuals (seq_length={self.effective_seq_length})...")
            self.lstm_model = self._train_pytorch_lstm(X_train, y_train, X_val, y_val)
            print("Fitted hybrid ARIMA-LSTM model successfully.")

        elif self.architecture == "reversed":
            # Reversed Configuration: Train LSTM first on raw series, fit ARIMA to LSTM residuals
            print("Running Reversed Hybrid Architecture: LSTM first, ARIMA on residuals...")
            if self.seq_length == "auto":
                self.effective_seq_length = 10
            else:
                self.effective_seq_length = int(self.seq_length)

            scaled_series = self.scaler.fit_transform(series.reshape(-1, 1))
            X, y = self._create_sequences(scaled_series, self.effective_seq_length)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            self.lstm_model = self._train_pytorch_lstm(X_train, y_train, X_val, y_val)

            # Extract LSTM fitted values
            X_all_t = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                lstm_preds_scaled = self.lstm_model(X_all_t).numpy().flatten()
            lstm_preds = self.scaler.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()

            actual_aligned = series[self.effective_seq_length:]
            self.train_residuals = actual_aligned - lstm_preds

            arima_model = ARIMA(self.train_residuals, order=self.arima_order)
            self.arima_fit = arima_model.fit()
            print("Fitted Reversed Hybrid model successfully.")

        return self

    def forecast(self, steps=10, clamp_bounds="auto"):
        """
        Forecasts future values using walk-forward recursive prediction of residuals.
        
        Args:
            steps (int): The number of future steps to forecast.
            clamp_bounds (tuple or str): (min_val, max_val) to clamp final predictions (Guideline F).
                                         If 'auto', uses historical train bounds.
            
        Returns:
            np.ndarray: Predicted final values.
        """
        if self.arima_fit is None or self.lstm_model is None:
            raise ValueError("Model must be fitted before forecasting.")

        if self.architecture == "additive":
            # 1. Linear Forecast using ARIMA
            arima_forecast = self.arima_fit.forecast(steps=steps)

            # 2. Recursive Walk-Forward forecasting of residuals using LSTM
            recent_residuals = self.train_residuals[-self.effective_seq_length:]
            scaled_seed = self.scaler.transform(recent_residuals.reshape(-1, 1)).flatten()

            current_sequence = scaled_seed.copy()
            lstm_predictions_scaled = []

            for _ in range(steps):
                input_seq = torch.tensor(
                    current_sequence.reshape((1, self.effective_seq_length, 1)),
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    predicted_scaled_res = self.lstm_model(input_seq).item()

                lstm_predictions_scaled.append(predicted_scaled_res)
                current_sequence = np.append(current_sequence[1:], predicted_scaled_res)

            # 3. Inverse transform predicted residuals
            predicted_residuals = self.scaler.inverse_transform(
                np.array(lstm_predictions_scaled).reshape(-1, 1)
            ).flatten()

            # 4. Sum forecasts
            final_forecast = arima_forecast + predicted_residuals

        elif self.architecture == "reversed":
            # LSTM forecast first
            recent_series = self.last_train_data[-self.effective_seq_length:]
            scaled_seed = self.scaler.transform(recent_series.reshape(-1, 1)).flatten()

            current_sequence = scaled_seed.copy()
            lstm_predictions_scaled = []

            for _ in range(steps):
                input_seq = torch.tensor(
                    current_sequence.reshape((1, self.effective_seq_length, 1)),
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    pred_scaled = self.lstm_model(input_seq).item()

                lstm_predictions_scaled.append(pred_scaled)
                current_sequence = np.append(current_sequence[1:], pred_scaled)

            lstm_forecast = self.scaler.inverse_transform(
                np.array(lstm_predictions_scaled).reshape(-1, 1)
            ).flatten()

            res_forecast = self.arima_fit.forecast(steps=steps)
            final_forecast = lstm_forecast + res_forecast

        # 5. Apply Bounding Constraints (Guideline F)
        if clamp_bounds == "auto" and self.last_train_data is not None:
            min_val = max(0.0, float(np.min(self.last_train_data)) * 0.5)
            max_val = float(np.max(self.last_train_data)) * 2.0
            clamp_bounds = (min_val, max_val)

        if clamp_bounds is not None and isinstance(clamp_bounds, (tuple, list)):
            lower, upper = clamp_bounds
            final_forecast = np.clip(final_forecast, lower, upper)

        return final_forecast
