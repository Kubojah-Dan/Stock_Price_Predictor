import os
import joblib
import numpy as np
import pandas as pd
import math
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import load_and_save_yahoo
from features import add_features
from hyperopt import tune_lstm, tune_xgb

# ----------------- CONFIG -----------------
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA"]
START_DATE = "2015-01-01"  # Expanded data window for better training
SEQ_LEN = 30               # Increased sequence length
LSTM_VAL_FRAC = 0.15       # More validation for stable tuning
BO_XGB_TRIALS = 40         # More trials for residue model

_tickers_env = os.getenv("TRAIN_TICKERS", "").strip()
if _tickers_env:
    TICKERS = [t.strip().upper() for t in _tickers_env.split(",") if t.strip()]
# ------------------------------------------

def build_sequences_for_range(df_scaled, feature_cols, target_col, seq_len, start_idx, end_idx=None):
    arr = df_scaled[feature_cols].values
    targets = df_scaled[target_col].values
    X, y = [], []
    N = len(df_scaled)
    if end_idx is None:
        end_idx = N - 1
    for i in range(seq_len, N):
        if i < start_idx:
            continue
        if i > end_idx:
            break
        # X[i] uses data from [i-seq_len : i] (indices t-seq_len to t-1)
        # y[i] is the target we want to predict. 
        # Since df[target_col] is Close.shift(-1), target[i-1] is Close at index i.
        # This matches sequence ending at t-1 predicting price at t.
        X.append(arr[i-seq_len:i])
        y.append(targets[i-1])
    if len(X) == 0:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,))
    return np.array(X), np.array(y)

def time_train_val_split(X, y, val_frac=0.1, min_val=32):
    n = len(X)
    if n < 2:
        return X, y, X, y
    val_n = max(int(n * val_frac), min_val)
    val_n = min(max(1, val_n), n - 1)
    split = n - val_n
    return X[:split], y[:split], X[split:], y[split:]

def compute_metrics(y_true, y_pred, y_prev):
    """
    Computes regression and directional accuracy metrics.
    y_prev is the close price at t-1 (used to determine actual and predicted direction).
    """
    r2 = r2_score(y_true, y_pred) if len(y_true)>0 else float("nan")
    mae = mean_absolute_error(y_true, y_pred) if len(y_true)>0 else float("nan")
    rmse = math.sqrt(mean_squared_error(y_true, y_pred)) if len(y_true)>0 else float("nan")
    mape = mean_absolute_percentage_error(y_true, y_pred) if len(y_true)>0 else float("nan")
    
    # Directional Acc: Price at t+1 vs Price at t
    actual_dir = (y_true > y_prev).astype(int)
    pred_dir = (y_pred > y_prev).astype(int)
    
    acc = accuracy_score(actual_dir, pred_dir) if len(y_true)>0 else float("nan")
    f1 = f1_score(actual_dir, pred_dir, zero_division=0) if len(y_true)>0 else float("nan")
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape * 100,
        "R2": r2,
        "Accuracy": acc,
        "F1-score": f1
    }

for T in TICKERS:
    print(f"\n{'='*50}\nProcessing {T}\n{'='*50}")
    
    # 1) Load Data
    df_raw = load_and_save_yahoo(T, START_DATE)
    df = add_features(df_raw, ticker=T, start_date=START_DATE)
    
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index.dayofweek < 5]
        
    # target_close_t1[t] = Close[t+1]
    df["target_close_t1"] = df["Close"].shift(-1)
    df = df.dropna(subset=["target_close_t1"]).copy()
    
    exclude = {"target_price", "target_return", "target", "target_price_reg", "target_return_reg", "target_close_t1"}
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    
    print(f"Total features: {len(feature_cols)}")
    
    split_idx = int(len(df) * 0.9)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()
    
    # 2) Data Normalization
    feat_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    df_train_scaled[feature_cols] = feat_scaler.fit_transform(df_train[feature_cols])
    df_train_scaled[["target_close_t1"]] = target_scaler.fit_transform(df_train[["target_close_t1"]])
    
    df_test_scaled[feature_cols] = feat_scaler.transform(df_test[feature_cols])
    df_test_scaled[["target_close_t1"]] = target_scaler.transform(df_test[["target_close_t1"]])
    
    unscaled_close_train = df_train["Close"].values
    unscaled_close_test = df_test["Close"].values
    unscaled_target_train = df_train["target_close_t1"].values
    unscaled_target_test = df_test["target_close_t1"].values
    
    # 3) Build Sequences for LSTM
    X_lstm_train, y_lstm_train_scaled = build_sequences_for_range(
        df_train_scaled, feature_cols, "target_close_t1", SEQ_LEN, start_idx=SEQ_LEN
    )
    X_lstm_test, y_lstm_test_scaled = build_sequences_for_range(
        df_test_scaled, feature_cols, "target_close_t1", SEQ_LEN, start_idx=0
    )
    
    # Alignment: Sequence ending at t-1 predicts Close at t.
    # So the "previous close" for directional check is Close at t-1.
    close_prev_train = unscaled_close_train[SEQ_LEN-1:-1]
    close_prev_test = unscaled_close_test[SEQ_LEN-1:-1]
    target_actual_train = unscaled_target_train[SEQ_LEN-1:-1]
    target_actual_test = unscaled_target_test[SEQ_LEN-1:-1]
    
    if X_lstm_train.shape[0] < 50:
        print(f"Not enough training samples for {T}. Skipping.")
        continue
        
    print(f"LSTM Sequences: Train={X_lstm_train.shape}, Test={X_lstm_test.shape}")
    
    # 4) Train LSTM Model
    X_fit, y_fit, X_val, y_val = time_train_val_split(X_lstm_train, y_lstm_train_scaled, val_frac=LSTM_VAL_FRAC)
    
    print("\n[Step 1] Tuning & Training LSTM...")
    try:
        lstm_params = tune_lstm(X_fit, y_fit, X_val, y_val, n_trials=15)
        lstm_units = lstm_params.get("units", 64)
        lstm_lr = lstm_params.get("lr", 1e-3)
        lstm_batch = int(lstm_params.get("batch", 32))
        use_second_layer = lstm_params.get("use_second_layer", False)
        dropout_rate = lstm_params.get("dropout", 0.2)
    except Exception as e:
        print("Optuna LSTM tuning failed, fallback to defaults:", e)
        lstm_units, lstm_lr, lstm_batch, use_second_layer, dropout_rate = 64, 1e-3, 32, False, 0.2
        
    def build_custom_lstm(n_features, units, lr, dropout, second_layer):
        model = Sequential()
        model.add(Input(shape=(SEQ_LEN, n_features)))
        if second_layer:
            model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout))
            model.add(LSTM(units // 2, return_sequences=False))
        else:
            model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(max(16, units//4), activation="relu"))
        model.add(Dense(1, activation="linear"))
        model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
        return model

    lstm_model = build_custom_lstm(len(feature_cols), lstm_units, lstm_lr, dropout_rate, use_second_layer)
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    lstm_model.fit(
        X_fit, y_fit,
        epochs=150,
        batch_size=lstm_batch,
        validation_data=(X_val, y_val),
        callbacks=[es],
        verbose=0,
        shuffle=False
    )
    
    # 5) Generate LSTM Predictions & Residuals
    lstm_train_pred_scaled = lstm_model.predict(X_lstm_train, verbose=0).flatten()
    lstm_test_pred_scaled = lstm_model.predict(X_lstm_test, verbose=0).flatten()
    
    lstm_train_pred = target_scaler.inverse_transform(lstm_train_pred_scaled.reshape(-1, 1)).flatten()
    lstm_test_pred = target_scaler.inverse_transform(lstm_test_pred_scaled.reshape(-1, 1)).flatten()
    
    res_train = target_actual_train - lstm_train_pred
    
    # 6) Train Bo-XGBoost on Residuals
    print("\n[Step 2] Tuning & Training Bo-XGBoost on Residuals...")
    
    # XGBoost features: Tabular features at time t-1 PLUS the LSTM prediction for time t
    feat_train_aligned = df_train_scaled[feature_cols].values[SEQ_LEN-1:-1]
    feat_test_aligned = df_test_scaled[feature_cols].values[SEQ_LEN-1:-1]
    
    X_xgb_train = np.column_stack([feat_train_aligned, lstm_train_pred_scaled])
    X_xgb_test = np.column_stack([feat_test_aligned, lstm_test_pred_scaled])
    
    X_xgb_fit, y_xgb_fit, X_xgb_val, y_xgb_val = time_train_val_split(X_xgb_train, res_train, val_frac=0.15)
    
    try:
        xgb_params = tune_xgb(X_xgb_fit, y_xgb_fit, X_xgb_val, y_xgb_val, n_trials=BO_XGB_TRIALS)
        xgb_res_model = XGBRegressor(**xgb_params, objective="reg:squarederror", random_state=42, verbosity=0)
    except Exception as e:
        print("Optuna XGB tuning failed, fallback to defaults:", e)
        xgb_res_model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, objective="reg:squarederror", random_state=42, verbosity=0)
        
    xgb_res_model.fit(X_xgb_train, res_train)
    xgb_test_res_pred = xgb_res_model.predict(X_xgb_test)
    
    # 7) Hybrid Prediction
    hybrid_test_pred = lstm_test_pred + xgb_test_res_pred
    
    # Metrics comparison
    mets_lstm = compute_metrics(target_actual_test, lstm_test_pred, close_prev_test)
    mets_hybrid = compute_metrics(target_actual_test, hybrid_test_pred, close_prev_test)
    
    res_df = pd.DataFrame([mets_lstm, mets_hybrid], index=["LSTM Standalone", "LSTM-Bo-XGBoost Hybrid"])
    print(f"\n--- Output Results for {T} ---")
    print(res_df.to_string())
    
    # 8) Save models
    joblib.dump(feat_scaler, os.path.join(MODEL_DIR, f"scaler_feat_hybrid_{T}.pkl"))
    joblib.dump(target_scaler, os.path.join(MODEL_DIR, f"scaler_target_hybrid_{T}.pkl"))
    lstm_model.save(os.path.join(MODEL_DIR, f"lstm_hybrid_{T}.h5"))
    joblib.dump(xgb_res_model, os.path.join(MODEL_DIR, f"xgb_res_{T}.pkl"))
    
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"hybrid_metrics_{T}.csv"))

print("\nHybrid model training complete.")
