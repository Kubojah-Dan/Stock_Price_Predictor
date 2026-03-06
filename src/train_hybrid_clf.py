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
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing import load_and_save_yahoo
from features import add_features
from hyperopt import tune_lstm_clf, tune_xgb_clf
from visualize import plot_roc_curve, plot_confusion_matrix

# ----------------- CONFIG -----------------
OUTPUT_DIR = "outputs"
MODEL_DIR = "models_clf"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TICKERS = ["AAPL", "GOOGL", "MSFT", "NVDA"]
START_DATE = "2015-01-01"
SEQ_LEN = 30
LSTM_VAL_FRAC = 0.15
BO_XGB_TRIALS = 40

_tickers_env = os.getenv("TRAIN_TICKERS", "").strip()
if _tickers_env:
    TICKERS = [t.strip().upper() for t in _tickers_env.split(",") if t.strip()]
# ------------------------------------------

def build_sequences_clf(df_scaled, feature_cols, target_col, seq_len, start_idx):
    arr = df_scaled[feature_cols].values
    targets = df_scaled[target_col].values
    X, y = [], []
    N = len(df_scaled)
    for i in range(seq_len, N):
        if i < start_idx:
            continue
        # Predict target at index i (which is direction for t+1 relative to t)
        # using data up to index i-1.
        X.append(arr[i-seq_len:i])
        y.append(targets[i-1])
    return np.array(X), np.array(y)

def time_split(X, y, val_frac=0.15):
    n = len(X)
    split = int(n * (1 - val_frac))
    return X[:split], y[:split], X[split:], y[split:]

def compute_clf_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else float("nan")
    return {
        "Accuracy": acc,
        "F1-score": f1,
        "Precision": prec,
        "Recall": rec,
        "AUC": auc
    }

for T in TICKERS:
    print(f"\n{'='*50}\nProcessing {T} (Classification Hybrid)\n{'='*50}")
    
    # 1) Load Data
    df_raw = load_and_save_yahoo(T, START_DATE)
    df = add_features(df_raw, ticker=T, start_date=START_DATE)
    
    # Target: 1 if next day's Close > today's Close
    df["target_return"] = df["Close"].pct_change().shift(-1)
    df["target_binary"] = (df["target_return"] > 0).astype(int)
    df = df.dropna(subset=["target_binary"]).copy()
    
    exclude = {"target_price", "target_return", "target", "target_binary", "target_price_reg", "target_return_reg", "target_close_t1"}
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    
    print(f"Features: {len(feature_cols)}, Samples: {len(df)}")
    
    split_idx = int(len(df) * 0.85)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()
    
    # 2) Normalization (RobustScaler for returns/indicators)
    scaler = RobustScaler()
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    df_train_scaled[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test_scaled[feature_cols] = scaler.transform(df_test[feature_cols])
    
    # 3) Sequences for LSTM
    X_lstm_train, y_lstm_train = build_sequences_clf(df_train_scaled, feature_cols, "target_binary", SEQ_LEN, SEQ_LEN)
    X_lstm_test, y_lstm_test = build_sequences_clf(df_test_scaled, feature_cols, "target_binary", SEQ_LEN, 0)
    
    # 4) Train LSTM Classifier
    print("\n[Step 1] Tuning & Training LSTM Classifier...")
    X_fit, y_fit, X_val, y_val = time_split(X_lstm_train, y_lstm_train, val_frac=LSTM_VAL_FRAC)
    
    try:
        lstm_params = tune_lstm_clf(X_fit, y_fit, X_val, y_val, n_trials=BO_XGB_TRIALS//2)
        units = lstm_params.get("units", 64)
        lr = lstm_params.get("lr", 1e-3)
        batch = int(lstm_params.get("batch", 32))
        dropout = lstm_params.get("dropout", 0.3)
        stacked = lstm_params.get("use_second_layer", False)
    except Exception as e:
        print("LSTM Tuning failed:", e)
        units, lr, batch, dropout, stacked = 64, 1e-3, 32, 0.3, False
        
    model_lstm = Sequential()
    model_lstm.add(Input(shape=(SEQ_LEN, len(feature_cols))))
    if stacked:
        model_lstm.add(LSTM(units, return_sequences=True))
        model_lstm.add(Dropout(dropout))
        model_lstm.add(LSTM(units // 2, return_sequences=False))
    else:
        model_lstm.add(LSTM(units, return_sequences=False))
    model_lstm.add(Dropout(dropout))
    model_lstm.add(Dense(1, activation="sigmoid"))
    model_lstm.compile(optimizer=Adam(lr), loss="binary_crossentropy", metrics=["AUC"])
    
    es = EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)
    model_lstm.fit(X_fit, y_fit, validation_data=(X_val, y_val), epochs=100, batch_size=batch, verbose=0, callbacks=[es])
    
    # 5) LSTM Proba for Stage 2
    lstm_train_proba = model_lstm.predict(X_lstm_train, verbose=0).flatten()
    lstm_test_proba = model_lstm.predict(X_lstm_test, verbose=0).flatten()
    
    # 6) Train XGBoost Classifier on Features + LSTM Proba
    print("\n[Step 2] Tuning & Training XGBoost Classifier Hybrid...")
    # Align features with sequences. The LSTM predicts `target_binary` at time t based on data up to t-1.
    # `y_lstm_train` corresponds to `target_binary` at time t (which is `target_return` from t to t+1 in original data shift).
    # We must use Tabular features at t-1 to avoid data leakage.
    feat_train_aligned = df_train_scaled[feature_cols].values[SEQ_LEN-1:-1]
    feat_test_aligned = df_test_scaled[feature_cols].values[SEQ_LEN-1:-1]
    
    X_xgb_train = np.column_stack([feat_train_aligned, lstm_train_proba])
    X_xgb_test = np.column_stack([feat_test_aligned, lstm_test_proba])
    y_xgb_train = y_lstm_train # These are already aligned
    y_xgb_test = y_lstm_test
    
    X_xf, y_xf, X_xv, y_xv = time_split(X_xgb_train, y_xgb_train, val_frac=0.15)
    
    try:
        xgb_params = tune_xgb_clf(X_xf, y_xf, X_xv, y_xv, n_trials=BO_XGB_TRIALS)
        clf_hybrid = XGBClassifier(**xgb_params, use_label_encoder=False, eval_metric="logloss", random_state=42, early_stopping_rounds=20)
    except Exception as e:
        print("XGB Tuning failed:", e)
        clf_hybrid = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, early_stopping_rounds=20)
        
    clf_hybrid.fit(X_xf, y_xf, eval_set=[(X_xv, y_xv)], verbose=False)
    
    # 7) Final Evaluation
    y_pred_lstm = (lstm_test_proba > 0.5).astype(int)
    y_proba_hybrid = clf_hybrid.predict_proba(X_xgb_test)[:, 1]
    y_pred_hybrid = (y_proba_hybrid > 0.5).astype(int)
    
    mets_lstm = compute_clf_metrics(y_xgb_test, y_pred_lstm, lstm_test_proba)
    mets_hybrid = compute_clf_metrics(y_xgb_test, y_pred_hybrid, y_proba_hybrid)
    
    res_df = pd.DataFrame([mets_lstm, mets_hybrid], index=["LSTM Standalone CLF", "Hybrid CLF (LSTM+XGB)"])
    print(f"\n--- Results for {T} ---")
    print(res_df.to_string())
    
    # 8) Generate Plots
    plot_roc_curve(y_xgb_test, y_proba_hybrid, save_path=os.path.join(OUTPUT_DIR, f"roc_curve_{T}.html"))
    plot_confusion_matrix(y_xgb_test, y_pred_hybrid, save_path=os.path.join(OUTPUT_DIR, f"confusion_matrix_{T}.html"))
    
    # 9) Save models
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{T}.pkl"))
    model_lstm.save(os.path.join(MODEL_DIR, f"lstm_clf_{T}.h5"))
    joblib.dump(clf_hybrid, os.path.join(MODEL_DIR, f"xgb_clf_{T}.pkl"))
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"hybrid_clf_metrics_{T}.csv"))

print("\nClassification Hybrid Training Complete.")
