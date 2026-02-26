import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from seq2seq_lstm import build_seq2seq
import joblib
import os

SEQ_LEN = 30
HORIZON = 5
MODEL_DIR = "models"

def build_sequences(df, feature_cols):
    X, y = [], []

    data = df[feature_cols].values
    returns = df["target_return"].values

    for i in range(SEQ_LEN, len(df) - HORIZON):
        X.append(data[i-SEQ_LEN:i])
        y.append(returns[i:i+HORIZON])

    return np.array(X), np.array(y)

def train_seq2seq(df, ticker, feature_cols):

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

    X, y = build_sequences(df_scaled, feature_cols)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_seq2seq(len(feature_cols), SEQ_LEN, HORIZON)

    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[es]
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/seq2seq_{ticker}.keras")
    joblib.dump(scaler, f"{MODEL_DIR}/seq2seq_scaler_{ticker}.pkl")

    print(f"Seq2Seq model saved for {ticker}")
