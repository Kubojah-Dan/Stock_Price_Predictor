import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config import *
from preprocessing import load_and_save_yahoo
from features import add_features, create_target

SEQ_LEN = 20

df = load_and_save_yahoo(TICKER, START_DATE)
df = create_target(add_features(df))

FEATURES = [
    "return_1d", "log_return",
    "sma_5", "sma_10", "sma_20",
    "ema_12", "ema_26",
    "macd", "rsi_14", "vol_change"
]
X = df[FEATURES].values
y = df["target"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

def make_seq(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = make_seq(X, y, SEQ_LEN)

split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

preds = (model.predict(X_test) > 0.5).astype(int)
print("LSTM Accuracy:", accuracy_score(y_test, preds) * 100)

model.save(f"{MODELS_DIR}/lstm_{TICKER}.h5")
joblib.dump(scaler, f"{MODELS_DIR}/scaler_lstm_{TICKER}.pkl")

