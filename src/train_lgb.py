import joblib
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config import *
from preprocessing import load_and_save_yahoo
from features import add_features, create_target
from utils import time_split

df = load_and_save_yahoo(TICKER, START_DATE)
df = create_target(add_features(df))

FEATURES = [
    "return_1d", "log_return",
    "sma_5", "sma_10", "sma_20",
    "ema_12", "ema_26",
    "macd", "rsi_14",
    "vol_change"
]

X = df[FEATURES].values
y = df["target"].values

X_train, X_test, y_train, y_test = time_split(X, y, TEST_SIZE)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("LightGBM Accuracy:", accuracy_score(y_test, preds) * 100)

joblib.dump(model, f"{MODELS_DIR}/lgb_{TICKER}.pkl")

