import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from config import *
from preprocessing import load_and_save_yahoo
from features import add_features, create_target
from utils import time_split

df = load_and_save_yahoo(TICKER, START_DATE)
df = add_features(df)
df = create_target(df)

FEATURES = [
    "return_1d",
    "sma_10", "sma_20",
    "ema_12", "ema_26",
    "macd", "rsi_14", "vol_change"
]

X = df[FEATURES].values
y = df["target"].values

X_train, X_test, y_train, y_test = time_split(X, y, TEST_SIZE)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

# ✅ FEATURE IMPORTANCE AFTER FIT
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print("\nFeature importance:")
print(importance)

TOP_FEATURES = importance.head(5).index.tolist()

# ✅ PROBABILITIES AFTER FIT
proba = model.predict_proba(X_test)[:, 1]

THRESHOLD = 0.6
signals = (proba > THRESHOLD).astype(int)

preds = model.predict(X_test)
print("\nXGBoost Accuracy:", accuracy_score(y_test, preds) * 100)

joblib.dump(model, f"{MODELS_DIR}/xgb_{TICKER}.pkl")
joblib.dump(scaler, f"{MODELS_DIR}/scaler_{TICKER}.pkl")
joblib.dump(FEATURES, f"{MODELS_DIR}/features_{TICKER}.pkl")
joblib.dump(THRESHOLD, f"{MODELS_DIR}/threshold_{TICKER}.pkl")


