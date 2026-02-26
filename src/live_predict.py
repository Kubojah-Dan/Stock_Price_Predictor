import joblib
import numpy as np
import pandas as pd
from preprocessing import load_and_save_yahoo
from features import add_features, create_target
import os

MODEL_DIR = "models"

def load_artifacts_for_ticker(ticker):
    model_reg = joblib.load(os.path.join(MODEL_DIR, f"xgb_reg_{ticker}.pkl"))
    model_clf = joblib.load(os.path.join(MODEL_DIR, f"xgb_clf_{ticker}.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"scaler_xgb_{ticker}.pkl"))
    return model_reg, model_clf, scaler

def predict_latest(ticker):
    df = load_and_save_yahoo(ticker, start_date="2015-01-01")
    df = add_features(df, ticker=ticker)
    df = df.dropna()
    # Load scaler + models (they were saved using chosen features)
    scaler = joblib.load(f"models/scaler_xgb_{ticker}.pkl")
    model_reg = joblib.load(f"models/xgb_reg_{ticker}.pkl")
    model_clf = joblib.load(f"models/xgb_clf_{ticker}.pkl")
    # Determine feature list by loading scaler feature names saved (if you saved them previously)
    # Here we assume scaler was fit on a dataframe slice: we'll pick the scaler.n_features_in_ features from df
    # Use last N numeric columns as fallback
    n_features = getattr(scaler, "n_features_in_", None)
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if n_features and len(numeric_cols) >= n_features:
        feature_cols = numeric_cols[-n_features:]
    else:
        feature_cols = numeric_cols

    X_latest = scaler.transform(df[feature_cols].values[-1:].reshape(1, -1))
    price_pred = model_reg.predict(X_latest)[0]
    prob = model_clf.predict_proba(X_latest)[:,1][0]
    signal = int(prob > 0.6)
    return {"ticker": ticker, "pred_price": price_pred, "prob": prob, "signal": signal}
