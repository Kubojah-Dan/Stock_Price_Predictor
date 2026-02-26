import os
import joblib
import numpy as np
import pandas as pd
from src.metrics import sharpe, sortino
from src.backtest import backtest_with_costs
from src.features import add_features, create_target

class StackingWrapper:
    def __init__(self, meta_learner, base_models):
        self.meta_learner = meta_learner
        self.base_models = base_models
        self.base_names = ['xgb', 'rf', 'et']
        
    def predict_proba(self, X):
        meta_features = []
        for name in self.base_names:
            meta_features.append(self.base_models[name].predict_proba(X)[:, 1])
        X_meta = np.column_stack(meta_features)
        return self.meta_learner.predict_proba(X_meta)

def load_artifacts(ticker):
    # Check for improved models first
    meta_path = f"models_improved/stacking_meta_lr_{ticker}.pkl"
    if os.path.exists(meta_path):
        meta_learner = joblib.load(meta_path)
        base_models = {}
        for name in ['xgb', 'rf', 'et']:
            base_models[name] = joblib.load(f"models_improved/stacking_base_{name}_{ticker}.pkl")
        model = StackingWrapper(meta_learner, base_models)
        scaler = joblib.load(f"models_improved/scaler_robust_{ticker}.pkl")
        features = joblib.load(f"models_improved/features_{ticker}.pkl")
        return model, scaler, features

def load_legacy_artifacts(ticker):
    model_candidates = [
        f"models/xgb_clf_{ticker}.pkl",          # ensemble classifier
        f"models/xgb_{ticker}_final.pkl",        # legacy single model
    ]
    scaler_candidates = [
        f"models/scaler_xgb_{ticker}.pkl",       # ensemble scaler
        f"models/scaler_{ticker}_final.pkl",     # legacy scaler
    ]
    features_candidates = [
        f"models/features_ensemble_{ticker}.pkl",
        f"models/features_{ticker}_final.pkl",
    ]

    def _first_existing(paths):
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    model_path = _first_existing(model_candidates)
    scaler_path = _first_existing(scaler_candidates)
    features_path = _first_existing(features_candidates)

    if model_path is None or scaler_path is None:
        raise FileNotFoundError(f"Missing legacy model/scaler for {ticker}.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if features_path and os.path.exists(features_path):
        features = joblib.load(features_path)
    elif hasattr(scaler, "feature_names_in_"):
        features = list(scaler.feature_names_in_)
    else:
        raise FileNotFoundError(f"No feature list found for {ticker} legacy.")

    return model, scaler, features


def load_artifacts(ticker):
    # Check for improved models first
    meta_path = f"models_improved/stacking_meta_lr_{ticker}.pkl"
    if os.path.exists(meta_path):
        meta_learner = joblib.load(meta_path)
        base_models = {}
        for name in ['xgb', 'rf', 'et']:
            base_models[name] = joblib.load(f"models_improved/stacking_base_{name}_{ticker}.pkl")
        model = StackingWrapper(meta_learner, base_models)
        scaler = joblib.load(f"models_improved/scaler_robust_{ticker}.pkl")
        features = joblib.load(f"models_improved/features_{ticker}.pkl")
        return model, scaler, features

    # If no improved model, fallback to legacy
    return load_legacy_artifacts(ticker)


def load_regressor(ticker):
    reg_candidates = [
        f"models/xgb_reg_{ticker}.pkl",
        f"models/xgb_{ticker}_reg.pkl",
    ]
    meta_path = f"models/meta_{ticker}.pkl"
    meta = {
        "reg_horizon": 1,
        "reg_shrink_w": 1.0,
        "reg_baseline": 0.0,
        "reg_model": "legacy",
    }
    if os.path.exists(meta_path):
        try:
            meta_loaded = joblib.load(meta_path)
            if isinstance(meta_loaded, dict):
                meta.update(meta_loaded)
        except Exception:
            pass
    for p in reg_candidates:
        if os.path.exists(p):
            return joblib.load(p), meta
    raise FileNotFoundError(f"Missing regressor for {ticker}. Checked {reg_candidates}")


def _ensure_feature_columns(df, features):
    dfc = df.copy()
    for col in features:
        if col not in dfc.columns:
            dfc[col] = 0.0
    return dfc[features]


def prepare_data(df, ticker=None, start_date=None):
    df = create_target(add_features(df, ticker=ticker, start_date=start_date))
    return df


def predict_proba(df, model, scaler, features):
    if hasattr(model, 'meta_learner'):
        # Improved model: reconstruct the 39 features needed by the RobustScaler
        drop_cols = ["target_price", "target_return", "target", "Date", "Ticker", 
                     "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        feature_cols = [c for c in df.columns if c not in drop_cols]
        X_full = _ensure_feature_columns(df, feature_cols)
        
        # RobustScaler was fit on these 39 features
        X_scaled = scaler.transform(X_full.values)
        
        # Subset to the selected 20 features
        indices = [feature_cols.index(f) for f in features if f in feature_cols]
        X = X_scaled[:, indices]
    else:
        # Legacy model
        X = scaler.transform(_ensure_feature_columns(df, features))
        
    proba = model.predict_proba(X)[:, 1]
    return proba


def _horizon_to_daily(ret_h, horizon):
    horizon = max(1, int(horizon))
    ret_h = max(float(ret_h), -0.99)
    return (1.0 + ret_h) ** (1.0 / horizon) - 1.0


def predict_returns(df, reg_model, scaler, features, horizons=(1, 10, 30), reg_meta=None):
    reg_meta = reg_meta or {}
    reg_h = int(reg_meta.get("reg_horizon", 1) or 1)
    reg_w = float(reg_meta.get("reg_shrink_w", 1.0) or 1.0)
    reg_base = float(reg_meta.get("reg_baseline", 0.0) or 0.0)

    last_row_raw = df.iloc[[-1]]
    X_last_df = _ensure_feature_columns(last_row_raw, features)
    X_last = scaler.transform(X_last_df)
    reg_ret_raw = float(reg_model.predict(X_last)[0])
    reg_ret = reg_w * reg_ret_raw + (1.0 - reg_w) * reg_base
    daily_ret = _horizon_to_daily(reg_ret, reg_h)
    last_price = float(last_row_raw["Close"].iloc[0])

    preds = {}
    for h in horizons:
        cum_ret = (1 + daily_ret) ** h - 1
        price = last_price * (1 + cum_ret)
        preds[h] = {
            "daily_return": daily_ret,
            "cum_return": cum_ret,
            "price": price
        }
    return preds


def run_backtest(df, proba, threshold):
    signals = (proba > threshold).astype(int)
    bt = backtest_with_costs(df, signals)

    sh = sharpe(bt["strategy"])
    so = sortino(bt["strategy"])

    return bt, signals, sh, so

def prob_to_signal(prob, buy_th=0.6, sell_th=0.4):
    if prob >= buy_th:
        return "BUY"
    elif prob <= sell_th:
        return "SELL"
    else:
        return "HOLD"

def kelly_fraction(win_prob, win_loss_ratio=1.0):
    """
    Fraction of capital to risk using Kelly criterion
    """
    k = win_prob - (1 - win_prob) / win_loss_ratio
    return max(0.0, min(k, 0.25))  # cap at 25% for safety


def volatility_target(df, window=20, target_vol=0.02):
    """
    Scale position size based on volatility
    """
    returns = df["Close"].pct_change().dropna()
    if len(returns) < window:
        return 1.0

    vol = returns.rolling(window).std().iloc[-1]
    if vol == 0 or np.isnan(vol):
        return 1.0

    return min(1.0, target_vol / vol)


def calculate_position_size(
    capital,
    price,
    probability,
    df
):
    """
    Final position sizing function
    """
    kelly = kelly_fraction(probability)
    vol_adj = volatility_target(df)

    fraction = kelly * vol_adj

    position_value = capital * fraction
    shares = int(position_value / price)

    return max(shares, 0)

