import joblib
import numpy as np
import pandas as pd
from src.metrics import sharpe, sortino
from src.backtest import backtest_with_costs
from src.features import add_features, create_target

def load_artifacts(ticker):
    model = joblib.load(f"models/xgb_{ticker}_final.pkl")
    scaler = joblib.load(f"models/scaler_{ticker}_final.pkl")
    features = joblib.load(f"models/features_{ticker}_final.pkl")
    return model, scaler, features


def prepare_data(df):
    df = create_target(add_features(df))
    return df


def predict_proba(df, model, scaler, features):
    X = scaler.transform(df[features].values)
    proba = model.predict_proba(X)[:, 1]
    return proba


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

