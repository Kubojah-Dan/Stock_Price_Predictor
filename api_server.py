from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import joblib

sys.path.append(str(Path(__file__).parent))
from src.preprocessing import load_and_save_yahoo
from src.features import add_features, create_target

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

portfolio = {
    "cash": 100000.0,
    "equity": 100000.0,
    "positions": {},
    "trades": []
}

# ── Model loading ──────────────────────────────────────────────────────────────

class StackingWrapper:
    def __init__(self, meta_learner, base_models):
        self.meta_learner = meta_learner
        self.base_models = base_models

    def predict_proba(self, X):
        meta_features = np.column_stack([
            self.base_models[n].predict_proba(X)[:, 1]
            for n in ['xgb', 'rf', 'et']
        ])
        return self.meta_learner.predict_proba(meta_features)


def load_model(ticker):
    """Load best available model for ticker."""
    # Try models/ (from train_improved.py - best models)
    try:
        model = joblib.load(f"models/xgb_clf_{ticker}.pkl")
        scaler = joblib.load(f"models/scaler_xgb_{ticker}.pkl")
        features = joblib.load(f"models/features_ensemble_{ticker}.pkl")
        return model, scaler, features, "xgb"
    except Exception:
        pass

    # Try models_improved/ (stacking ensemble)
    try:
        meta = joblib.load(f"models_improved/stacking_meta_lr_{ticker}.pkl")
        base = {n: joblib.load(f"models_improved/stacking_base_{n}_{ticker}.pkl")
                for n in ['xgb', 'rf', 'et']}
        model = StackingWrapper(meta, base)
        scaler = joblib.load(f"models_improved/scaler_robust_{ticker}.pkl")
        features = joblib.load(f"models_improved/features_{ticker}.pkl")
        return model, scaler, features, "stacking"
    except Exception:
        pass

    return None, None, None, None


def get_df(ticker):
    df_raw = load_and_save_yahoo(ticker, "2015-01-01")
    df = add_features(df_raw)
    df = create_target(df)
    df = df.dropna()
    return df


def kelly_fraction(prob, win_loss_ratio=1.0):
    k = prob - (1 - prob) / win_loss_ratio
    return max(0.0, min(k, 0.25))


def volatility_target(df, window=20, target_vol=0.02):
    returns = df["Close"].pct_change().dropna()
    if len(returns) < window:
        return 1.0
    vol = returns.rolling(window).std().iloc[-1]
    if vol == 0 or np.isnan(vol):
        return 1.0
    return min(1.0, target_vol / vol)


def calc_position_size(capital, price, prob, df):
    kelly = kelly_fraction(prob)
    vol_adj = volatility_target(df)
    shares = int((capital * kelly * vol_adj) / price)
    return max(shares, 0)


def get_prediction(ticker, threshold, df):
    model, scaler, features, model_type = load_model(ticker)
    latest_price = float(df["Close"].iloc[-1])
    returns = df["Close"].pct_change()

    if model is not None:
        try:
            drop_cols = {"target_price", "target_return", "target", "Date",
                         "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
            all_feat = [c for c in df.columns if c not in drop_cols
                        and np.issubdtype(df[c].dtype, np.number)]

            if model_type == "stacking":
                X_full = df[all_feat].iloc[[-1]].values
                X_scaled = scaler.transform(X_full)
                idx = [all_feat.index(f) for f in features if f in all_feat]
                X = X_scaled[:, idx]
            else:
                feat_cols = [f for f in features if f in df.columns]
                X = scaler.transform(df[feat_cols].iloc[[-1]].values)

            proba = float(model.predict_proba(X)[0, 1])
        except Exception as e:
            print(f"Model predict failed: {e}, falling back to momentum")
            proba = float(np.clip(0.5 + returns.tail(5).mean() * 5, 0.35, 0.75))
    else:
        proba = float(np.clip(0.5 + returns.tail(5).mean() * 5, 0.35, 0.75))

    if proba >= threshold:
        signal = "BUY"
    elif proba <= (1 - threshold):
        signal = "SELL"
    else:
        signal = "HOLD"

    size = calc_position_size(portfolio["cash"], latest_price, proba, df) if signal == "BUY" else 0

    avg_ret = returns.tail(30).mean()
    forecasts = [
        {"days": d,
         "expected_return": round(avg_ret * d * 100, 2),
         "forecast_price": round(latest_price * (1 + avg_ret * d), 2)}
        for d in [1, 10, 30]
    ]

    return {
        "ticker": ticker,
        "price": round(latest_price, 2),
        "probability": round(proba, 3),
        "signal": signal,
        "suggested_size": size,
        "volatility": round(float(returns.tail(20).std()), 4),
        "date": str(df.index[-1]),
        "forecasts": forecasts
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/predict/{ticker}")
async def predict(ticker: str, threshold: float = 0.60):
    try:
        df = get_df(ticker)
        return get_prediction(ticker, threshold, df)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio")
async def get_portfolio():
    return portfolio


@app.post("/api/trade")
async def execute_trade(ticker: str, signal: str, price: float, size: int):
    global portfolio
    try:
        if signal == "BUY" and size > 0:
            cost = price * size
            if cost <= portfolio["cash"]:
                portfolio["cash"] -= cost
                if ticker in portfolio["positions"]:
                    pos = portfolio["positions"][ticker]
                    total = pos["shares"] + size
                    pos["avg_price"] = (pos["shares"] * pos["avg_price"] + size * price) / total
                    pos["shares"] = total
                else:
                    portfolio["positions"][ticker] = {"shares": size, "avg_price": price}
                portfolio["trades"].append({
                    "ticker": ticker, "action": "BUY",
                    "price": price, "shares": size,
                    "timestamp": pd.Timestamp.now().isoformat()
                })
                print(f"BUY {size} {ticker} @ ${price:.2f}")

        elif signal == "SELL" and ticker in portfolio["positions"]:
            pos = portfolio["positions"][ticker]
            portfolio["cash"] += price * pos["shares"]
            portfolio["trades"].append({
                "ticker": ticker, "action": "SELL",
                "price": price, "shares": pos["shares"],
                "timestamp": pd.Timestamp.now().isoformat()
            })
            del portfolio["positions"][ticker]
            print(f"SELL {pos['shares']} {ticker} @ ${price:.2f}")

        # Recalculate equity using current prices
        portfolio["equity"] = portfolio["cash"]
        for t, pos in portfolio["positions"].items():
            try:
                cur_price = float(load_and_save_yahoo(t, "2020-01-01")["Close"].iloc[-1])
            except Exception:
                cur_price = pos["avg_price"]
            portfolio["equity"] += pos["shares"] * cur_price

        print(f"Portfolio: Cash=${portfolio['cash']:.2f}, Equity=${portfolio['equity']:.2f}")
        return portfolio
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/portfolio/reset")
async def reset_portfolio():
    global portfolio
    portfolio = {"cash": 100000.0, "equity": 100000.0, "positions": {}, "trades": []}
    return portfolio


@app.get("/api/metrics/{ticker}")
async def get_metrics(ticker: str):
    try:
        df = pd.read_csv(f"outputs/{ticker}_metrics.csv")
        row = df.iloc[0]
        return {
            "hybrid_accuracy": round(float(row["accuracy"]) * 100, 1),
            "lstm_accuracy": round(float(row["bal_accuracy"]) * 100, 1),
            "f1_score": round(float(row["f1"]), 2),
            "auc": round(float(row["auc"]), 2),
            "mcc": round(float(row["mcc"]), 3),
            "precision": round(float(row["precision"]), 2),
            "recall": round(float(row["recall"]), 2),
            "wf_auc": round(float(row["wf_auc"]), 3)
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
