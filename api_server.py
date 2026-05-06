from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import joblib
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent))
from src.preprocessing import load_and_save_yahoo
from src.features import add_features, create_target
from src.auth import create_user, authenticate_user, verify_totp, create_access_token, decode_token, update_xm_credentials, get_db
from src.agent import get_agent_decision
import src.broker as broker

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

portfolio = {
    "cash": 100000.0,
    "equity": 100000.0,
    "positions": {},
    "trades": []
}

# ── Auth Models & Dependencies ──────────────────────────────────────────────────

class RegisterModel(BaseModel):
    email: str
    password: str

class LoginModel(BaseModel):
    email: str
    password: str
    totp_code: str = ""

class Verify2FAModel(BaseModel):
    email: str
    totp_code: str

class XMCredentialsModel(BaseModel):
    account: str
    password: str
    server: str

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    email = payload.get("sub")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)

# ── Auth Endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
async def register(data: RegisterModel):
    try:
        res = create_user(data.email, data.password)
        return {"message": "User created", "qr_uri": res["qr_uri"]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auth/verify-2fa")
async def verify_2fa_setup(data: Verify2FAModel):
    is_valid = verify_totp(data.email, data.totp_code, enable=True)
    if is_valid:
        return {"message": "2FA enabled successfully"}
    raise HTTPException(status_code=400, detail="Invalid 2FA code")

@app.post("/api/auth/login")
async def login(data: LoginModel):
    user = authenticate_user(data.email, data.password, data.totp_code)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials or 2FA code")
    token = create_access_token({"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/user/me")
async def get_me(user: dict = Depends(get_current_user)):
    user.pop("password_hash", None)
    user.pop("totp_secret", None)
    return user

@app.post("/api/user/xm")
async def update_xm(data: XMCredentialsModel, user: dict = Depends(get_current_user)):
    update_xm_credentials(user["email"], data.account, data.password, data.server)
    return {"message": "XM Credentials updated"}

# ── Broker Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/broker/status")
async def broker_status(user: dict = Depends(get_current_user)):
    if not user.get("xm_account"):
        return {"status": "not_configured"}
    
    success = broker.init_mt5(user["xm_account"], user["xm_password"], user["xm_server"])
    if not success:
        return {"status": "error", "message": "Could not connect to MT5"}
        
    info = broker.get_account_info()
    positions = broker.get_positions()
    return {"status": "connected", "info": info, "positions": positions}

# ── ML Models ──────────────────────────────────────────────────────────────────

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
    try:
        model = joblib.load(f"models/xgb_clf_{ticker}.pkl")
        scaler = joblib.load(f"models/scaler_xgb_{ticker}.pkl")
        features = joblib.load(f"models/features_ensemble_{ticker}.pkl")
        return model, scaler, features, "xgb"
    except Exception:
        pass

    try:
        model = joblib.load(f"models/model_{ticker}.pkl")
        scaler = joblib.load(f"models/scaler_{ticker}.pkl")
        features = joblib.load(f"models/features_{ticker}.pkl")
        return model, scaler, features, "calibrated_xgb"
    except Exception:
        pass

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
                X_full = df[all_feat].iloc[[-1]].values
                X_scaled = scaler.transform(X_full)
                idx = [all_feat.index(f) for f in features if f in all_feat]
                X = X_scaled[:, idx]

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

    avg_ret = returns.tail(30).mean()
    forecasts = [
        {"days": d,
         "expected_return": round(avg_ret * d * 100, 2),
         "forecast_price": round(latest_price * (1 + avg_ret * d), 2)}
        for d in [1, 3, 10]
    ]

    return {
        "ticker": ticker,
        "price": round(latest_price, 2),
        "probability": round(proba, 3),
        "signal": signal,
        "volatility": round(float(returns.tail(20).std()), 4),
        "date": str(df.index[-1]),
        "forecasts": forecasts
    }

@app.get("/api/metrics/{ticker}")
async def get_metrics(ticker: str):
    try:
        # Try ticker-specific file first
        metrics_path = f"outputs/{ticker}_metrics.csv"
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
        elif os.path.exists("outputs/summary.csv"):
            # Fallback to summary.csv
            summary_df = pd.read_csv("outputs/summary.csv")
            df = summary_df[summary_df["ticker"] == ticker]
            if df.empty:
                return {
                    "hybrid_accuracy": "N/A", "lstm_accuracy": "N/A", "f1_score": "N/A",
                    "auc": "N/A", "mcc": "N/A", "precision": "N/A", "recall": "N/A", "wf_auc": "N/A"
                }
        else:
            return {
                "hybrid_accuracy": "N/A", "lstm_accuracy": "N/A", "f1_score": "N/A",
                "auc": "N/A", "mcc": "N/A", "precision": "N/A", "recall": "N/A", "wf_auc": "N/A"
            }

        row = df.iloc[0]
        
        # Map columns dynamically based on what's available
        return {
            "hybrid_accuracy": round(float(row.get("accuracy", 0)) * 100, 1),
            "lstm_accuracy": round(float(row.get("balanced_acc", row.get("bal_accuracy", 0))) * 100, 1),
            "f1_score": round(float(row.get("f1", 0)), 2),
            "auc": round(float(row.get("roc_auc", row.get("auc", 0))), 2),
            "mcc": round(float(row.get("mcc", 0)), 3),
            "precision": round(float(row.get("precision", 0)), 2),
            "recall": round(float(row.get("recall", 0)), 2),
            "threshold": round(float(row.get("threshold", 0.5)), 2)
        }
    except Exception as e:
        print(f"Metrics fetch error for {ticker}: {e}")
        return {
            "hybrid_accuracy": "N/A", "lstm_accuracy": "N/A", "f1_score": "N/A",
            "auc": "N/A", "mcc": "N/A", "precision": "N/A", "recall": "N/A", "threshold": "N/A"
        }

@app.get("/api/plots/trend/{ticker}")
async def get_trend_plot(ticker: str):
    try:
        df = load_and_save_yahoo(ticker, "2024-01-01")
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
        fig.update_layout(title=f"{ticker} Recent Trend", template="plotly_dark", xaxis_rangeslider_visible=False)
        
        path = f"outputs/{ticker}_trend.html"
        fig.write_html(path)
        return {"plot_url": f"http://localhost:8000/outputs/{ticker}_trend.html"}
    except Exception as e:
        print(f"Trend plot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict/{ticker}")
async def predict(ticker: str, threshold: float = 0.60):
    try:
        df = get_df(ticker)
        return get_prediction(ticker, threshold, df)
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agent/chat/{ticker}")
async def agent_chat(ticker: str, threshold: float = 0.60, user: dict = Depends(get_current_user)):
    try:
        df = get_df(ticker)
        pred = get_prediction(ticker, threshold, df)
        
        # Add recent trend context (last 10 days OHLC)
        recent = df.tail(10).copy()
        trend_summary = []
        for d, r in recent.iterrows():
            trend_summary.append(f"{str(d.date())}: O:{r['Open']:.2f} H:{r['High']:.2f} L:{r['Low']:.2f} C:{r['Close']:.2f}")
        
        pred["trend_context"] = "\n".join(trend_summary)
        
        reasoning = get_agent_decision(pred)
        return {"decision": reasoning, "data": pred}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
