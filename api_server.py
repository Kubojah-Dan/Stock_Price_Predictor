from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from src.preprocessing import load_and_save_yahoo
from src.features import add_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Portfolio state (in-memory for demo)
portfolio = {
    "cash": 100000.0,
    "equity": 100000.0,
    "positions": {},
    "trades": []
}

class PredictionRequest(BaseModel):
    ticker: str
    threshold: float = 0.60

def load_model_artifacts(ticker):
    """Load XGBoost classifier, scaler, and features"""
    try:
        # Try models_clf first (best models)
        with open(f"models_clf/xgb_clf_{ticker}.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"models_clf/scaler_{ticker}.pkl", "rb") as f:
            scaler = pickle.load(f)
        # Use features from models folder
        with open(f"models/features_ensemble_{ticker}.pkl", "rb") as f:
            features = pickle.load(f)
        return model, scaler, features
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        return None, None, None

def prepare_features(ticker):
    """Load data and prepare features"""
    df = load_and_save_yahoo(ticker, "2015-01-01")
    df = add_features(df)
    df = df.dropna()
    return df

@app.get("/api/predict/{ticker}")
async def predict(ticker: str, threshold: float = 0.60):
    """Get prediction and trading signal for a ticker"""
    try:
        df = prepare_features(ticker)
        latest_price = float(df['Close'].iloc[-1])
        
        # Simple momentum-based prediction
        returns = df['Close'].pct_change()
        recent_return = returns.tail(5).mean()
        proba = 0.5 + (recent_return * 5)
        proba = max(0.35, min(0.75, proba))
        
        if proba >= threshold:
            signal = "BUY"
        elif proba <= (1 - threshold):
            signal = "SELL"
        else:
            signal = "HOLD"
        
        size = int((portfolio["cash"] * 0.1) / latest_price) if signal == "BUY" else 0
        volatility = float(returns.tail(20).std())
        
        # Calculate forecasts for 1, 10, 30 days
        avg_daily_return = returns.tail(30).mean()
        forecasts = []
        for days in [1, 10, 30]:
            expected_return = avg_daily_return * days
            forecast_price = latest_price * (1 + expected_return)
            forecasts.append({
                "days": days,
                "expected_return": round(expected_return * 100, 2),
                "forecast_price": round(forecast_price, 2)
            })
        
        return {
            "ticker": ticker,
            "price": round(latest_price, 2),
            "probability": round(float(proba), 3),
            "signal": signal,
            "suggested_size": size,
            "volatility": round(volatility, 4),
            "date": str(df.index[-1]),
            "forecasts": forecasts
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio state"""
    return portfolio

@app.post("/api/trade")
async def execute_trade(ticker: str, signal: str, price: float, size: int):
    """Execute a paper trade"""
    global portfolio
    
    try:
        if signal == "BUY" and size > 0:
            cost = price * size
            if cost <= portfolio["cash"]:
                portfolio["cash"] -= cost
                if ticker in portfolio["positions"]:
                    old_shares = portfolio["positions"][ticker]["shares"]
                    old_price = portfolio["positions"][ticker]["avg_price"]
                    new_shares = old_shares + size
                    portfolio["positions"][ticker]["shares"] = new_shares
                    portfolio["positions"][ticker]["avg_price"] = (
                        (old_shares * old_price) + (size * price)
                    ) / new_shares
                else:
                    portfolio["positions"][ticker] = {
                        "shares": size,
                        "avg_price": price
                    }
                portfolio["trades"].append({
                    "ticker": ticker,
                    "action": "BUY",
                    "price": price,
                    "shares": size,
                    "timestamp": pd.Timestamp.now().isoformat()
                })
                print(f"BUY executed: {size} shares of {ticker} at ${price}")
        
        elif signal == "SELL" and ticker in portfolio["positions"]:
            shares = portfolio["positions"][ticker]["shares"]
            revenue = price * shares
            portfolio["cash"] += revenue
            del portfolio["positions"][ticker]
            portfolio["trades"].append({
                "ticker": ticker,
                "action": "SELL",
                "price": price,
                "shares": shares,
                "timestamp": pd.Timestamp.now().isoformat()
            })
            print(f"SELL executed: {shares} shares of {ticker} at ${price}")
        
        # Update equity based on current positions
        portfolio["equity"] = portfolio["cash"]
        for t, pos in portfolio["positions"].items():
            # Get current price for each position
            try:
                df = prepare_features(t)
                current_price = float(df['Close'].iloc[-1])
                portfolio["equity"] += pos["shares"] * current_price
            except:
                portfolio["equity"] += pos["shares"] * pos["avg_price"]
        
        print(f"Portfolio updated: Cash=${portfolio['cash']:.2f}, Equity=${portfolio['equity']:.2f}")
        return portfolio
    except Exception as e:
        print(f"Trade error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolio/reset")
async def reset_portfolio():
    """Reset portfolio to initial state"""
    global portfolio
    portfolio = {
        "cash": 100000.0,
        "equity": 100000.0,
        "positions": {},
        "trades": []
    }
    return portfolio

@app.get("/api/metrics/{ticker}")
async def get_metrics(ticker: str):
    """Get model metrics for a ticker"""
    try:
        df = pd.read_csv(f"outputs/hybrid_clf_metrics_{ticker}.csv")
        hybrid = df.iloc[1].to_dict()
        return {
            "hybrid_accuracy": round(float(hybrid["Accuracy"]) * 100, 1),
            "lstm_accuracy": round(float(df.iloc[0]["Accuracy"]) * 100, 1),
            "f1_score": round(float(hybrid["F1-score"]), 2),
            "auc": round(float(hybrid["AUC"]), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
