import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import optuna
import datetime
import yfinance as yf
from pandas_datareader import data as pdr
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, matthews_corrcoef, log_loss, precision_score, recall_score,
    roc_curve, auc, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Colab Setup & Config
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TICKERS         = ["AAPL", "MSFT", "NVDA", "GOOGL", "GC=F"]
START_DATE      = "2015-01-01"
CLF_HORIZON     = 21
MAX_FEATURES    = 25
N_OPTUNA_TRIALS = 50
N_CV_SPLITS     = 5
MODEL_DIR       = "models"
OUTPUT_DIR      = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DROP_COLS = {"target_return", "target", "Date", "Ticker",
             "Open", "High", "Low", "Close", "Adj Close", "Volume"}

FRED_SERIES = {
    "CPIAUCSL": "CPI",
    "UNRATE": "UNRATE",
    "FEDFUNDS": "FFR",
    "GS10": "10Y_Treasury"
}

# ---------------------------------------------------------------------------
# Preprocessing Logic
# ---------------------------------------------------------------------------
def load_and_save_yahoo(ticker, start="2015-01-01", data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    path = f"{data_dir}/{ticker}_yahoo.csv"
    
    if os.path.exists(path):
        print(f"Using existing data at {path}")
        return pd.read_csv(path, parse_dates=["Date"])

    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df["Ticker"] = ticker
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.to_csv(path, index=False)
    return df

# ---------------------------------------------------------------------------
# Macro Logic
# ---------------------------------------------------------------------------
def fetch_macro(start_date="2000-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.date.today().isoformat()
    out = pd.DataFrame()
    for series, colname in FRED_SERIES.items():
        try:
            s = pdr.DataReader(series, "fred", start_date, end_date)
            s = s.rename(columns={series: colname})
            out = pd.concat([out, s], axis=1)
        except Exception as e:
            print(f"Macro fetch failed for {series}: {e}")
    if out.empty: return out
    return out.ffill().asfreq("D").ffill()

# ---------------------------------------------------------------------------
# Features Logic
# ---------------------------------------------------------------------------
def _rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    return 100 - (100 / (1 + up / (down + 1e-9)))

def add_features(df, ticker=None, start_date=None):
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["ret_1d"] = c.pct_change()
    df["ret_5d"] = c.pct_change(5)
    df["ret_21d"] = c.pct_change(21)
    df["vol_21d"] = df["ret_1d"].rolling(21).std()
    
    df["price_sma20_ratio"] = c / (c.rolling(20).mean() + 1e-9) - 1
    df["price_sma50_ratio"] = c / (c.rolling(50).mean() + 1e-9) - 1
    
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = (ema12 - ema26) / (c + 1e-9)
    
    df["rsi_14"] = _rsi(c, 14)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pct"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)
    
    # Macro
    try:
        macro = fetch_macro(start_date=(start_date or df.index.min().date()))
        if not macro.empty:
            macro = macro.reindex(df.index).ffill().bfill().fillna(0.0).add_prefix("macro_")
            df = pd.concat([df, macro], axis=1)
    except: pass
    
    for col in ["sentiment", "sentiment_lag1", "sentiment_sma5"]:
        df[col] = 0.0

    return df.ffill().bfill().fillna(0.0)

def create_target(df, horizon=21):
    df = df.copy()
    future_ret = df["Close"].pct_change(horizon).shift(-horizon)
    df["target_return"] = future_ret
    df["target"] = (future_ret > 0.0).astype(int)
    return df.dropna(subset=["target"])

# ---------------------------------------------------------------------------
# Visualisation & Metrics Logic
# ---------------------------------------------------------------------------
def sharpe(returns): return np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
def sortino(returns):
    downside = returns[returns < 0]
    return np.mean(returns) / (np.std(downside) + 1e-9) * np.sqrt(252)

def _save_plot(fig, path):
    fig.write_html(path)
    print(f"Saved plot: {path}")

def plot_equity_curve(df, path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["cum_strategy"], name="Strategy"))
    fig.add_trace(go.Scatter(x=df.index, y=df["cum_market"], name="Market"))
    fig.update_layout(title="Equity Curve", template="plotly_dark")
    _save_plot(fig, path)

def plot_feature_importance(features, importances, path, title):
    fi = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance").tail(20)
    fig = go.Figure(go.Bar(x=fi["Importance"], y=fi["Feature"], orientation="h"))
    fig.update_layout(title=title, template="plotly_dark")
    _save_plot(fig, path)

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------
def _clean(df):
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return df.dropna()

def _sample_weights(y): return compute_sample_weight("balanced", y)

def _optuna_tune(X_tr, y_tr):
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "random_state": 42, "n_jobs": -1, "eval_metric": "logloss"
        }
        scores = []
        for train_idx, val_idx in tscv.split(X_tr):
            clf = XGBClassifier(**params)
            clf.fit(X_tr[train_idx], y_tr[train_idx])
            scores.append(roc_auc_score(y_tr[val_idx], clf.predict_proba(X_tr[val_idx])[:,1]))
        return np.mean(scores)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)
    return study.best_params

def train_ticker(ticker):
    print(f"\n--- Training {ticker} ---")
    df_raw = load_and_save_yahoo(ticker, START_DATE)
    df = add_features(df_raw, start_date=START_DATE)
    df = create_target(df, horizon=CLF_HORIZON)
    df = _clean(df)
    
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    split = int(len(df) * 0.8)
    df_tr, df_te = df.iloc[:split], df.iloc[split:]
    
    X_tr_raw, y_tr = df_tr[feature_cols].values, df_tr["target"].values
    X_te_raw, y_te = df_te[feature_cols].values, df_te["target"].values
    
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X_tr_raw)
    X_te = scaler.transform(X_te_raw)
    
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=MAX_FEATURES)
    X_tr_sel = selector.fit_transform(X_tr, y_tr)
    X_te_sel = selector.transform(X_te)
    sel_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    
    best_params = _optuna_tune(X_tr_sel, y_tr)
    model = XGBClassifier(**best_params)
    model.fit(X_tr_sel, y_tr, sample_weight=_sample_weights(y_tr))
    
    # Save
    joblib.dump(model, os.path.join(MODEL_DIR, f"model_{ticker}.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{ticker}.pkl"))
    joblib.dump(sel_features, os.path.join(MODEL_DIR, f"features_{ticker}.pkl"))
    
    y_prob = model.predict_proba(X_te_sel)[:, 1]
    acc = accuracy_score(y_te, (y_prob > 0.5).astype(int))
    print(f"Accuracy: {acc:.4f}")
    
    plot_feature_importance(sel_features, model.feature_importances_, os.path.join(OUTPUT_DIR, f"{ticker}_fi.html"), ticker)
    return {"ticker": ticker, "accuracy": acc}

if __name__ == "__main__":
    results = []
    for t in TICKERS:
        try: results.append(train_ticker(t))
        except Exception as e: print(f"Failed {t}: {e}")
    if results:
        pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
        print("\nAll tasks complete.")
