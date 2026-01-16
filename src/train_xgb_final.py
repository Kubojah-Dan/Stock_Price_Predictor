import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score

from preprocessing import load_and_save_yahoo
from features import add_features, create_target
from walk_forward import walk_forward_validation
from backtest import backtest_with_costs
from metrics import sharpe, sortino  

from visualize import (
    plot_equity_curve,
    plot_drawdown,
    plot_feature_importance,
    plot_trades,
    plot_confidence_vs_return
)

# ================= CONFIG =================
TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL"]
START_DATE = "2015-01-01"
THRESHOLDS = [0.55, 0.6, 0.65, 0.7]
TOP_K_FEATURES = 5
OUTPUT_DIR = "outputs"
# =========================================

os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_FEATURES = [
    "return_1d", "log_return",
    "sma_5", "sma_10", "sma_20",
    "ema_12", "ema_26",
    "macd", "rsi_14",
    "vol_change"
]

# =========================================================
# LOOP OVER TICKERS
# =========================================================
for TICKER in TICKERS:
    print(f"\n🚀 Training XGBoost pipeline for {TICKER}")

    # =================================================
    # 1️⃣ Load Yahoo data
    # =================================================
    df = load_and_save_yahoo(TICKER, START_DATE)

    # =================================================
    # 2️⃣ Feature engineering + target
    # =================================================
    df = create_target(add_features(df))
    df_features = add_features(df)

    # =================================================
    # STAGE 1 — Feature importance (temporary model)
    # =================================================
    X = df[ALL_FEATURES].values
    y = df["target"].values

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler_all = StandardScaler()
    X_train = scaler_all.fit_transform(X_train)
    X_test = scaler_all.transform(X_test)

    temp_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    temp_model.fit(X_train, y_train)

    importances = pd.Series(
        temp_model.feature_importances_,
        index=ALL_FEATURES
    ).sort_values(ascending=False)

    TOP_FEATURES = importances.head(TOP_K_FEATURES).index.tolist()

    print("🔝 Selected Features:", TOP_FEATURES)

    # =================================================
    # STAGE 2 — Final XGBoost model (pruned features)
    # =================================================
    X_final = df[TOP_FEATURES].values
    y_final = y

    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_final)

    final_model = XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    final_model.fit(X_final, y_final)

    # =================================================
    # 3️⃣ Walk-forward validation
    # =================================================
    def model_fn():
        return XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        )

    wf_acc, _ = walk_forward_validation(
        df, TOP_FEATURES, "target", model_fn
    )

    print(f"📈 Walk-forward accuracy ({TICKER}): {wf_acc:.4f}")

    # =================================================
    # 4️⃣ Probabilities (XGBoost)
    # =================================================
    xgb_proba = final_model.predict_proba(X_final)[:, 1]

    # Placeholder LSTM probabilities (for ensemble viz)
    lstm_proba = (
        pd.Series(xgb_proba)
        .rolling(5)
        .mean()
        .fillna(0)
        .values
    )

    # =================================================
    # 5️⃣ Ensemble visualization (XGB vs LSTM)
    # =================================================
    fig_ens = px.scatter(
        x=xgb_proba,
        y=lstm_proba,
        labels={
            "x": "XGBoost Confidence",
            "y": "LSTM Confidence"
        },
        title=f"{TICKER} — XGBoost vs LSTM Confidence"
    )
    fig_ens.write_html(
        f"{OUTPUT_DIR}/{TICKER}_xgb_vs_lstm.html"
    )

    # =================================================
    # 6️⃣ Threshold tuning + backtest
    # =================================================
    best_bt = None
    best_signals = None

    for th in THRESHOLDS:
        signals = (xgb_proba > th).astype(int)
        precision = precision_score(
            y_final, signals, zero_division=0
        )

        bt = backtest_with_costs(df, signals)
        final_return = bt["cum_strategy"].iloc[-1] - 1

        sh = sharpe(bt["strategy"])
        so = sortino(bt["strategy"])

        print(
            f"{TICKER} | Threshold {th:.2f} | "
            f"Precision {precision:.2f} | "
            f"Return {final_return:.2%} | "
            f"Sharpe {sh:.2f} | "
            f"Sortino {so:.2f}"
        )

        # keep last backtest for visuals
        best_bt = bt.copy()
        best_signals = signals.copy()

    # =================================================
    # 7️⃣ Save artifacts
    # =================================================
    joblib.dump(
        final_model,
        f"models/xgb_{TICKER}_final.pkl"
    )
    joblib.dump(
        scaler,
        f"models/scaler_{TICKER}_final.pkl"
    )
    joblib.dump(
        TOP_FEATURES,
        f"models/features_{TICKER}_final.pkl"
    )

    print(f"✅ Saved model artifacts for {TICKER}")

    # =================================================
    # 8️⃣ Visualizations (saved per ticker)
    # =================================================
    best_bt["signal"] = best_signals

    plot_equity_curve(
        best_bt,
        save_path=f"{OUTPUT_DIR}/{TICKER}_equity_curve.html"
    )

    plot_drawdown(
        best_bt,
        save_path=f"{OUTPUT_DIR}/{TICKER}_drawdown.html"
    )

    plot_feature_importance(
        TOP_FEATURES,
        final_model.feature_importances_,
        save_path=f"{OUTPUT_DIR}/{TICKER}_feature_importance.html"
    )

    plot_trades(
        best_bt,
        save_path=f"{OUTPUT_DIR}/{TICKER}_trades.html"
    )

    plot_confidence_vs_return(
        df,
        xgb_proba,
        save_path=f"{OUTPUT_DIR}/{TICKER}_confidence_vs_return.html"
    )

    print(f"📊 Saved visualizations for {TICKER}")

print("\n🎉 All tickers trained successfully")

