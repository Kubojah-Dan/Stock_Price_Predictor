import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
import numpy as np
import os
import webbrowser

def _open(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path)
    webbrowser.open(f"file:///{os.path.abspath(path)}")
    print(f"📊 Opened: {path}")
    
def plot_equity_curve(df, save_path="outputs/equity_curve.html"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_strategy"], name="Strategy"
    ))
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["cum_market"], name="Market"
    ))

    fig.update_layout(
        title="Equity Curve",
        template="plotly_dark"
    )

    fig.write_html(save_path)
    print(f"📊 Visualization saved to {save_path}")

    #fig.show()
    _open(fig, save_path)

# 1️⃣ DRAW DOWN CHART

def plot_drawdown(df, save_path="outputs/drawdown.html"):
    equity = df["cum_strategy"]
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=drawdown,
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="red")
    ))

    fig.update_layout(
        title="Drawdown Curve",
        yaxis_title="Drawdown %",
        xaxis_title="Date",
        template="plotly_dark"
    )

    _open(fig, save_path)

# 2️⃣ FEATURE IMPORTANCE BAR CHART

def plot_feature_importance(features, importances,
                            save_path="outputs/feature_importance.html"):

    fi = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=fi["Importance"],
        y=fi["Feature"],
        orientation="h"
    ))

    fig.update_layout(
        title="Feature Importance (XGBoost)",
        xaxis_title="Importance",
        template="plotly_dark"
    )

    _open(fig, save_path)

# 3️⃣ TRADE ENTRY / EXIT MARKERS

def plot_trades(df, save_path="outputs/trades.html"):
    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == 0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        name="Price",
        line=dict(color="lightblue")
    ))

    fig.add_trace(go.Scatter(
        x=buys["Date"],
        y=buys["Close"],
        mode="markers",
        marker=dict(color="green", size=6),
        name="Buy"
    ))

    fig.add_trace(go.Scatter(
        x=sells["Date"],
        y=sells["Close"],
        mode="markers",
        marker=dict(color="red", size=6),
        name="Sell"
    ))

    fig.update_layout(
        title="Trade Entry / Exit",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark"
    )

    _open(fig, save_path)

# 4️⃣ CONFIDENCE vs RETURN SCATTER

def plot_confidence_vs_return(df, probas,
                              save_path="outputs/confidence_vs_return.html"):
    future_returns = df["Close"].pct_change().shift(-1)

    plot_df = pd.DataFrame({
        "Confidence": probas,
        "Future Return": future_returns
    }).dropna()

    fig = px.scatter(
        plot_df,
        x="Confidence",
        y="Future Return",
        title="Model Confidence vs Next-Day Return",
        template="plotly_dark",
        opacity=0.6
    )

    _open(fig, save_path)
