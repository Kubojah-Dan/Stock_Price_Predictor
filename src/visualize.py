import os
import webbrowser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

AUTO_OPEN_PLOTS = os.getenv("AUTO_OPEN_PLOTS", "0") == "1"


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.write_html(path)
    print(f"Saved: {path}")
    if AUTO_OPEN_PLOTS:
        try:
            webbrowser.open(f"file:///{os.path.abspath(path)}")
        except Exception:
            pass


def plot_equity_curve(df, save_path="outputs/equity_curve.html"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["cum_strategy"], name="Strategy", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=df.index, y=df["cum_market"], name="Market", line=dict(color="orange")))
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Cumulative Return", template="plotly_dark")
    _save(fig, save_path)


def plot_drawdown(df, save_path="outputs/drawdown.html"):
    equity = df["cum_strategy"]
    drawdown = (equity - equity.cummax()) / (equity.cummax() + 1e-9)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=drawdown, fill="tozeroy", name="Drawdown", line=dict(color="red")))
    fig.update_layout(title="Drawdown", yaxis_title="Drawdown %", xaxis_title="Date", template="plotly_dark")
    _save(fig, save_path)


def plot_feature_importance(features, importances, save_path="outputs/feature_importance.html", title="Feature Importance"):
    fi = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=True).tail(20)
    fig = go.Figure(go.Bar(x=fi["Importance"], y=fi["Feature"], orientation="h", marker_color="teal"))
    fig.update_layout(title=title, xaxis_title="Importance", template="plotly_dark", height=600)
    _save(fig, save_path)


def plot_trades(df, save_path="outputs/trades.html"):
    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Price", line=dict(color="lightblue")))
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers",
                             marker=dict(color="lime", size=6, symbol="triangle-up"), name="Buy"))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers",
                             marker=dict(color="red", size=6, symbol="triangle-down"), name="Sell"))
    fig.update_layout(title="Trade Signals", xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    _save(fig, save_path)


def plot_confidence_vs_return(df, probas, save_path="outputs/confidence_vs_return.html"):
    future_returns = df["Close"].pct_change().shift(-1)
    plot_df = pd.DataFrame({"Confidence": probas, "Future Return": future_returns.values}).dropna()
    fig = px.scatter(plot_df, x="Confidence", y="Future Return",
                     title="Model Confidence vs Next-Day Return",
                     template="plotly_dark", opacity=0.5,
                     trendline="lowess")
    _save(fig, save_path)


def plot_roc_curve(y_true, y_proba, save_path="outputs/roc_curve.html"):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", template="plotly_dark")
    _save(fig, save_path)


def plot_confusion_matrix(y_true, y_pred, save_path="outputs/confusion_matrix.html"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual"),
                    x=["Down", "Up"], y=["Down", "Up"])
    fig.update_layout(title="Confusion Matrix", template="plotly_dark")
    _save(fig, save_path)


def plot_calibration_curve(y_true, y_proba, save_path="outputs/calibration.html"):
    from sklearn.calibration import calibration_curve
    fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_pred, y=fraction_pos, mode="lines+markers", name="Model", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect"))
    fig.update_layout(title="Calibration Curve", xaxis_title="Mean Predicted Probability",
                      yaxis_title="Fraction of Positives", template="plotly_dark")
    _save(fig, save_path)


def plot_walk_forward_accuracy(fold_accs, ticker, save_path="outputs/walk_forward.html"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=fold_accs, mode="lines+markers", name="Fold Accuracy", line=dict(color="lime")))
    fig.add_hline(y=np.mean(fold_accs), line_dash="dash", line_color="orange",
                  annotation_text=f"Mean: {np.mean(fold_accs):.3f}")
    fig.update_layout(title=f"{ticker} Walk-Forward Accuracy", xaxis_title="Fold", yaxis_title="Accuracy",
                      template="plotly_dark")
    _save(fig, save_path)
