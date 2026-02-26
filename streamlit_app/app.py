import time
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
import os

from src.preprocessing import load_and_save_yahoo
from streamlit_app.utils import (
    load_artifacts,
    load_regressor,
    prepare_data,
    predict_proba,
    predict_returns,
    prob_to_signal,
    calculate_position_size,
    run_backtest
)
from streamlit_app.portfolio import init_portfolio, update_equity
from streamlit_app.paper_trader import execute_trade

# =====================================================
# Page config
# =====================================================
st.set_page_config(
    page_title="ML Trading Dashboard",
    layout="wide"
)

st.title("📈 ML-Based Stock Trading Dashboard")

# =====================================================
# Session state
# =====================================================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = init_portfolio(100000)

if "last_candle_time" not in st.session_state:
    st.session_state.last_candle_time = {}  # per ticker

# =====================================================
# Sidebar controls
# =====================================================
st.sidebar.header("⚙️ Controls")

tickers = st.sidebar.multiselect(
    "Select Assets",
    ["AAPL", "MSFT", "NVDA", "GOOGL"],
    default=["AAPL"]
)

start_date = st.sidebar.text_input("Start Date", "2015-01-01")

threshold = st.sidebar.slider(
    "Probability Threshold",
    min_value=0.50,
    max_value=0.75,
    value=0.60,
    step=0.01
)

auto_trade = st.sidebar.toggle("🤖 Enable Auto-Trade")

reload_data = st.sidebar.button("🔄 Reload Data")

# =====================================================
# Data loading
# =====================================================
if reload_data:
    st.cache_data.clear()

@st.cache_data
def load_data_cached(ticker, start):
    return load_and_save_yahoo(ticker, start)

# =====================================================
# Paper trading
# =====================================================
st.subheader("🧪 Paper Trading")

latest_prices = {}

for ticker in tickers:
    # -------------------------------
    # Load data
    # -------------------------------
    df_raw = load_data_cached(ticker, start_date)
    df = prepare_data(df_raw, ticker=ticker, start_date=start_date)

    # -------------------------------
    # Load correct model PER TICKER
    # -------------------------------
    model, scaler, features = load_artifacts(ticker)

    # -------------------------------
    # Predict
    # -------------------------------
    proba = predict_proba(df, model, scaler, features)
    latest_prob = proba[-1]
    latest_price = df.iloc[-1]["Close"]
    latest_time = df.iloc[-1]["Date"]

    latest_prices[ticker] = latest_price

    signal = prob_to_signal(
        latest_prob,
        buy_th=threshold,
        sell_th=1 - threshold
    )

    size = calculate_position_size(
        capital=st.session_state.portfolio["cash"],
        price=latest_price,
        probability=latest_prob,
        df=df
    )

    st.markdown(f"### {ticker}")
    st.write(f"📈 Price: **{latest_price:.2f}**")
    st.write(f"🤖 Probability: **{latest_prob:.2f}**")
    st.write(f"📌 Signal: **{signal}**")
    st.write(f"📦 Suggested Size: **{size} shares**")

    # -------------------------------
    # Forecasts (1 / 10 / 30 days)
    # -------------------------------
    try:
        reg_model, reg_meta = load_regressor(ticker)
        from streamlit_app.utils import load_legacy_artifacts
        _, reg_scaler, reg_features = load_legacy_artifacts(ticker)
        preds = predict_returns(df, reg_model, reg_scaler, reg_features, horizons=(1, 10, 30), reg_meta=reg_meta)
        forecast_rows = []
        for h in (1, 10, 30):
            forecast_rows.append({
                "Horizon (days)": h,
                "Expected Return": f"{preds[h]['cum_return']*100:.2f}%",
                "Forecast Price": f"{preds[h]['price']:.2f}"
            })
        st.table(pd.DataFrame(forecast_rows))
    except Exception as e:
        st.info(f"Forecast unavailable: {e}")

    # -------------------------------
    # Auto-trade (per ticker candle)
    # -------------------------------
    last_time = st.session_state.last_candle_time.get(ticker)

    if auto_trade and (last_time is None or latest_time > last_time):
        st.session_state.portfolio = execute_trade(
            portfolio=st.session_state.portfolio,
            ticker=ticker,
            signal=signal,
            price=latest_price,
            timestamp=latest_time,
            size=size
        )
        st.session_state.last_candle_time[ticker] = latest_time

# =====================================================
# Update portfolio equity
# =====================================================
st.session_state.portfolio = update_equity(
    st.session_state.portfolio,
    latest_prices
)

# =====================================================
# Portfolio summary
# =====================================================
st.subheader("📊 Portfolio Summary")

col1, col2 = st.columns(2)
col1.metric("💵 Cash", f"${st.session_state.portfolio['cash']:.2f}")
col2.metric("💰 Equity", f"${st.session_state.portfolio['equity']:.2f}")

st.subheader("📦 Open Positions")
if st.session_state.portfolio["positions"]:
    st.dataframe(
        pd.DataFrame.from_dict(
            st.session_state.portfolio["positions"],
            orient="index"
        )
    )
else:
    st.info("No open positions")

st.subheader("📜 Trade History")
if st.session_state.portfolio["trades"]:
    st.dataframe(pd.DataFrame(st.session_state.portfolio["trades"]))
else:
    st.info("No trades yet")

# =====================================================
# Backtest analytics (AAPL only)
# =====================================================
st.subheader("📈 Strategy Analytics (AAPL)")

df_bt = prepare_data(load_data_cached("AAPL", start_date))
model, scaler, features = load_artifacts("AAPL")
proba_bt = predict_proba(df_bt, model, scaler, features)

bt, signals, sharpe, sortino = run_backtest(
    df_bt, proba_bt, threshold
)

col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
col2.metric("Sortino Ratio", f"{sortino:.2f}")
col3.metric("Trades Taken", int(signals.sum()))

# =====================================================
# Equity curve
# =====================================================
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=bt["Date"],
    y=bt["cum_strategy"],
    name="Strategy"
))
fig_eq.add_trace(go.Scatter(
    x=bt["Date"],
    y=bt["cum_market"],
    name="Market"
))
fig_eq.update_layout(template="plotly_dark")
st.plotly_chart(fig_eq, width="stretch")

# =====================================================
# Drawdown
# =====================================================
equity = bt["cum_strategy"]
drawdown = (equity - equity.cummax()) / equity.cummax()

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=bt["Date"],
    y=drawdown,
    fill="tozeroy",
    name="Drawdown"
))
fig_dd.update_layout(template="plotly_dark")
st.plotly_chart(fig_dd, width="stretch")

# =====================================================
# Saved Metrics & Plots
# =====================================================
st.subheader("📑 Saved Metrics & Reports")
for ticker in tickers:
    st.markdown(f"#### {ticker}")
    metrics_path = f"outputs/{ticker}_improved_metrics.csv"
    if not os.path.exists(metrics_path):
        metrics_path = f"outputs/{ticker}_metrics.csv"
        
    if os.path.exists(metrics_path):
        st.dataframe(pd.read_csv(metrics_path))
    else:
        st.info("Metrics file not found.")

    html_reports = []
    if os.path.exists(f"outputs/{ticker}_equity_test.html"):
        html_reports.append(("Equity Curve (test)", f"outputs/{ticker}_equity_test.html", 450))
    if os.path.exists(f"outputs/{ticker}_conf_vs_return_test.html"):
        html_reports.append(("Confidence vs Return", f"outputs/{ticker}_conf_vs_return_test.html", 450))
        
    if os.path.exists(f"outputs/{ticker}_improved_fi.html"):
        html_reports.append(("Feature Importance (Improved)", f"outputs/{ticker}_improved_fi.html", 500))
    elif os.path.exists(f"outputs/{ticker}_fi.html"):
        html_reports.append(("Feature Importance", f"outputs/{ticker}_fi.html", 450))
        
    for title, path, height in html_reports:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                html_str = f.read()
            with st.expander(f"{title}"):
                components.html(html_str, height=height, scrolling=True)
        else:
            st.info(f"{title} not found.")

# =====================================================
# Auto refresh
# =====================================================
if auto_trade:
    time.sleep(60)
    st.rerun()

