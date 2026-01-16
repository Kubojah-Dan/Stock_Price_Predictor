# 📈 ML-Based Stock Trading System

A **production-grade machine learning trading system** built using Python, XGBoost, Streamlit, and Yahoo Finance data.  
This project combines **data science, machine learning, backtesting, portfolio management, and live paper trading** into a single cohesive system.

<br/>

## 🚀 Key Features

### ✅ Market Data
- Historical OHLCV data downloaded directly from **Yahoo Finance**
- Automatic caching & local storage
- Multi-asset support (AAPL, MSFT, NVDA, GOOGL, etc.)

<br/>

### 🧠 Machine Learning
- **XGBoost classifier** trained per ticker
- Feature engineering with technical indicators:
  - Returns & log returns
  - SMA / EMA
  - MACD
  - RSI
  - Volatility change
- **Feature importance analysis + pruning**
- Walk-forward (rolling window) validation
- Probability-based predictions

<br/>

### 🤖 Trading Logic
- Converts probabilities into **BUY / SELL / HOLD** signals
- **Probability thresholding** (trade only when confidence is high)
- **Risk-aware position sizing** (capital + volatility aware)
- Long-only strategy (safe default)

<br/>

### 💼 Portfolio & Paper Trading
- Multi-asset paper trading engine
- Shared portfolio capital
- Per-asset positions
- Trade execution engine with:
  - Entry price
  - Position size
  - Timestamped trades
- Real-time portfolio equity tracking

<br/>

### 📊 Backtesting & Metrics
- Strategy vs Market comparison
- Transaction cost simulation
- Performance metrics:
  - Accuracy
  - Precision
  - Sharpe Ratio
  - Sortino Ratio
- Equity curve & drawdown analysis

<br/>

### 📈 Visualizations (Plotly)
- Equity curve
- Drawdown chart
- Feature importance bar chart
- Trade entry markers
- Confidence vs return plots
- XGBoost vs LSTM confidence (ensemble visualization placeholder)

All visualizations are saved as interactive HTML files.

<br/>

### 🖥️ Interactive Dashboard (Streamlit)
- Multi-asset selection
- Adjustable probability threshold
- Auto-trade toggle
- Live paper trading simulation
- Portfolio overview
- Trade history
- Strategy analytics

<br/>

### 1️⃣ Create virtual environment
```bash
python -m venv .StockVenv
source .StockVenv/bin/activate   # macOS/Linux
.StockVenv\Scripts\activate      # Windows

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Train models (multi-asset)
python src/train_xgb_final.py


This will:

Download Yahoo Finance data

Train one model per ticker

Save models, scalers, features

Generate visualizations in outputs/

### 4️⃣ Launch the dashboard
streamlit run streamlit_app/app.py

<br/>

### ⚠️ Disclaimer

## This project is for educational and research purposes only.
## It is not financial advice and should not be used with real money without extensive testing.

<br/>

### 👨‍💻 Author

## Built with ❤️ by Kuboja Daniel

## If you like this project, feel free to ⭐ the repository!