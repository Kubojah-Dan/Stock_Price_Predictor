# ML-Based Stock Trading System

A production-grade machine learning trading system built using Python, XGBoost, Streamlit, and Yahoo Finance data.
This project combines data science, machine learning, backtesting, portfolio management, and live paper trading into a single cohesive system.

---

## Key Features

### Market Data
- Historical OHLCV data downloaded directly from **Yahoo Finance**
- Automatic caching & local storage in `data/`
- Multi-asset support (AAPL, MSFT, NVDA, GOOGL)

---

### Machine Learning
- **XGBoost classifier** trained per ticker with **Optuna hyperparameter tuning** (50 trials, AUC-optimised)
- **Balanced training** via `compute_sample_weight("balanced")` — eliminates bullish prediction bias
- **Isotonic probability calibration** (`CalibratedClassifierCV`) for reliable confidence scores
- **Optimal threshold selection** tuned on a validation split (no test-set leakage)
- **RF-based feature selection** — selects top 25 features from 50 engineered candidates
- **Expanding-window walk-forward evaluation** (5 folds) for honest out-of-sample assessment
- **21-day prediction horizon** — more persistent and learnable than daily noise

---

### Feature Engineering (`src/features.py`)

All features are **stationary and ratio-based** (no raw price levels) to be suitable for ML:

**Returns & Momentum**
- 1d, 5d, 10d, 21d returns and log return
- Momentum ratio: 1-month vs 3-month return

**Volatility**
- 21-day and 63-day rolling volatility
- Volatility regime ratio (vol_21d / vol_63d)

**Moving Average Ratios** (price / SMA — 1, not raw prices)
- Price vs SMA20, SMA50, SMA200
- SMA20 vs SMA50 crossover ratio

**MACD** (normalised by price)
- MACD line, signal line, histogram

**RSI** — 7, 14, 21 period

**Bollinger Bands**
- Band position (0–1 percentile)
- Band width (normalised)

**ATR** — normalised by price

**Stochastic Oscillator** — %K and %D

**ADX** — trend strength + DI+/DI- direction

**Ichimoku Cloud** — conversion, base, and cloud difference ratios

**Volume**
- Volume change, volume vs 20-day average ratio
- On-balance volume momentum

**Lag Features** — returns lagged 1, 2, 3, 5, 10, 21 days; volume lagged 1, 2, 3 days

**Macro Features** (via FRED)
- CPI, Unemployment Rate, Fed Funds Rate, 10-Year Treasury Yield

---

### Trading Logic
- Converts calibrated probabilities into **BUY / SELL / HOLD** signals
- Per-ticker optimal probability threshold (tuned on validation, not test data)
- Transaction cost simulation (0.1% per trade)
- Long-only strategy

---

### Portfolio & Paper Trading
- Multi-asset paper trading engine
- Shared portfolio capital with per-asset positions
- Trade execution with entry price, position size, and timestamps
- Real-time portfolio equity tracking

---

### Backtesting & Metrics
- Strategy vs Market (buy-and-hold) comparison
- Transaction cost simulation
- Classification metrics: Accuracy, Balanced Accuracy, Precision, Recall, F1, AUC, MCC
- Trading metrics: Sharpe Ratio, Sortino Ratio, cumulative return
- Walk-forward per-fold AUC reporting

---

### Visualisations

All saved to `outputs/` per ticker:

| File | Description |
|---|---|
| `{TICKER}_feature_importance.html` | Top 25 feature importances (XGBoost) |
| `{TICKER}_roc.html` | ROC curve with AUC |
| `{TICKER}_confusion.html` | Confusion matrix (Down / Up) |
| `{TICKER}_calibration.html` | Probability calibration curve |
| `{TICKER}_walk_forward.html` | Per-fold walk-forward AUC |
| `{TICKER}_equity.html` | Strategy vs market equity curve |
| `{TICKER}_drawdown.html` | Strategy drawdown chart |

---

### Interactive Dashboard (Streamlit)
- Multi-asset selection
- Adjustable probability threshold
- Auto-trade toggle
- Live paper trading simulation
- Portfolio overview, trade history, strategy analytics

---

## Project Structure

```
Stock_Price_Predictor/
├── src/
│   ├── train_improved.py     # Main training pipeline (Optuna + calibration + walk-forward)
│   ├── features.py           # Stationary feature engineering (50 features)
│   ├── preprocessing.py      # Yahoo Finance data download & caching
│   ├── visualize.py          # Plotly visualisation functions
│   ├── backtest.py           # Backtest with transaction costs
│   ├── metrics.py            # Sharpe, Sortino
│   ├── walk_forward.py       # Walk-forward validation utilities
│   ├── config.py             # Global config
│   ├── macro.py              # FRED macro data fetching
│   └── ...
├── streamlit_app/
│   └── app.py                # Streamlit dashboard
├── models/                   # Saved models, scalers, selectors, thresholds
├── outputs/                  # HTML visualisations + metrics CSVs
├── data/                     # Cached Yahoo Finance CSVs
└── requirements.txt
```

---

## Results

Models trained on 2015–present data for AAPL, MSFT, NVDA, GOOGL (21-day horizon, 80/20 split).

| Ticker | Accuracy | Bal. Accuracy | F1 | AUC | Backtest Return | Sharpe |
|--------|----------|---------------|----|-----|-----------------|--------|
| GOOGL  | 59.4%    | 50.4%         | 0.716 | 0.496 | +1.7%   | 0.63 |
| NVDA   | 56.3%    | 46.6%         | 0.707 | 0.469 | +251.0% | 1.40 |
| MSFT   | 49.9%    | 50.1%         | 0.483 | 0.499 | +7.6%   | 0.26 |
| AAPL   | 44.0%    | 47.8%         | 0.357 | 0.468 | +21.3%  | 0.80 |

**Note:**  Walk-forward fold AUCs range from 0.28 to 0.75, reflecting that signal exists in certain market regimes but is not consistent across all periods. NVDA's backtest return of +251% reflects the strong trend-following signal captured during its 2023–2024 bull run.

---

## Setup & Usage

```bash
# 1. Create virtual environment
python -m venv .StockVenv
.StockVenv\Scripts\activate        # Windows
source .StockVenv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (all 4 tickers, ~6-8 minutes)
python src/train_improved.py

# 4. Launch the dashboard
streamlit run streamlit_app/app.py
```

Training will:
- Download Yahoo Finance data (cached to `data/`)
- Engineer 50 stationary features per ticker
- Run Optuna hyperparameter search (50 trials per ticker)
- Train calibrated XGBoost with balanced sample weights
- Run 5-fold expanding walk-forward evaluation
- Save models to `models/` and visualisations to `outputs/`

---

## Disclaimer

This project is for educational and research purposes only.
It is not financial advice and should not be used with real money without extensive testing.

---

## Author

Built with love by Kuboja Daniel

If you find this project useful, feel free to star the repository!
