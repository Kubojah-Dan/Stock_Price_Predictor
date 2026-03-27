import numpy as np
import pandas as pd

try:
    from src.macro import fetch_macro, FRED_SERIES
except ImportError:
    from macro import fetch_macro, FRED_SERIES  # type: ignore


def _rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    return 100 - (100 / (1 + up / (down + 1e-9)))


def add_technical_indicators(df):
    c = df["Close"]
    h, l, v = df["High"], df["Low"], df["Volume"]

    # --- Returns (stationary) ---
    df["ret_1d"] = c.pct_change()
    df["ret_5d"] = c.pct_change(5)
    df["ret_10d"] = c.pct_change(10)
    df["ret_21d"] = c.pct_change(21)
    df["log_ret"] = np.log(c / c.shift(1))

    # --- Volatility ---
    df["vol_21d"] = df["ret_1d"].rolling(21).std()
    df["vol_63d"] = df["ret_1d"].rolling(63).std()
    df["vol_ratio"] = df["vol_21d"] / (df["vol_63d"] + 1e-9)  # vol regime

    # --- Moving average RATIOS (stationary, not raw prices) ---
    df["price_sma20_ratio"] = c / (c.rolling(20).mean() + 1e-9) - 1
    df["price_sma50_ratio"] = c / (c.rolling(50).mean() + 1e-9) - 1
    df["price_sma200_ratio"] = c / (c.rolling(200).mean() + 1e-9) - 1
    df["sma20_sma50_ratio"] = c.rolling(20).mean() / (c.rolling(50).mean() + 1e-9) - 1

    # --- EMA crossover ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = (ema12 - ema26) / (c + 1e-9)          # normalised
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- RSI ---
    df["rsi_14"] = _rsi(c, 14)
    df["rsi_7"] = _rsi(c, 7)
    df["rsi_21"] = _rsi(c, 21)

    # --- Bollinger Band position (0-1) ---
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_pct"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)
    df["bb_width"] = (4 * bb_std) / (bb_mid + 1e-9)

    # --- ATR (normalised) ---
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr_pct"] = tr.rolling(14).mean() / (c + 1e-9)

    # --- Stochastic ---
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # --- ADX ---
    plus_dm = np.where((h.diff() > -l.diff()) & (h.diff() > 0), h.diff(), 0.0)
    minus_dm = np.where((-l.diff() > h.diff()) & (-l.diff() > 0), -l.diff(), 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=c.index).rolling(14).sum() / (tr14 + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=c.index).rolling(14).sum() / (tr14 + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["adx"] = dx.rolling(14).mean()
    df["plus_di_minus_di"] = plus_di - minus_di   # direction of trend

    # --- Ichimoku ratios ---
    conv = (h.rolling(9).max() + l.rolling(9).min()) / 2
    base = (h.rolling(26).max() + l.rolling(26).min()) / 2
    df["ichimoku_conv_ratio"] = conv / (c + 1e-9) - 1
    df["ichimoku_base_ratio"] = base / (c + 1e-9) - 1
    df["ichimoku_cb_diff"] = (conv - base) / (c + 1e-9)

    # --- Volume features ---
    df["vol_change"] = v.pct_change()
    df["vol_sma20_ratio"] = v / (v.rolling(20).mean() + 1e-9) - 1
    df["obv_change"] = (np.sign(df["ret_1d"]) * v).rolling(10).sum() / (v.rolling(10).sum() + 1e-9)

    # --- Lag features ---
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"ret_lag_{lag}"] = df["ret_1d"].shift(lag)
    for lag in [1, 2, 3]:
        df[f"vol_lag_{lag}"] = df["vol_change"].shift(lag)

    # --- Higher-timeframe momentum ---
    df["mom_1m_3m"] = df["ret_21d"] / (df["ret_1d"].rolling(63).sum() + 1e-9)

    return df


def add_features(df, ticker=None, start_date=None):
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index)

    df = df[df.index.dayofweek < 5].sort_index().ffill().bfill()
    df = add_technical_indicators(df)

    # Macro features
    macro_cols = [f"macro_{v}" for v in FRED_SERIES.values()]
    try:
        s_date = start_date if start_date else df.index.min().date()
        macro = fetch_macro(start_date=s_date)
        if macro is None or macro.empty:
            macro = pd.DataFrame(index=df.index)
        macro = macro.reindex(df.index).ffill().bfill().fillna(0.0).add_prefix("macro_")
        for col in macro_cols:
            if col not in macro.columns:
                macro[col] = 0.0
        df = pd.concat([df, macro[macro_cols]], axis=1)
    except Exception as e:
        print("Macro fetch failed:", e)
        for col in macro_cols:
            df[col] = 0.0

    # Sentiment (zero-filled when unavailable — avoids slow API calls blocking training)
    for col in ["sentiment", "sentiment_lag1", "sentiment_lag2", "sentiment_sma5", "sentiment_sma10"]:
        df[col] = 0.0

    df = df.ffill().bfill().fillna(0.0)
    return df


def create_target(df, horizon=5):
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    future_ret = df["Close"].pct_change(horizon).shift(-horizon)
    df["target_return"] = future_ret
    # Use a small positive threshold to avoid labelling tiny noise as "Up"
    df["target"] = (future_ret > 0.0).astype(int)
    df.dropna(subset=["target", "target_return"], inplace=True)
    df["Date"] = df.index
    df.index.name = None
    return df
