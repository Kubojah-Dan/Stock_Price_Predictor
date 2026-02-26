import numpy as np
import pandas as pd
import datetime

# Support both package (src.*) and script execution from repo root
try:
    from src.macro import fetch_macro, FRED_SERIES
except ImportError:
    from macro import fetch_macro, FRED_SERIES  # type: ignore

try:
    from src.sentiment import daily_sentiment_for_ticker
except ImportError:
    from sentiment import daily_sentiment_for_ticker  # type: ignore

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df):
    df = df.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"]).diff()
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_200"] = df["Close"].rolling(200).mean()
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df["vol_change"] = df["Volume"].pct_change()
    # Bollinger Bands
    rolling_20 = df["Close"].rolling(20)
    bb_mid = rolling_20.mean()
    bb_std = rolling_20.std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    # Volatility & momentum
    df["volatility_20"] = df["return_1d"].rolling(20).std()
    df["momentum_10"] = df["Close"].pct_change(10)

    # --- New Features ---
    # Ichimoku Cloud
    high_9 = df["High"].rolling(window=9).max()
    low_9 = df["Low"].rolling(window=9).min()
    df["ichimoku_conv"] = (high_9 + low_9) / 2

    high_26 = df["High"].rolling(window=26).max()
    low_26 = df["Low"].rolling(window=26).min()
    df["ichimoku_base"] = (high_26 + low_26) / 2

    df["ichimoku_span_a"] = ((df["ichimoku_conv"] + df["ichimoku_base"]) / 2).shift(26)
    
    high_52 = df["High"].rolling(window=52).max()
    low_52 = df["Low"].rolling(window=52).min()
    df["ichimoku_span_b"] = ((high_52 + low_52) / 2).shift(26)

    # Stochastic Oscillator
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["stoch_k"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14 + 1e-9))
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # ADX (Trend Strength)
    plus_dm = df["High"].diff()
    minus_dm = df["Low"].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr_s = tr.rolling(window=14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).sum() / (tr_s + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).sum() / (tr_s + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    df["adx"] = dx.rolling(window=14).mean()

    # Lag features
    for lag in [1, 2, 3, 5, 21]:
        df[f"ret_lag_{lag}"] = df["return_1d"].shift(lag)
        df[f"vol_lag_{lag}"] = df["Volume"].pct_change().shift(lag)

    return df

def add_features(df, ticker=None, start_date=None):
    """
    Produce dataframe with technical indicators + macro + sentiment.
    Preserves datetime index.
    """
    df = df.copy()
    # Ensure Date index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        df.index = pd.to_datetime(df.index)

    # Drop weekends (user requirement)
    df = df[df.index.dayofweek < 5]

    # Basic forward/backward fill for missed days
    df = df.sort_index().ffill().bfill()

    # Technical indicators
    df = add_technical_indicators(df)

    # Macro features (daily, reindexed to price dates)
    macro_cols = [f"macro_{v}" for v in FRED_SERIES.values()]
    try:
        s_date = start_date if start_date else df.index.min().date()
        macro = fetch_macro(start_date=s_date)
        if macro is None or macro.empty:
            macro = pd.DataFrame(index=df.index)
        macro = macro.reindex(df.index).ffill().bfill().fillna(0.0)
        macro = macro.add_prefix("macro_")
        for col in macro_cols:
            if col not in macro.columns:
                macro[col] = 0.0
        macro = macro[macro_cols]
        df = pd.concat([df, macro], axis=1)
    except Exception as e:
        print("Macro fetch failed:", e)
        # still ensure macro columns exist to keep feature set stable
        for col in macro_cols:
            df[col] = 0.0

    # Sentiment per day (aggregate) - safe fallback when API missing
    if ticker is not None:
        sentiments = []
        for d in df.index.date:
            try:
                s = daily_sentiment_for_ticker(ticker, d)
            except Exception:
                s = 0.0
            sentiments.append(s)
        df["sentiment"] = sentiments
    else:
        df["sentiment"] = 0.0

    # Final cleaning: ensure NO NaNs remain
    df = df.ffill().bfill().fillna(0.0)
    return df


def create_target(df, horizon=1):
    """
    Create targets:
      - target_price (price after `horizon` days)
      - target_return (relative return over horizon)
      - target (direction binary)
    """
    df = df.copy()
    # ensure index is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    df["target_price"] = df["Close"].shift(-horizon)
    df["target_return"] = (df["target_price"] - df["Close"]) / df["Close"]
    df["target"] = (df["target_return"] > 0).astype(int)
    df.dropna(inplace=True)

    # keep Date as a column for downstream plotting / time tracking
    df["Date"] = df.index
    df.index.name = None
    return df
