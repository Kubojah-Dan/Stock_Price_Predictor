import numpy as np

def add_features(df):
    df = df.copy()

    df["return_1d"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()

    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["macd"] = df["ema_12"] - df["ema_26"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["vol_change"] = df["Volume"].pct_change()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def create_target(df):
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df

    