import numpy as np

def detect_regime(df, window=50):
    slope = np.polyfit(
        range(window),
        df["Close"].tail(window),
        1
    )[0]

    volatility = df["Close"].pct_change().rolling(window).std().iloc[-1]

    if abs(slope) > volatility:
        return "TREND"
    else:
        return "RANGE"
