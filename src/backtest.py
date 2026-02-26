def backtest_with_costs(df, signals, cost=0.001):
    """
    df: DataFrame (should contain Close and Date)
    signals: numpy array or series with same length as df
    Returns df with strategy, cum_strategy, cum_market
    """
    df2 = df.copy()
    # if df has Date column (reset_index earlier), ensure ordering
    if "Date" in df2.columns:
        df2 = df2.sort_values("Date").reset_index(drop=True)
    else:
        # preserve chronological order and expose Date column for plotting
        df2 = df2.reset_index(drop=False).rename(columns={"index": "Date"})
    df2 = df2.iloc[-len(signals):].copy()
    df2["signal"] = signals
    df2["returns"] = df2["Close"].pct_change().fillna(0)
    trades = df2["signal"].diff().abs().fillna(0)
    df2["strategy"] = (df2["signal"].shift(1).fillna(0) * df2["returns"]) - trades * cost
    df2["cum_strategy"] = (1 + df2["strategy"]).cumprod()
    df2["cum_market"] = (1 + df2["returns"]).cumprod()
    return df2
