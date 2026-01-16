#from visualize import plot_equity_curve

def backtest_with_costs(df, signals, cost=0.001):
    df = df.iloc[-len(signals):].copy()
    df["signal"] = signals
    df["returns"] = df["Close"].pct_change().fillna(0)

    trades = df["signal"].diff().abs().fillna(0)
    df["strategy"] = (
        df["signal"].shift(1) * df["returns"]
        - trades * cost
    )

    df["cum_strategy"] = (1 + df["strategy"]).cumprod()
    df["cum_market"] = (1 + df["returns"]).cumprod()
    
    #plot_equity_curve(bt)

    return df
