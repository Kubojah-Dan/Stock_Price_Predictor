import yfinance as yf
import pandas as pd
import os

def load_and_save_yahoo(ticker, start="2015-01-01", data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)

    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if df.empty:
        raise RuntimeError("Yahoo Finance returned no data")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df["Ticker"] = ticker
    df["Date"] = pd.to_datetime(df["Date"])

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    path = f"{data_dir}/{ticker}_yahoo.csv"
    df.to_csv(path, index=False)

    print(f"✔ Yahoo data saved to {path}")
    return df
