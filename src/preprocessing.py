import os
import yfinance as yf
import pandas as pd


def _load_cached(path):
    if os.path.exists(path):
        df_cached = pd.read_csv(path, parse_dates=["Date"])
        # Ensure expected columns exist
        if not df_cached.empty and "Close" in df_cached.columns:
            print(f"Using cached data at {path}")
            return df_cached
    return None


def load_and_save_yahoo(ticker, start="2015-01-01", data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    path = f"{data_dir}/{ticker}_yahoo.csv"

    try:
        df = yf.download(
            ticker,
            start=start,
            auto_adjust=True,
            progress=False,
            group_by="column"
        )
    except Exception as e:
        df = None
        dl_error = e
    else:
        dl_error = None

    # Handle empty/failed download
    if df is None or getattr(df, "empty", True):
        cached = _load_cached(path)
        if cached is not None:
            return cached
        msg = f"Yahoo Finance returned no data for {ticker}"
        if dl_error:
            msg += f": {dl_error}"
        raise RuntimeError(msg)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df["Ticker"] = ticker
    df["Date"] = pd.to_datetime(df["Date"])

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(path, index=False)
    print(f"Yahoo data saved to {path}")
    return df
