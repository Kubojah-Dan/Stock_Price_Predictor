import os
from pandas_datareader import data as pdr
import pandas as pd
import datetime

FRED_SERIES = {
    "CPIAUCSL": "CPI",
    "UNRATE": "UNRATE",
    "FEDFUNDS": "FFR",
    "GS10": "10Y_Treasury"
}

def fetch_macro(start_date="2000-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.date.today().isoformat()

    out = pd.DataFrame()
    for series, colname in FRED_SERIES.items():
        try:
            s = pdr.DataReader(series, "fred", start_date, end_date)
            s = s.rename(columns={series: colname})
            out = pd.concat([out, s], axis=1)
        except Exception as e:
            print(f"Warning: cannot fetch {series}: {e}")
            continue

    if out.empty:
        return out

    out = out.ffill().asfreq("D").ffill()
    return out
