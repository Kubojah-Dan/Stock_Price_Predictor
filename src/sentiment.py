import os
import datetime
import pandas as pd
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    from src.sentiment_cache import get_cached, set_cached
except ImportError:
    from sentiment_cache import get_cached, set_cached  # type: ignore
from dotenv import load_dotenv

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", None)
analyzer = SentimentIntensityAnalyzer()
newsapi_client = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None

MAX_LOOKBACK_DAYS = int(os.getenv("SENTIMENT_LOOKBACK_DAYS", "20"))
_rate_limited = False
_rate_limit_logged = False

def _mark_rate_limited():
    global _rate_limited, _rate_limit_logged
    _rate_limited = True
    if not _rate_limit_logged:
        print("NewsAPI rate limit reached. Sentiment fetch disabled for this run.")
        _rate_limit_logged = True

def fetch_headlines(ticker, from_dt, to_dt):
    if NEWSAPI_KEY is None or newsapi_client is None or _rate_limited:
        return []

    try:
        res = newsapi_client.get_everything(
            q=ticker,
            from_param=from_dt.isoformat(),
            to=to_dt.isoformat(),
            language="en",
            sort_by="relevancy",
            page_size=100
        )
        if isinstance(res, dict) and res.get("status") == "error":
            if str(res.get("code", "")).strip() == "rateLimited":
                _mark_rate_limited()
            else:
                print("NewsAPI fetch failed:", res)
            return []
        articles = res.get("articles", [])
        return [a.get("title") or "" for a in articles]
    except Exception as e:
        if "rateLimited" in str(e):
            _mark_rate_limited()
        else:
            print("NewsAPI fetch failed:", e)
        return []

def daily_sentiment_for_ticker(ticker, date):
    """
    Returns sentiment score.
    Only fetches for recent dates to avoid rate limit.
    """

    today = datetime.date.today()

    # 🔴 Skip old dates entirely
    if (today - date).days > MAX_LOOKBACK_DAYS:
        return 0.0

    key = f"{ticker}_{date}"
    cached = get_cached(key)
    if cached is not None:
        return cached

    if NEWSAPI_KEY is None or _rate_limited:
        return 0.0

    from_dt = datetime.datetime.combine(date, datetime.time(0, 0))
    to_dt = from_dt + datetime.timedelta(days=1)

    headlines = fetch_headlines(ticker, from_dt, to_dt)

    if not headlines:
        score = 0.0
        set_cached(key, score)
        return score

    scores = [
        analyzer.polarity_scores(h)["compound"]
        for h in headlines if h.strip()
    ]

    score = float(pd.Series(scores).mean()) if scores else 0.0
    set_cached(key, score)
    return score
