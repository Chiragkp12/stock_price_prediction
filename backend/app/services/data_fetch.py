import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from typing import Optional
import redis
import json
import requests
import io

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis = None
try:
    _redis = redis.Redis.from_url(REDIS_URL)
    _redis.ping()
except Exception:
    _redis = None

POPULAR_TICKERS = [
    "AAPL","MSFT","GOOG","AMZN","TSLA","META","NFLX","NVDA","AMD","INTC",
    "BA","DIS","V","MA","PYPL","JPM","BAC","WFC","XOM","CVX"
]

def _cache_get(key: str) -> Optional[pd.DataFrame]:
    if not _redis:
        return None
    try:
        raw = _redis.get(key)
        if not raw:
            return None
        data = json.loads(raw)
        df = pd.DataFrame(data)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        return df
    except Exception:
        return None

def _cache_set(key: str, df: pd.DataFrame, ttl: int = 3600):
    if not _redis:
        return
    try:
        payload = df.reset_index().to_dict(orient="records")
        _redis.setex(key, ttl, json.dumps(payload))
    except Exception:
        pass

def search_tickers(query: str) -> list[str]:
    q = query.upper()
    # Simple heuristic: suggest popular tickers matching substring; yfinance has no official search.
    return [t for t in POPULAR_TICKERS if q in t]

def get_historical_prices(ticker: str, years: int = 2) -> Optional[pd.DataFrame]:
    ticker = ticker.upper()
    cache_key = f"prices:{ticker}:{years}y"
    cached = _cache_get(cache_key)
    if cached is not None and not cached.empty:
        return cached

    end = datetime.today()
    start = end - timedelta(days=365 * max(1, years))

    # Prefer yfinance for simplicity and reliability
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, interval='1d')
    except Exception:
        df = pd.DataFrame()
    if df is None or df.empty:
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, interval='1d')
        except Exception:
            df = pd.DataFrame()
    if df is None or df.empty:
        # Fallback: Alpha Vantage if key is set
        key = os.getenv("ALPHAVANTAGE_API_KEY")
        if key:
            try:
                ts = TimeSeries(key, output_format="pandas")
                data, meta = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
                df = data.rename(columns={
                    '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
                    '4. close': 'Close', '6. volume': 'Volume'
                })
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
            except Exception:
                df = pd.DataFrame()

    if df is None or df.empty:
        # Fallback: Stooq CSV (no API key required)
        try:
            stooq_symbol = f"{ticker.lower()}.us"
            url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and resp.text:
                csv_buf = io.StringIO(resp.text)
                sdf = pd.read_csv(csv_buf)
                # Expect columns: Date, Open, High, Low, Close, Volume
                sdf['Date'] = pd.to_datetime(sdf['Date'])
                sdf = sdf[(sdf['Date'] >= pd.to_datetime(start)) & (sdf['Date'] <= pd.to_datetime(end))]
                sdf = sdf.set_index('Date').sort_index()
                df = sdf
            else:
                df = pd.DataFrame()
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return None

    # Normalize columns
    cols = ["Open","High","Low","Close","Adj Close","Volume"]
    # If Adj Close missing, use Close
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    # Some sources use lowercase columns
    for c in list(df.columns):
        if c.lower() == "adj close" and "Adj Close" not in df.columns:
            df["Adj Close"] = df[c]
        if c.lower() == "open" and "Open" not in df.columns:
            df["Open"] = df[c]
        if c.lower() == "high" and "High" not in df.columns:
            df["High"] = df[c]
        if c.lower() == "low" and "Low" not in df.columns:
            df["Low"] = df[c]
        if c.lower() == "close" and "Close" not in df.columns:
            df["Close"] = df[c]
        if c.lower() == "volume" and "Volume" not in df.columns:
            df["Volume"] = df[c]

    df = df[["Open","High","Low","Close","Adj Close","Volume"]].copy()
    df.index.name = "Date"
    df = df.dropna()
    _cache_set(cache_key, df)
    return df