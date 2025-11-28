import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

def _to_list_safe(series: pd.Series) -> list:
    values = series.tolist()
    out = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            try:
                fv = float(v)
                if np.isnan(fv) or np.isinf(fv):
                    out.append(None)
                else:
                    out.append(fv)
            except Exception:
                out.append(None)
    return out

def compute_indicators_bundle(df: pd.DataFrame) -> dict:
    close = df["Close"]
    volume = df["Volume"]

    sma20 = SMAIndicator(close, window=20).sma_indicator()
    sma50 = SMAIndicator(close, window=50).sma_indicator()
    sma200 = SMAIndicator(close, window=200).sma_indicator()
    ema20 = EMAIndicator(close, window=20).ema_indicator()
    rsi14 = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    macd_line = macd.macd()
    macd_signal = macd.macd_signal()
    macd_hist = macd.macd_diff()
    bb = BollingerBands(close, window=20, window_dev=2)
    bb_h = bb.bollinger_hband()
    bb_l = bb.bollinger_lband()

    out = {
        "sma20": _to_list_safe(sma20),
        "sma50": _to_list_safe(sma50),
        "sma200": _to_list_safe(sma200),
        "ema20": _to_list_safe(ema20),
        "rsi14": _to_list_safe(rsi14),
        "macd": _to_list_safe(macd_line),
        "macd_signal": _to_list_safe(macd_signal),
        "macd_hist": _to_list_safe(macd_hist),
        "bb_upper": _to_list_safe(bb_h),
        "bb_lower": _to_list_safe(bb_l),
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
    }
    return out

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    xdf = df.copy()
    xdf["sma20"] = SMAIndicator(xdf["Close"], window=20).sma_indicator()
    xdf["sma50"] = SMAIndicator(xdf["Close"], window=50).sma_indicator()
    xdf["sma200"] = SMAIndicator(xdf["Close"], window=200).sma_indicator()
    xdf["ema20"] = EMAIndicator(xdf["Close"], window=20).ema_indicator()
    xdf["rsi14"] = RSIIndicator(xdf["Close"], window=14).rsi()
    macd = MACD(xdf["Close"]) 
    xdf["macd"] = macd.macd()
    xdf["macd_signal"] = macd.macd_signal()
    bb = BollingerBands(xdf["Close"], window=20, window_dev=2)
    xdf["bb_upper"] = bb.bollinger_hband()
    xdf["bb_lower"] = bb.bollinger_lband()
    # Lag features
    for l in [1,2,3]:
        xdf[f"lag_close_{l}"] = xdf["Close"].shift(l)
    # Temporal features
    xdf["day"] = xdf.index.day
    xdf["month"] = xdf.index.month
    xdf["quarter"] = xdf.index.quarter
    return xdf