import pandas as pd
import numpy as np

def ema(s: pd.Series, span: int):
    return s.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int=14):
    """Wilder's RSI - 업비트/TradingView 표준"""
    delta = close.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Wilder's Smoothing (RMA)
    avg_gains = gains.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_losses = losses.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    rs = avg_gains / (avg_losses + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df: pd.DataFrame):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period:int=14):
    """Wilder's ATR"""
    tr = true_range(df)
    # Wilder's smoothing
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def bollinger(close: pd.Series, period:int=20, nstd:float=2.0):
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=1)  # 표본 표준편차 사용
    upper = ma + nstd*sd
    lower = ma - nstd*sd
    width = (upper - lower) / (ma + 1e-9)
    return ma, upper, lower, width

def vwap(df: pd.DataFrame, period:int=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"]
    vwap = (tp * vol).rolling(period).sum() / (vol.rolling(period).sum() + 1e-9)
    return vwap

def adx_simplified(df: pd.DataFrame, period:int=14):
    # simple/fast proxy (not exact Welles Wilder), good enough for regime split
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up>dn) & (up>0), up, 0.0)
    minus_dm = np.where((dn>up) & (dn>0), dn, 0.0)
    tr = (pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1))
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / (atr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / (atr + 1e-9)
    dx = ( (plus_di - minus_di).abs() / ((plus_di + minus_di)+1e-9) ) * 100
    adx = dx.rolling(period).mean()
    return adx