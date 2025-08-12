import pandas as pd
import numpy as np
from typing import List, Tuple

FEATURES = [
    # price/returns
    "ret_1", "ret_3", "ret_5", "ret_10",
    # trend
    "ema_9", "ema_21", "ema_50", "ema_200",
    "macd", "macd_sig", "macd_hist",
    # volatility
    "atr", "bb_mid", "bb_high", "bb_low", "bb_width",
    # momentum
    "rsi",
    # volume
    "vol_z",
    # value-ish
    "vwap_dev",
    # regime-ish
    "adx",  # simple proxy (see indicators)
]

TARGET = "label_3c"  # {-1, 0, +1}
PROBA_COLS = ["proba_dn","proba_ne","proba_up"]

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = np.log(df["close"]).diff(1)
    df["ret_3"] = np.log(df["close"]).diff(3)
    df["ret_5"] = np.log(df["close"]).diff(5)
    df["ret_10"] = np.log(df["close"]).diff(10)
    return df

def zscore(s: pd.Series, win:int=20):
    return (s - s.rolling(win, min_periods=win).mean()) / (s.rolling(win, min_periods=win).std() + 1e-9)

def finalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # dropna and keep necessary columns
    cols = ["timestamp","open","high","low","close","volume"] + FEATURES + [TARGET, "tp_up","tp_dn","horizon_end"]
    df = df.dropna().reset_index(drop=True)
    return df