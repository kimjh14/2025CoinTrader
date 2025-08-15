"""
python tools/build_dataset.py --in data/krw_btc_1m.parquet --out data/dataset_mtf.parquet --horizon 20 --up 0.003 --dn -0.003 --tfs 3,5
"""
import argparse
import os, sys
import pandas as pd
import numpy as np
from typing import List

# ---- ensure project root on sys.path so "ml" is importable ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Try to import indicators if present; otherwise use local fallbacks
try:
    from ml.indicators import ema as _ema, rsi as _rsi, macd as _macd
    HAVE_ML = True
except Exception:
    HAVE_ML = False

def ema(series: pd.Series, period: int) -> pd.Series:
    if HAVE_ML:
        return _ema(series, period)
    return series.ewm(span=period, adjust=False, min_periods=1).mean()

def rsi(series: pd.Series, period: int=14) -> pd.Series:
    if HAVE_ML:
        return _rsi(series, period)
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast:int=12, slow:int=26, signal:int=9):
    if HAVE_ML:
        return _macd(series, fast, slow, signal)
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, period:int=20, stds:float=2.0):
    ma = series.rolling(window=period, min_periods=1).mean()
    sd = series.rolling(window=period, min_periods=1).std(ddof=0)
    upper = ma + stds*sd
    lower = ma - stds*sd
    width = (upper - lower) / ma.replace(0, np.nan)
    return ma, upper, lower, width

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period:int=14):
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

# -----------------------
# Feature builders
# -----------------------
def add_features_ohlcv(df: pd.DataFrame, prefix: str="") -> pd.DataFrame:
    """
    df: index=timestamp(UTC), cols=[open,high,low,close,volume,value]
    prefix: feature name prefix (e.g., "m1_", "m3_", ...)
    """
    out = df.copy()

    # returns & vol
    out[f"{prefix}ret_1"]  = out["close"].pct_change()
    out[f"{prefix}ret_5"]  = out["close"].pct_change(5)
    out[f"{prefix}ret_15"] = out["close"].pct_change(15)
    out[f"{prefix}vol_20"] = out[f"{prefix}ret_1"].rolling(20, min_periods=5).std(ddof=0)
    out[f"{prefix}vol_60"] = out[f"{prefix}ret_1"].rolling(60, min_periods=10).std(ddof=0)

    # EMA / MACD / RSI
    out[f"{prefix}ema_12"] = ema(out["close"], 12)
    out[f"{prefix}ema_26"] = ema(out["close"], 26)
    m, s, h = macd(out["close"], 12, 26, 9)
    out[f"{prefix}macd"] = m
    out[f"{prefix}macd_sig"] = s
    out[f"{prefix}macd_hist"] = h
    out[f"{prefix}rsi_14"] = rsi(out["close"], 14)

    # Bollinger
    bb_ma, bb_up, bb_dn, bb_width = bollinger(out["close"], 20, 2.0)
    out[f"{prefix}bb_ma"]    = bb_ma
    out[f"{prefix}bb_up"]    = bb_up
    out[f"{prefix}bb_dn"]    = bb_dn
    out[f"{prefix}bb_width"] = bb_width

    # ATR
    out[f"{prefix}atr_14"] = atr(out["high"], out["low"], out["close"], 14)

    # Price position features
    out[f"{prefix}dist_bb_up"] = (out[f"{prefix}bb_up"] - out["close"]) / out[f"{prefix}bb_up"].replace(0, np.nan)
    out[f"{prefix}dist_bb_dn"] = (out["close"] - out[f"{prefix}bb_dn"]) / out[f"{prefix}bb_dn"].replace(0, np.nan)
    out[f"{prefix}ema_gap"]    = (out[f"{prefix}ema_12"] - out[f"{prefix}ema_26"]) / out[f"{prefix}ema_26"].replace(0, np.nan)

    return out

def resample_ohlcv(df_1m: pd.DataFrame, m: int) -> pd.DataFrame:
    """Resample 1m to m-minute OHLCV (right-closed)."""
    rule = f"{m}min"  # <-- 'T' 대신 'min'
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "value": "sum",
    }
    r = df_1m.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["close"])
    return r

# -----------------------
# Original 1m feature set
# -----------------------
def add_features_1m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    # returns
    df["ret_1"]     = df["close"].pct_change()
    df["logret_1"]  = np.log(df["close"]).diff()
    df["ret_5"]     = df["close"].pct_change(5)
    df["ret_15"]    = df["close"].pct_change(15)
    df["vol_20"]    = df["ret_1"].rolling(20, min_periods=5).std(ddof=0)
    df["vol_60"]    = df["ret_1"].rolling(60, min_periods=10).std(ddof=0)

    # EMA / MACD / RSI
    df["ema_12"]    = ema(df["close"], 12)
    df["ema_26"]    = ema(df["close"], 26)
    macd_line, macd_sig, macd_hist = macd(df["close"], 12, 26, 9)
    df["macd"]      = macd_line
    df["macd_sig"]  = macd_sig
    df["macd_hist"] = macd_hist
    df["rsi_14"]    = rsi(df["close"], 14)

    # Bollinger
    bb_ma, bb_up, bb_dn, bb_width = bollinger(df["close"], 20, 2.0)
    df["bb_ma"]     = bb_ma
    df["bb_up"]     = bb_up
    df["bb_dn"]     = bb_dn
    df["bb_width"]  = bb_width

    # ATR
    df["atr_14"]    = atr(df["high"], df["low"], df["close"], 14)

    # Price position
    df["dist_bb_up"] = (df["bb_up"] - df["close"]) / df["bb_up"].replace(0, np.nan)
    df["dist_bb_dn"] = (df["close"] - df["bb_dn"]) / df["bb_dn"].replace(0, np.nan)
    df["ema_gap"]    = (df["ema_12"] - df["ema_26"]) / df["ema_26"].replace(0, np.nan)

    return df

# -----------------------
# Labeling
# -----------------------
def add_labels(df: pd.DataFrame, horizon:int=20, up:float=0.003, dn:float=-0.003):
    """
    horizon: minutes ahead to evaluate future return
    up/dn:   thresholds (e.g., +0.3% / -0.3%)
    label:
        1  if future return >= up
        -1 if future return <= dn
        0  otherwise
    """
    future = df["close"].shift(-horizon)
    df["fwd_ret"] = (future - df["close"]) / df["close"]
    cond_up = df["fwd_ret"] >= up
    cond_dn = df["fwd_ret"] <= dn
    df["label"] = 0
    df.loc[cond_up, "label"] = 1
    df.loc[cond_dn, "label"] = -1
    # Drop the last tail where future is not available
    df = df.iloc[:-horizon].copy()
    return df

# -----------------------
# Build
# -----------------------
def parse_tfs(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def build(inp_path: str, out_path: str, horizon:int, up:float, dn:float, tfs: List[int]):
    df = pd.read_parquet(inp_path)
    if "timestamp" not in df.columns:
        raise ValueError("Input parquet must contain 'timestamp' column (UTC).")

    # Basic cleanup
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 1분 기준 타임라인 준비 (UTC aware index)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    base_1m = df.set_index(ts)[["open","high","low","close","volume","value"]].copy()

    # 1분 지표 (기존과 동일)
    feat_1m = add_features_1m(df.copy())

    # 멀티 타임프레임 지표 추가 (선택)
    if tfs:
        merged = feat_1m.copy()
        # 왼쪽 키: 1분 타임라인
        merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
        left_df = merged[["timestamp"]].copy().sort_values("timestamp").reset_index(drop=True)

        # 오른쪽: 각 TF 지표를 'ts'로 정렬해서 asof 매칭
        for m in tfs:
            tf = resample_ohlcv(base_1m, m)
            tf_feat = add_features_ohlcv(tf, prefix=f"m{m}_")
            right_df = tf_feat.copy()
            right_df["ts"] = right_df.index.tz_convert("UTC")
            right_df = right_df.sort_index()
            right_df = right_df.reset_index(drop=True)

            # asof 머지
            aligned = pd.merge_asof(
                left_df.sort_values("timestamp"),
                right_df.sort_values("ts"),
                left_on="timestamp",
                right_on="ts",
                direction="backward",
                allow_exact_matches=True,
            )

            # 불필요 컬럼 제거(open/high/low/close/volume/value, ts 등)
            aligned = aligned.drop(columns=["open","high","low","close","volume","value","ts"], errors="ignore")

            # 병합
            merged = pd.concat([merged.reset_index(drop=True), aligned.drop(columns=["timestamp"], errors="ignore")], axis=1)
    else:
        merged = feat_1m

    # Warmup (가장 긴 lookback 고려: EMA/Bollinger 등)
    warmup = 200
    merged = merged.iloc[warmup:].copy()

    # 레이블 (1분 close 기준)
    merged = add_labels(merged, horizon=horizon, up=up, dn=dn)

    # 최종 NA 최소 정리
    keep_core = ["ret_1","ema_12","ema_26","macd","rsi_14","bb_ma","atr_14","fwd_ret","label"]
    core_exists = [c for c in keep_core if c in merged.columns]
    if core_exists:
        merged = merged.dropna(subset=core_exists)

    # 저장
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print("dataset rows:", len(merged))
    if len(merged):
        prop = (merged["label"].value_counts(normalize=True).sort_index()
                .rename({-1:"short",0:"flat",1:"long"}))
        print(prop)
    else:
        print("Empty dataset after processing. Consider reducing horizon/thresholds or adjusting tfs.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input parquet from tools/collect.py (1m OHLCV)")
    ap.add_argument("--out", dest="out", required=True, help="output dataset parquet")
    ap.add_argument("--horizon", type=int, default=20, help="label horizon in minutes (default: 20)")
    ap.add_argument("--up", type=float, default=0.003, help="upper threshold for label (default: 0.3%)")
    ap.add_argument("--dn", type=float, default=-0.003, help="lower threshold for label (default: -0.3%)")
    ap.add_argument("--tfs", type=str, default="", help="comma-separated higher TF minutes (e.g., '3,5,15')")
    args = ap.parse_args()

    tfs = parse_tfs(args.tfs)
    if tfs:
        print(f"[build] multi-TF enabled: {tfs}")
    else:
        print("[build] single-TF (1m) only")

    build(args.inp, args.out, args.horizon, args.up, args.dn, tfs)

if __name__ == "__main__":
    main()
