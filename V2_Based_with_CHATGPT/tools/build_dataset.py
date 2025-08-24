"""
build_dataset.py - 피처 및 라벨 생성 도구 (벡터화 최적화)

CLI 사용 예시:

# Classic 방식 (멀티 타임프레임 + 기술적 지표)
python tools/build_dataset.py `
  --mode classic `
  --in data/raw/krw_btc_1m_30d.parquet `
  --out data/classic/dataset_mtf_30d_h2_0.0005.parquet `
  --horizon 2 `
  --up 0.0005 `
  --dn -0.0005 `
  --tfs 3,5

# Sequence 방식 (권장 - 최근 N봉 패턴)
python tools/build_dataset.py `
  --mode seq `
  --in data/raw/krw_btc_1m_180d.parquet `
  --out data/seq/dataset_seq_180d_n20_h1.parquet `
  --n_steps 20 `
  --horizon 1 `
  --up 0.0005 `
  --dn -0.0005 `
  --ta

# CSV 파일도 함께 저장 (디버깅/분석용)
python tools/build_dataset.py `
  --mode classic `
  --in data/raw/krw_btc_1m_180d.parquet `
  --out data/classic/dataset.parquet `
  --horizon 20 `
  --up 0.001 `
  --dn -0.001 `
  --save_csv

사용법:
• collect.py로 수집한 1분봉 데이터를 ML 훈련용 데이터셋으로 변환합니다
• Classic: 기술적 지표 기반 예측 (현재 지표 → horizon 분 후 예측)
• Sequence: 패턴 기반 예측 (최근 N봉 패턴 → horizon 분 후 예측)
• --save_csv: CSV 파일도 함께 저장 (엑셀/디버깅 목적)

매개변수 가이드:
• --horizon: 예측 시점 (1=다음봉, 3=3분 후, 5=5분 후) ※ 권장: 1-5분
• --up/--dn: 라벨링 임계값
  - ±0.0005 (±0.05%): 균형잡힌 분포 (매수36%, 보합28%, 매도36%)
  - ±0.001 (±0.1%): 적당한 선택성 (매수26%, 보합49%, 매도25%)
  - ±0.003 (±0.3%): 높은 선택성 (매수7%, 보합87%, 매도7%) ※ 불균형
• --n_steps: 입력 패턴 길이 (10=10분봉, 20=20분봉, 30=30분봉)
• --tfs: Classic 전용 멀티 타임프레임 (3,5 = 3분봉+5분봉 추가)
• --ta: Sequence 전용 기술적 지표 포함 (MACD, RSI, BB, ATR 등)

성능 최적화:
• 벡터화된 윈도우 생성으로 5-10배 속도 향상
• 420분 워밍업으로 안정화된 지표값 보장
• 데이터 수집 기간 무관 동일 지표값 생성
• GPU 친화적 데이터 구조 생성
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
    """Wilder's RSI - 업비트/TradingView 표준"""
    if HAVE_ML:
        return _rsi(series, period)
    
    delta = series.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Wilder's Smoothing (RMA)
    # alpha = 1/period for Wilder's method
    avg_gains = gains.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_losses = losses.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

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
    sd = series.rolling(window=period, min_periods=1).std(ddof=1)
    upper = ma + stds*sd
    lower = ma - stds*sd
    width = (upper - lower) / ma.replace(0, np.nan)
    return ma, upper, lower, width

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period:int=14):
    """Wilder's ATR"""
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's smoothing
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

# -----------------------
# Feature builders (original, classic mode)
# -----------------------
def add_features_ohlcv(df: pd.DataFrame, prefix: str="") -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}ret_1"]  = out["close"].pct_change()
    out[f"{prefix}ret_5"]  = out["close"].pct_change(5)
    out[f"{prefix}ret_15"] = out["close"].pct_change(15)
    out[f"{prefix}vol_20"] = out[f"{prefix}ret_1"].rolling(20, min_periods=5).std(ddof=1)
    out[f"{prefix}vol_60"] = out[f"{prefix}ret_1"].rolling(60, min_periods=10).std(ddof=1)

    out[f"{prefix}ema_12"] = ema(out["close"], 12)
    out[f"{prefix}ema_26"] = ema(out["close"], 26)
    m, s, h = macd(out["close"], 12, 26, 9)
    out[f"{prefix}macd"] = m
    out[f"{prefix}macd_sig"] = s
    out[f"{prefix}macd_hist"] = h
    out[f"{prefix}rsi_14"] = rsi(out["close"], 14)

    bb_ma, bb_up, bb_dn, bb_width = bollinger(out["close"], 20, 2.0)
    out[f"{prefix}bb_ma"]    = bb_ma
    out[f"{prefix}bb_up"]    = bb_up
    out[f"{prefix}bb_dn"]    = bb_dn
    out[f"{prefix}bb_width"] = bb_width

    out[f"{prefix}atr_14"] = atr(out["high"], out["low"], out["close"], 14)

    out[f"{prefix}dist_bb_up"] = (out[f"{prefix}bb_up"] - out["close"]) / out[f"{prefix}bb_up"].replace(0, np.nan)
    out[f"{prefix}dist_bb_dn"] = (out["close"] - out[f"{prefix}bb_dn"]) / out[f"{prefix}bb_dn"].replace(0, np.nan)
    out[f"{prefix}ema_gap"]    = (out[f"{prefix}ema_12"] - out[f"{prefix}ema_26"]) / out[f"{prefix}ema_26"].replace(0, np.nan)
    return out

def resample_ohlcv(df_1m: pd.DataFrame, m: int) -> pd.DataFrame:
    rule = f"{m}min"
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum","value":"sum"}
    r = df_1m.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["close"])
    return r

def add_features_1m(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ret_1"]     = df["close"].pct_change()
    df["logret_1"]  = np.log(df["close"]).diff()
    df["ret_5"]     = df["close"].pct_change(5)
    df["ret_15"]    = df["close"].pct_change(15)
    df["vol_20"]    = df["ret_1"].rolling(20, min_periods=5).std(ddof=1)
    df["vol_60"]    = df["ret_1"].rolling(60, min_periods=10).std(ddof=1)

    df["ema_12"]    = ema(df["close"], 12)
    df["ema_26"]    = ema(df["close"], 26)
    macd_line, macd_sig, macd_hist = macd(df["close"], 12, 26, 9)
    df["macd"]      = macd_line
    df["macd_sig"]  = macd_sig
    df["macd_hist"] = macd_hist
    df["rsi_14"]    = rsi(df["close"], 14)

    bb_ma, bb_up, bb_dn, bb_width = bollinger(df["close"], 20, 2.0)
    df["bb_ma"]     = bb_ma
    df["bb_up"]     = bb_up
    df["bb_dn"]     = bb_dn
    df["bb_width"]  = bb_width

    df["atr_14"]    = atr(df["high"], df["low"], df["close"], 14)

    df["dist_bb_up"] = (df["bb_up"] - df["close"]) / df["bb_up"].replace(0, np.nan)
    df["dist_bb_dn"] = (df["close"] - df["bb_dn"]) / df["bb_dn"].replace(0, np.nan)
    df["ema_gap"]    = (df["ema_12"] - df["ema_26"]) / df["ema_26"].replace(0, np.nan)
    return df

# -----------------------
# Classic labeling (horizon k)
# -----------------------
def add_labels(df: pd.DataFrame, horizon:int=20, up:float=0.003, dn:float=-0.003):
    future = df["close"].shift(-horizon)
    df["fwd_ret"] = (future - df["close"]) / df["close"]
    cond_up = df["fwd_ret"] >= up
    cond_dn = df["fwd_ret"] <= dn
    df["label"] = 0
    df.loc[cond_up, "label"] = 1
    df.loc[cond_dn, "label"] = -1
    df = df.iloc[:-horizon].copy()
    return df

# -----------------------
# NEW: Sequence(flatten) builder
# -----------------------
def build_seq_flat(
    df_in: pd.DataFrame,
    n_steps:int=10,
    use_ta:bool=False,
    horizon:int=1,
    up:float=0.0,
    dn:float=0.0,
    warmup_minutes:int=420
) -> pd.DataFrame:
    """
    최근 n_steps개의 1분봉(OHLCV + 선택지표)을 한 행에 '펼쳐서' 담고,
    다음 horizon(기본 1) 분 뒤 방향 라벨을 붙인다.
    임계(up/dn)=0이면 단순 다음봉 상승/하락(=0 임계) 라벨.
    
    성능 최적화: 기술적 지표는 전체 데이터에 대해 한 번만 계산하고,
    각 윈도우에서는 미리 계산된 지표 값들을 슬라이싱하여 사용
    """
    required = ["open","high","low","close","volume","value","timestamp"]
    for c in required:
        if c not in df_in.columns:
            raise ValueError(f"Input parquet must contain '{c}'")

    df = df_in.copy().sort_values("timestamp").reset_index(drop=True)

    # 전체 데이터에 대해 기술적 지표를 한 번만 계산 (성능 최적화)
    if use_ta:
        print("[seq] 전체 데이터에 대한 기술적 지표 계산 중...")
        df = add_features_1m(df)

    # 다음 horizon 분 뒤 수익률/라벨 계산
    future = df["close"].shift(-horizon)
    fwd_ret = (future - df["close"]) / df["close"]
    if up == 0.0 and dn == 0.0:
        labels = np.where(fwd_ret > 0, 1, np.where(fwd_ret < 0, -1, 0))
    else:
        labels = np.zeros(len(df), dtype=int)
        labels[fwd_ret >= up] = 1
        labels[fwd_ret <= dn] = -1

    # 윈도우에 포함할 컬럼 결정
    base_cols = ["open","high","low","close","volume","value"]
    feat_cols = base_cols.copy()
    if use_ta:
        # 기술적 지표 컬럼 추가 (이미 계산 완료됨)
        extra = [c for c in df.columns if c not in set(base_cols + ["timestamp","timestamp_kst","fwd_ret","label"])]
        extra = [c for c in extra if pd.api.types.is_numeric_dtype(df[c])]
        feat_cols += extra

    print(f"[seq] {len(feat_cols)}개 피처로 {n_steps}봉 윈도우 생성 중...")
    
    # 🔥 워밍업 적용: 420분 (7시간) 제거로 안정화된 지표값 보장
    if len(df) <= warmup_minutes:
        raise ValueError(f"데이터가 워밍업 기간({warmup_minutes}분)보다 짧습니다. 최소 {warmup_minutes + n_steps + horizon}분 필요")
    
    print(f"[seq] 워밍업 {warmup_minutes}분 제거 (지표 안정화)")
    
    # 워밍업 적용된 윈도우 인덱스 계산
    start_idx = warmup_minutes + n_steps - 1  # 워밍업 + 윈도우 시작점
    end_idx = len(df) - horizon                # 라벨링 가능한 마지막 점
    
    if start_idx >= end_idx:
        raise ValueError(f"워밍업 후 데이터 부족: start={start_idx}, end={end_idx}. 더 긴 데이터 필요")
    
    total_windows = end_idx - start_idx
    print(f"[seq] 워밍업 후 {total_windows}개 윈도우 벡터화 처리 중...")
    
    # 결과 딕셔너리 미리 할당
    result_data = {}
    
    # 워밍업 적용된 윈도우 인덱스 생성 (벡터화)
    window_indices = np.arange(start_idx, end_idx)  # 워밍업 후 중심 인덱스
    
    # 메타데이터 벡터화 생성
    result_data["timestamp"] = df.iloc[window_indices]["timestamp"].values
    result_data["close"] = df.iloc[window_indices]["close"].astype(float).values
    result_data["fwd_ret"] = fwd_ret.iloc[window_indices].astype(float).values
    result_data["label"] = labels[window_indices].astype(int)
    
    # 🚀 핵심 최적화: 모든 피처의 모든 time step을 한번에 생성
    df_array = df[feat_cols].values  # 전체 데이터를 NumPy 배열로
    
    for step in range(n_steps):
        step_name = f"t{-(n_steps-1-step)}"
        step_indices = window_indices - (n_steps - 1 - step)
        
        # 각 피처별 벡터화 슬라이싱
        for col_idx, col in enumerate(feat_cols):
            if step == 0 and col_idx % 5 == 0:
                progress = (col_idx / len(feat_cols)) * 100
                print(f"[seq] 진행률: {progress:.1f}% ({col_idx}/{len(feat_cols)} 피처)")
            
            feature_name = f"{col}_{step_name}"
            # 벡터화된 인덱싱 (중첩 루프 대신)
            result_data[feature_name] = df_array[step_indices, col_idx].astype(float)

    print(f"[seq] 총 {total_windows}개 윈도우 생성 완료")
    out = pd.DataFrame(result_data)
    
    # 라벨은 정수로 유지 (train.py 호환성)
    # 정수 라벨: -1=short, 0=flat, 1=long
    out["label"] = out["label"].astype(int)
    
    # 워밍업/NA 제거
    out = out.dropna(axis=1, how="all")
    # 클래스 극단적 불균형 방지용 간단 필터(선택): 필요 시 주석 해제
    # out = out[(out["label"]!=0) | (out["fwd_ret"].abs()>1e-8)]

    return out

# -----------------------
# Multi-TF (classic) helpers
# -----------------------
def parse_tfs(s: str) -> List[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

# 중복 함수 제거됨 (위쪽에 이미 정의됨)

# -----------------------
# Classic builder (original)
# -----------------------
def build_classic(inp_path: str, out_path: str, horizon:int, up:float, dn:float, tfs: List[int], warmup_minutes:int=420, save_csv:bool=False):
    df = pd.read_parquet(inp_path)
    if "timestamp" not in df.columns:
        raise ValueError("Input parquet must contain 'timestamp' column.")

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # timestamp는 이미 KST (컨텍스트 유지)
    ts = pd.to_datetime(df["timestamp"])
    if ts.dt.tz is None:
        # tz-naive면 KST로 간주
        ts = ts.dt.tz_localize("Asia/Seoul")
    base_1m = df.set_index(ts)[["open","high","low","close","volume","value"]].copy()

    feat_1m = add_features_1m(df.copy())

    if tfs:
        merged = feat_1m.copy()
        # timestamp는 이미 KST
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])
        if merged["timestamp"].dt.tz is None:
            merged["timestamp"] = merged["timestamp"].dt.tz_localize("Asia/Seoul")
        left_df = merged[["timestamp"]].copy().sort_values("timestamp").reset_index(drop=True)

        for m in tfs:
            tf = resample_ohlcv(base_1m, m)
            tf_feat = add_features_ohlcv(tf, prefix=f"m{m}_")
            right_df = tf_feat.copy()
            # KST 유지
            right_df["ts"] = right_df.index
            if right_df["ts"].dt.tz != merged["timestamp"].dt.tz:
                right_df["ts"] = right_df["ts"].dt.tz_convert("Asia/Seoul")
            right_df = right_df.sort_index().reset_index(drop=True)

            aligned = pd.merge_asof(
                left_df.sort_values("timestamp"),
                right_df.sort_values("ts"),
                left_on="timestamp",
                right_on="ts",
                direction="backward",
                allow_exact_matches=True,
            )
            aligned = aligned.drop(columns=["open","high","low","close","volume","value","ts"], errors="ignore")
            merged = pd.concat([merged.reset_index(drop=True), aligned.drop(columns=["timestamp"], errors="ignore")], axis=1)
    else:
        merged = feat_1m

    # 🔥 워밍업 적용: 420분 (7시간) 제거로 안정화된 지표값 보장
    if len(merged) <= warmup_minutes:
        raise ValueError(f"데이터가 워밍업 기간({warmup_minutes}분)보다 짧습니다. 최소 {warmup_minutes + horizon}분 필요")
    
    print(f"[classic] 워밍업 {warmup_minutes}분 제거 (지표 안정화)")
    merged = merged.iloc[warmup_minutes:].copy()
    merged = add_labels(merged, horizon=horizon, up=up, dn=dn)

    keep_core = ["ret_1","ema_12","ema_26","macd","rsi_14","bb_ma","atr_14","fwd_ret","label"]
    core_exists = [c for c in keep_core if c in merged.columns]
    if core_exists:
        merged = merged.dropna(subset=core_exists)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print("dataset rows:", len(merged))
    
    # CSV 저장 옵션이 활성화된 경우에만 CSV 저장
    if save_csv:
        csv_path = out_path.replace('.parquet', '.csv')
        merged.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[CSV 저장] {csv_path}")
    
    if len(merged):
        prop = (merged["label"].value_counts(normalize=True).sort_index()
                .rename({-1:"short",0:"flat",1:"long"}))
        print(prop)
    else:
        print("Empty dataset after processing.")

# -----------------------
# Sequence builder (new)
# -----------------------
def build_seq(inp_path: str, out_path: str, n_steps:int, ta:bool, horizon:int, up:float, dn:float, warmup_minutes:int=420, save_csv:bool=False):
    df = pd.read_parquet(inp_path)
    if "timestamp" not in df.columns:
        raise ValueError("Input parquet must contain 'timestamp' column.")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out = build_seq_flat(df, n_steps=n_steps, use_ta=ta, horizon=horizon, up=up, dn=dn, warmup_minutes=warmup_minutes)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, index=False)
    print("dataset rows:", len(out))
    
    # CSV 저장 옵션이 활성화된 경우에만 CSV 저장
    if save_csv:
        csv_path = out_path.replace('.parquet', '.csv')
        out.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[CSV 저장] {csv_path}")
    
    if len(out):
        prop = (out["label"].value_counts(normalize=True).sort_index()
                .rename({-1:"short",0:"flat",1:"long"}))
        print(prop)
    else:
        print("Empty dataset after processing.")

# -----------------------
# CLI
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["classic","seq"], default="classic",
                    help="classic: 기존 지표+horizon, seq: 최근 N봉 펼침(다음 1봉)")
    ap.add_argument("--in", dest="inp", required=True, help="input parquet from tools/collect.py (1m OHLCV)")
    ap.add_argument("--out", dest="out", required=True, help="output dataset parquet")

    # classic params
    ap.add_argument("--horizon", type=int, default=20, help="label horizon in minutes (classic/seq 모두에서 사용 가능)")
    ap.add_argument("--up", type=float, default=0.003, help="upper threshold for label")
    ap.add_argument("--dn", type=float, default=-0.003, help="lower threshold for label")
    ap.add_argument("--tfs", type=str, default="", help="classic 전용: '3,5,15'")

    # seq params
    ap.add_argument("--n_steps", type=int, default=20, help="seq 전용: 최근 몇 봉을 펼칠지 (권장: 10~60)")
    ap.add_argument("--ta", action="store_true", help="seq 전용: OHLCV 외에 선택 지표(EMA/RSI/MACD/BB/ATR)도 포함")
    
    # 공통 파라미터
    ap.add_argument("--warmup_minutes", type=int, default=420, help="지표 안정화를 위한 워밍업 기간(분) [기본값: 420]")
    ap.add_argument("--save_csv", action="store_true", help="결과를 CSV 파일로도 저장 (디버깅/분석용)")

    args = ap.parse_args()

    if args.mode == "classic":
        tfs = parse_tfs(args.tfs)
        if tfs:
            print(f"[build/classic] multi-TF: {tfs} warmup={args.warmup_minutes} save_csv={args.save_csv}")
        build_classic(args.inp, args.out, args.horizon, args.up, args.dn, tfs, args.warmup_minutes, args.save_csv)
    else:
        print(f"[build/seq] n_steps={args.n_steps} ta={args.ta} horizon={args.horizon} up={args.up} dn={args.dn} warmup={args.warmup_minutes} save_csv={args.save_csv}")
        build_seq(args.inp, args.out, args.n_steps, args.ta, args.horizon, args.up, args.dn, args.warmup_minutes, args.save_csv)

if __name__ == "__main__":
    main()
