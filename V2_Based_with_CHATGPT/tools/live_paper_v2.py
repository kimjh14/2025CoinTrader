"""
live_paper_v2.py - 개선된 실시간 페이퍼 트레이딩 시스템

주요 기능:
1. 실행 시점 기준 최신 완성된 봉부터 450개 수집 (420 워밍업 + 30 여유분)
2. 420분 워밍업으로 지표 안정화 (build_dataset.py 방식)
3. 최신 봉의 p_long, p_short, p_flat 계산
4. 실시간 가상 거래 시뮬레이션 (포지션 관리, 수익률 계산)
5. 거래 로그 CSV 저장 및 상태 관리

CLI 사용 예시:
python tools/live_paper_v2.py `
  --market KRW-BTC `
  --model artifacts/train/model_classic_200d_h3_0.003.joblib `
  --meta artifacts/train/meta_classic_200d_h3_0.003.json `
  --scaler artifacts/train/scaler_classic_200d_h3_0.003.joblib `
  --fee 0.0005 `
  --state artifacts/live/state_v2.json `
  --log_csv artifacts/live/trades_v2.csv `
  --once
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import joblib

# 프로젝트 루트 설정
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# 기존 모듈 임포트
from tools.collect import fetch_upbit_minutes, make_session, floor_minute
from tools.build_dataset import (
    add_features_1m, add_features_ohlcv, resample_ohlcv,
    ema, rsi, macd, bollinger, atr
)

# 상수 정의
WARMUP_MINUTES = 420  # 7시간 워밍업 (지표 안정화)
COLLECT_MINUTES = 450  # 총 수집할 분봉 수 (420 워밍업 + 30 여유분)
MIN_REQUIRED = WARMUP_MINUTES + 1  # 최소 필요 데이터 (워밍업 + 예측용 1개)

# 세션 초기화
SESSION = make_session()

def now_kst() -> datetime:
    """현재 KST 시간 반환"""
    return datetime.now(timezone(timedelta(hours=9)))

def collect_latest_data(market: str = "KRW-BTC", total_minutes: int = COLLECT_MINUTES) -> pd.DataFrame:
    """
    최신 완성된 봉부터 역순으로 데이터 수집
    - floor_minute으로 미완성 봉 제외
    - 충분한 데이터 확보 (워밍업 + 분석)
    """
    print(f"[수집] {market} 최신 {total_minutes}개 분봉 수집 시작...")
    
    # 현재 시각을 분 단위로 내림 (완성된 봉만)
    now_utc = datetime.now(timezone.utc)
    to_anchor = floor_minute(now_utc)
    to_utc = to_anchor.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    frames = []
    collected = 0
    page = 0
    last_ts_holder = [None]
    
    while collected < total_minutes:
        page += 1
        need = min(200, total_minutes - collected)
        
        try:
            # API 호출 (SESSION 사용)
            resp = SESSION.get(
                f"https://api.upbit.com/v1/candles/minutes/1",
                params={"market": market, "count": need, "to": to_utc},
                timeout=5
            )
            resp.raise_for_status()
            
            # Rate limit 처리
            from tools.collect import throttle_from_header
            throttle_from_header(resp, rps_limit=8.0, last_ts_holder=last_ts_holder)
            
            js = resp.json()
            if not js:
                break
                
            # DataFrame 변환
            df = pd.DataFrame(js)
            out = pd.DataFrame({
                "timestamp": pd.to_datetime(df["candle_date_time_kst"]).dt.tz_localize("Asia/Seoul"),
                "open": df["opening_price"],
                "high": df["high_price"],
                "low": df["low_price"],
                "close": df["trade_price"],
                "volume": df["candle_acc_trade_volume"],
                "value": df["candle_acc_trade_price"],
            })
            out["timestamp_utc"] = pd.to_datetime(df["candle_date_time_utc"]).dt.tz_localize("UTC")
            
            frames.append(out)
            collected += len(out)
            
            # 다음 페이지 준비
            oldest = pd.to_datetime(df["candle_date_time_utc"].iloc[-1])
            to_anchor = oldest - timedelta(minutes=1)
            to_utc = to_anchor.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            print(f"[수집] 페이지 {page}: {len(out)}개 수집 (누적 {collected}/{total_minutes})")
            
        except Exception as e:
            print(f"[수집 오류] {e}")
            time.sleep(1)
            continue
    
    if not frames:
        return pd.DataFrame()
    
    # 데이터 정리
    data = pd.concat(frames).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    
    # 미완성 봉 제외
    current_floor = floor_minute(now_utc)
    data = data[data["timestamp_utc"] < current_floor].reset_index(drop=True)
    
    print(f"[수집 완료] 총 {len(data)}개 분봉 수집")
    if len(data) > 0:
        oldest_kst = data["timestamp"].iloc[0]
        newest_kst = data["timestamp"].iloc[-1]
        print(f"[데이터 범위] {oldest_kst.strftime('%Y-%m-%d %H:%M KST')} ~ {newest_kst.strftime('%Y-%m-%d %H:%M KST')}")
    
    return data

def build_features_with_warmup(df: pd.DataFrame, tfs: List[int] = None) -> Tuple[pd.DataFrame, int]:
    """
    워밍업을 적용한 피처 생성
    - 420분 워밍업으로 지표 안정화
    - Classic 모드 피처 생성
    
    Returns:
    --------
    features_df : 워밍업 이후 피처 데이터
    warmup_used : 실제 사용된 워밍업 분 수
    """
    if len(df) < MIN_REQUIRED:
        raise ValueError(f"데이터 부족: {len(df)}개 (최소 {MIN_REQUIRED}개 필요)")
    
    print(f"[피처] 지표 계산 시작 (전체 {len(df)}개 행)")
    
    # 1분봉 피처 생성
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_features_1m(df)
    
    # Multi-timeframe 피처 추가 (옵션)
    if tfs:
        print(f"[피처] Multi-timeframe 추가: {tfs}")
        
        # timestamp가 KST이므로 인덱스 설정
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize("Asia/Seoul")
        
        base_1m = df.set_index(ts)[["open", "high", "low", "close", "volume", "value"]].copy()
        merged = df.copy()
        
        for m in tfs:
            # 리샘플링
            tf = resample_ohlcv(base_1m, m)
            tf_feat = add_features_ohlcv(tf, prefix=f"m{m}_")
            
            # merge_asof로 정합
            right_df = tf_feat.copy()
            right_df["ts"] = right_df.index
            if right_df["ts"].dt.tz != ts.dt.tz:
                right_df["ts"] = right_df["ts"].dt.tz_convert("Asia/Seoul")
            
            aligned = pd.merge_asof(
                df[["timestamp"]].sort_values("timestamp"),
                right_df.sort_values("ts"),
                left_on="timestamp",
                right_on="ts",
                direction="backward",
                allow_exact_matches=True
            )
            
            # 불필요한 컬럼 제거
            aligned = aligned.drop(columns=["open", "high", "low", "close", "volume", "value", "ts"], errors="ignore")
            merged = pd.concat([merged.reset_index(drop=True), aligned.drop(columns=["timestamp"], errors="ignore")], axis=1)
        
        df = merged
    
    # 워밍업 적용
    warmup_used = WARMUP_MINUTES  # 항상 420분 워밍업 사용
    if len(df) <= warmup_used:
        raise ValueError(f"데이터 부족: {len(df)}개 (워밍업 {warmup_used}개 + 최소 1개 필요)")
    print(f"[워밍업] {warmup_used}분 제거 (지표 안정화)")
    
    df_after_warmup = df.iloc[warmup_used:].copy()
    df_after_warmup = df_after_warmup.reset_index(drop=True)
    
    print(f"[피처 완료] 워밍업 후 {len(df_after_warmup)}개 행")
    
    return df_after_warmup, warmup_used

def predict_latest(df_features: pd.DataFrame, model_path: str, meta_path: str, 
                  scaler_path: Optional[str] = None) -> Dict:
    """
    최신 봉에 대한 예측 수행
    
    Returns:
    --------
    dict: p_long, p_short, p_flat 및 메타 정보
    """
    # 모델 로드
    model = joblib.load(model_path)
    
    # 메타데이터 로드
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    features = meta.get("features", [])
    allow_short = meta.get("allow_short", False)
    long_p = meta.get("optimized_long_p", 0.6)
    short_p = meta.get("optimized_short_p", 0.6)
    
    # 최신 행 추출
    if df_features.empty:
        raise ValueError("피처 데이터가 비어있습니다")
    
    latest_row = df_features.iloc[-1].copy()
    
    # 피처 확인 및 추출
    missing_features = set(features) - set(df_features.columns)
    if missing_features:
        print(f"[경고] 누락된 피처: {missing_features}")
        # 누락된 피처를 0으로 채움
        for feat in missing_features:
            latest_row[feat] = 0
    
    # 피처 벡터 생성
    X = latest_row[features].values.reshape(1, -1)
    
    # 스케일러 적용 (있는 경우)
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        if scaler is not None:
            X = scaler.transform(X)
    
    # 예측
    try:
        proba = model.predict_proba(X)[0]
    except AttributeError:
        # CalibratedClassifierCV의 경우
        if hasattr(model, "estimator"):
            proba = model.estimator.predict_proba(X)[0]
        else:
            raise
    
    # 클래스 매핑
    classes = list(getattr(model, "classes_", [-1, 0, 1]))
    if hasattr(model, "estimator"):
        classes = list(getattr(model.estimator, "classes_", [-1, 0, 1]))
    
    idx_short = classes.index(-1) if -1 in classes else None
    idx_flat = classes.index(0) if 0 in classes else None
    idx_long = classes.index(1) if 1 in classes else None
    
    # 확률 추출
    p_short = proba[idx_short] if idx_short is not None else 0.0
    p_flat = proba[idx_flat] if idx_flat is not None else 0.0
    p_long = proba[idx_long] if idx_long is not None else 0.0
    
    # 예측 결정
    if allow_short:
        if p_long >= long_p:
            decision = "LONG"
        elif p_short >= short_p:
            decision = "SHORT"
        else:
            decision = "FLAT"
    else:
        if p_long >= long_p:
            decision = "LONG"
        else:
            decision = "FLAT"
    
    # 주요 지표 값 추출 - 모든 숫자형 컬럼 중 지표만 추출
    indicators = {}
    skip_cols = ["timestamp", "open", "high", "low", "close", "volume", "label", "timestamp_utc"]
    
    for col in latest_row.index:
        if col not in skip_cols and not col.startswith("lag_"):
            try:
                val = float(latest_row[col])
                if not pd.isna(val):
                    indicators[col] = val
            except:
                pass
    
    # timestamp를 문자열로 변환
    timestamp_str = latest_row["timestamp"].strftime("%Y-%m-%d %H:%M:%S KST") if hasattr(latest_row["timestamp"], "strftime") else str(latest_row["timestamp"])
    
    result = {
        "timestamp": timestamp_str,  # JSON 직렬화를 위해 문자열로 저장
        "timestamp_kst": timestamp_str,
        "open": float(latest_row["open"]),
        "high": float(latest_row["high"]),
        "low": float(latest_row["low"]),
        "close": float(latest_row["close"]),
        "volume": float(latest_row["volume"]),
        "p_long": float(p_long),
        "p_short": float(p_short),
        "p_flat": float(p_flat),
        "threshold_long": float(long_p),
        "threshold_short": float(short_p),
        "allow_short": allow_short,
        "decision": decision,
        "features_used": len(features),
        "warmup_applied": True,
        "indicators": indicators
    }
    
    return result

def save_state(state_path: str, state: dict):
    """상태 저장"""
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def save_trades_csv(csv_path: str, trades: list):
    """거래 기록을 CSV로 저장"""
    if not trades:
        return
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # DataFrame 생성
    df = pd.DataFrame(trades)
    
    # CSV 저장
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[CSV 저장] {csv_path} ({len(trades)}개 거래)")

def load_state(state_path: str) -> dict:
    """상태 로드"""
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "position": 0,  # 0: 없음, 1: 롱, -1: 숏
        "entry_time": None,
        "entry_price": None,
        "cum_equity": 1.0,
        "trades": [],
        "total_trades": 0,
        "win_trades": 0,
        "lose_trades": 0
    }

def main():
    parser = argparse.ArgumentParser(description="개선된 실시간 페이퍼 트레이딩 (1단계)")
    parser.add_argument("--market", default="KRW-BTC", help="거래 마켓")
    parser.add_argument("--model", required=True, help="모델 파일 경로")
    parser.add_argument("--meta", required=True, help="메타데이터 파일 경로")
    parser.add_argument("--scaler", help="스케일러 파일 경로 (옵션)")
    parser.add_argument("--fee", type=float, default=0.0005, help="거래 수수료")
    parser.add_argument("--state", default="artifacts/live/state_v2.json", help="상태 저장 파일")
    parser.add_argument("--log_csv", default="artifacts/live/trades_v2.csv", help="거래 로그 CSV")
    parser.add_argument("--tfs", type=str, default="3,5", help="Multi-timeframe (예: '3,5')")
    parser.add_argument("--once", action="store_true", help="한 번만 실행")
    
    args = parser.parse_args()
    
    # Multi-timeframe 파싱
    tfs = []
    if args.tfs:
        tfs = [int(x.strip()) for x in args.tfs.split(",") if x.strip()]
    
    print(f"\n{'='*60}")
    print(f"실시간 페이퍼 트레이딩 V2 (1단계)")
    print(f"{'='*60}")
    print(f"마켓: {args.market}")
    print(f"모델: {args.model}")
    print(f"메타: {args.meta}")
    print(f"Multi-TF: {tfs if tfs else '없음'}")
    print(f"{'='*60}\n")
    
    while True:
        try:
            # 1. 데이터 수집
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 1단계 시작")
            df_raw = collect_latest_data(market=args.market, total_minutes=COLLECT_MINUTES)
            
            if df_raw.empty:
                print("[오류] 데이터 수집 실패")
                if args.once:
                    break
                time.sleep(60)
                continue
            
            # 2. 피처 생성 (워밍업 적용)
            try:
                df_features, warmup_used = build_features_with_warmup(df_raw, tfs=tfs)
            except ValueError as e:
                print(f"[오류] 피처 생성 실패: {e}")
                if args.once:
                    break
                time.sleep(60)
                continue
            
            # 3. 최신 봉 예측
            try:
                prediction = predict_latest(df_features, args.model, args.meta, args.scaler)
            except Exception as e:
                print(f"[오류] 예측 실패: {e}")
                if args.once:
                    break
                time.sleep(60)
                continue
            
            # 4. 결과 출력
            print("\n" + "="*60)
            print("1단계 완료: 최신 봉 예측 결과")
            print("="*60)
            print(f"시간: {prediction['timestamp_kst']}")
            print(f"OHLC: {prediction['open']:,.0f} / {prediction['high']:,.0f} / {prediction['low']:,.0f} / {prediction['close']:,.0f}")
            print(f"거래량: {prediction['volume']:.4f}")
            print("-"*60)
            
            # 기술적 지표 출력
            if "indicators" in prediction and prediction["indicators"]:
                print("기술적 지표:")
                ind = prediction["indicators"]
                
                # 1분봉 기본 지표
                print("  [1분봉 지표]")
                if "rsi_14" in ind:
                    print(f"    RSI(14):        {ind['rsi_14']:.2f}")
                if "macd" in ind:
                    print(f"    MACD:           {ind['macd']:.2f}")
                if "macd_sig" in ind:
                    print(f"    MACD Signal:    {ind['macd_sig']:.2f}")
                if "macd_hist" in ind:
                    print(f"    MACD Hist:      {ind['macd_hist']:.2f}")
                if "atr_14" in ind:
                    print(f"    ATR(14):        {ind['atr_14']:.2f}")
                
                # 볼린저 밴드
                if "bb_width" in ind:
                    print(f"    BB Width:       {ind['bb_width']:.6f}")
                if "dist_bb_up" in ind:
                    print(f"    거리(상단):      {ind['dist_bb_up']:.6f}")
                if "dist_bb_dn" in ind:
                    print(f"    거리(하단):      {ind['dist_bb_dn']:.6f}")
                
                # 수익률 및 변동성
                if "ret_1" in ind:
                    print(f"    수익률(1분):     {ind['ret_1']:.6f}")
                if "ret_5" in ind:
                    print(f"    수익률(5분):     {ind['ret_5']:.6f}")
                if "ret_15" in ind:
                    print(f"    수익률(15분):    {ind['ret_15']:.6f}")
                if "vol_20" in ind:
                    print(f"    변동성(20):      {ind['vol_20']:.6f}")
                if "vol_60" in ind:
                    print(f"    변동성(60):      {ind['vol_60']:.6f}")
                
                # 3분봉 지표
                print("\n  [3분봉 지표]")
                if "m3_rsi_14" in ind:
                    print(f"    RSI(14):        {ind['m3_rsi_14']:.2f}")
                if "m3_macd" in ind:
                    print(f"    MACD:           {ind['m3_macd']:.2f}")
                if "m3_macd_hist" in ind:
                    print(f"    MACD Hist:      {ind['m3_macd_hist']:.2f}")
                if "m3_atr_14" in ind:
                    print(f"    ATR(14):        {ind['m3_atr_14']:.2f}")
                if "m3_bb_width" in ind:
                    print(f"    BB Width:       {ind['m3_bb_width']:.6f}")
                
                # 5분봉 지표
                print("\n  [5분봉 지표]")
                if "m5_rsi_14" in ind:
                    print(f"    RSI(14):        {ind['m5_rsi_14']:.2f}")
                if "m5_macd" in ind:
                    print(f"    MACD:           {ind['m5_macd']:.2f}")
                if "m5_macd_hist" in ind:
                    print(f"    MACD Hist:      {ind['m5_macd_hist']:.2f}")
                if "m5_atr_14" in ind:
                    print(f"    ATR(14):        {ind['m5_atr_14']:.2f}")
                if "m5_bb_width" in ind:
                    print(f"    BB Width:       {ind['m5_bb_width']:.6f}")
                
                print("-"*60)
            
            print(f"예측 확률:")
            print(f"  - P(LONG):  {prediction['p_long']:.6f} (임계치: {prediction['threshold_long']:.3f})")
            print(f"  - P(FLAT):  {prediction['p_flat']:.6f}")
            print(f"  - P(SHORT): {prediction['p_short']:.6f} (임계치: {prediction['threshold_short']:.3f})")
            print("-"*60)
            print(f"결정: {prediction['decision']}")
            print(f"숏 허용: {prediction['allow_short']}")
            print(f"사용 피처: {prediction['features_used']}개")
            print(f"워밍업: {warmup_used}분 적용")
            print("="*60)
            
            # 5. 거래 시뮬레이션 및 상태 관리
            state = load_state(args.state)
            state["last_prediction"] = prediction
            state["last_update"] = datetime.now().isoformat()
            
            # 가상 거래 실행
            current_price = prediction['close']
            decision = prediction['decision']
            timestamp = prediction['timestamp']
            
            # 포지션 관리
            if state["position"] == 0:  # 포지션 없음
                if decision == "LONG":
                    # 롱 진입
                    state["position"] = 1
                    state["entry_price"] = current_price
                    state["entry_time"] = timestamp
                    print(f"\n[거래] 롱 진입 - 가격: {current_price:,.0f} KRW")
                    
                elif decision == "SHORT" and prediction.get('allow_short', False):
                    # 숏 진입
                    state["position"] = -1
                    state["entry_price"] = current_price
                    state["entry_time"] = timestamp
                    print(f"\n[거래] 숏 진입 - 가격: {current_price:,.0f} KRW")
                    
            elif state["position"] == 1:  # 롱 포지션 보유 중
                if decision == "SHORT" or decision == "FLAT":
                    # 롱 청산
                    entry_price = state["entry_price"]
                    ret = (current_price - entry_price) / entry_price - args.fee
                    state["cum_equity"] *= (1 + ret)
                    
                    # 거래 기록
                    trade = {
                        "entry_time": state["entry_time"],
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position": "LONG",
                        "return": ret,
                        "cum_equity": state["cum_equity"]
                    }
                    state["trades"].append(trade)
                    state["total_trades"] = state.get("total_trades", 0) + 1
                    if ret > 0:
                        state["win_trades"] = state.get("win_trades", 0) + 1
                    else:
                        state["lose_trades"] = state.get("lose_trades", 0) + 1
                    
                    print(f"\n[거래] 롱 청산 - 가격: {current_price:,.0f} KRW")
                    print(f"  수익률: {ret*100:.2f}%")
                    print(f"  누적 수익률: {(state['cum_equity']-1)*100:.2f}%")
                    
                    # 즉시 숏 진입 체크
                    if decision == "SHORT" and prediction.get('allow_short', False):
                        state["position"] = -1
                        state["entry_price"] = current_price
                        state["entry_time"] = timestamp
                        print(f"[거래] 숏 진입 - 가격: {current_price:,.0f} KRW")
                    else:
                        state["position"] = 0
                        state["entry_price"] = None
                        state["entry_time"] = None
                        
            elif state["position"] == -1:  # 숏 포지션 보유 중
                if decision == "LONG" or decision == "FLAT":
                    # 숏 청산
                    entry_price = state["entry_price"]
                    ret = (entry_price - current_price) / entry_price - args.fee
                    state["cum_equity"] *= (1 + ret)
                    
                    # 거래 기록
                    trade = {
                        "entry_time": state["entry_time"],
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "position": "SHORT",
                        "return": ret,
                        "cum_equity": state["cum_equity"]
                    }
                    state["trades"].append(trade)
                    state["total_trades"] = state.get("total_trades", 0) + 1
                    if ret > 0:
                        state["win_trades"] = state.get("win_trades", 0) + 1
                    else:
                        state["lose_trades"] = state.get("lose_trades", 0) + 1
                    
                    print(f"\n[거래] 숏 청산 - 가격: {current_price:,.0f} KRW")
                    print(f"  수익률: {ret*100:.2f}%")
                    print(f"  누적 수익률: {(state['cum_equity']-1)*100:.2f}%")
                    
                    # 즉시 롱 진입 체크
                    if decision == "LONG":
                        state["position"] = 1
                        state["entry_price"] = current_price
                        state["entry_time"] = timestamp
                        print(f"[거래] 롱 진입 - 가격: {current_price:,.0f} KRW")
                    else:
                        state["position"] = 0
                        state["entry_price"] = None
                        state["entry_time"] = None
            
            # 상태 저장
            save_state(args.state, state)
            
            # CSV 로그 저장
            if args.log_csv and state["trades"]:
                save_trades_csv(args.log_csv, state["trades"])
            
            # 현재 상태 출력
            print(f"\n[상태] 포지션: {'롱' if state['position']==1 else '숏' if state['position']==-1 else '없음'}")
            print(f"[상태] 누적 수익률: {(state.get('cum_equity', 1.0)-1)*100:.2f}%")
            total_trades = state.get('total_trades', 0)
            win_trades = state.get('win_trades', 0)
            lose_trades = state.get('lose_trades', 0)
            print(f"[상태] 거래 횟수: {total_trades}회 (승: {win_trades}, 패: {lose_trades})")
            print(f"[상태 저장] {args.state}")
            
            if args.once:
                print("\n[완료] --once 옵션으로 종료")
                break
            
            # 다음 분봉까지 대기
            now = datetime.now(timezone.utc)
            next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
            wait_seconds = (next_minute - now).total_seconds() + 1.5  # 1.5초 버퍼
            
            print(f"\n[대기] 다음 분봉까지 {wait_seconds:.1f}초...")
            time.sleep(wait_seconds)
            
        except KeyboardInterrupt:
            print("\n[중단] 사용자 인터럽트")
            break
        except Exception as e:
            print(f"\n[오류] 예외 발생: {e}")
            if args.once:
                break
            time.sleep(60)

if __name__ == "__main__":
    main()