
"""
collect.py - 업비트 OHLCV 데이터 수집 도구 (최적화)

CLI 사용 예시:

python tools/collect.py `
  --market KRW-BTC `
  --minutes 1 `
  --days 30 `
  --out data/raw/krw_btc_1m_30d.parquet  

사용법:
• 업비트 공개 API에서 암호화폐 OHLCV 데이터를 수집합니다
• API 키 불필요: 공개 데이터 수집이므로 인증 없이 사용 가능
• 자동 재시도: 네트워크 오류 시 자동으로 재시도 (5회)
• 안정적 수집: rate limit 준수로 안정적 데이터 수집

매개변수:
• --market: 거래 마켓 (KRW-BTC, KRW-ETH, KRW-ADA, KRW-SOL 등)
• --minutes: 분봉 단위 (1, 3, 5, 10, 15, 30, 60, 240분 지원)
• --days: 수집 기간 (일 단위, 권장: 180-270일)
• --out: 저장 경로 (.parquet 형식 권장)

출력 데이터:
• timestamp: UTC 시간 (timezone aware)
• open, high, low, close: OHLC 가격
• volume: 거래량, value: 거래금액
• build_dataset.py의 직접 입력으로 사용 가능
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
import random

try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config import Config  # noqa: F401
except Exception:
    class Config:
        UPBIT_ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY", "")
        UPBIT_SECRET_KEY = os.getenv("UPBIT_SECRET_KEY", "")

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

def make_session(retries:int=5, backoff:float=0.2, timeout:int=5) -> requests.Session:
    s = requests.Session()
    r = Retry(
        total=retries, connect=retries, read=retries,
        status=retries, backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update({
        "Accept": "application/json",
        "User-Agent": "upbit-btc-pipeline/0.5-fast (+https://local)"
    })
    s.request_timeout = timeout
    return s

SESSION = make_session()

def throttle_from_header(resp: requests.Response, rps_limit: Optional[float]=None, last_ts_holder: list=None) -> None:
    h = resp.headers.get("Remaining-Req", "")
    sec_allow = None
    try:
        parts = dict(x.strip().split("=", 1) for x in h.split(";"))
        sec_allow = int(parts.get("sec", "").strip())
    except Exception:
        pass
    if sec_allow is not None:
        if sec_allow <= 0:
            time.sleep(1.02)
        elif sec_allow <= 2:
            # Jitter 추가로 동시성 스파이크 방지
            base_sleep = 0.12
            jitter = random.uniform(-0.03, 0.08)  # ±30-80ms jitter
            time.sleep(base_sleep + jitter)
    # Optional manual cap (best-effort)
    if rps_limit and rps_limit > 0 and last_ts_holder is not None:
        now = time.time()
        if last_ts_holder[0] is not None:
            elapsed = now - last_ts_holder[0]
            min_gap = 1.0 / rps_limit
            if elapsed < min_gap:
                time.sleep(min_gap - elapsed)
        last_ts_holder[0] = time.time()

def upbit_get(url: str, params: dict, rps_limit: Optional[float], last_ts_holder: list):
    resp = SESSION.get(url, params=params, timeout=SESSION.request_timeout)
    resp.raise_for_status()
    throttle_from_header(resp, rps_limit, last_ts_holder)
    return resp

def fetch_upbit_minutes(market="KRW-BTC", unit=1, count=200, to_utc_isoz: Optional[str]=None,
                        rps_limit: Optional[float]=None, last_ts_holder: list=None) -> pd.DataFrame:
    url = f"https://api.upbit.com/v1/candles/minutes/{unit}"
    params = {"market": market, "count": count}
    if to_utc_isoz:
        params["to"] = to_utc_isoz
    r = upbit_get(url, params, rps_limit, last_ts_holder)
    js = r.json()
    if not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    # KST를 기본 timestamp로 사용
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df["candle_date_time_kst"]).dt.tz_localize("Asia/Seoul"),  # KST가 기본
        "open": df["opening_price"],
        "high": df["high_price"],
        "low": df["low_price"],
        "close": df["trade_price"],
        "volume": df["candle_acc_trade_volume"],
        "value": df["candle_acc_trade_price"],
    })
    # UTC는 보조 컴럼 (필요시 사용)
    out["timestamp_utc"] = pd.to_datetime(df["candle_date_time_utc"], format="%Y-%m-%dT%H:%M:%S").dt.tz_localize("UTC")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out

def floor_minute(dt_utc: datetime) -> datetime:
    # 완성된 봉만 수집하기 위해 현재 분의 0초로 내림
    return dt_utc.replace(second=0, microsecond=0)

def collect_days_fast(market="KRW-BTC", unit=1, days=1, verbose=True, rps_limit: Optional[float]=None) -> pd.DataFrame:
    """
    Deterministic stepping:
      to_utc starts at the current minute (floor) to get only completed candles.
      Each page, subtract exactly 200*unit minutes from `to_utc`.
    """
    now_utc = datetime.now(timezone.utc)
    now_kst = now_utc.astimezone(timezone(timedelta(hours=9)))  # KST = UTC+9
    start_utc = now_utc - timedelta(days=days)
    start_kst = start_utc.astimezone(timezone(timedelta(hours=9)))
    
    # 현재 분의 시작(완성된 봉만 가져오기 위해)
    to_anchor = floor_minute(now_utc)
    to_utc = to_anchor.strftime("%Y-%m-%dT%H:%M:%SZ")

    frames = []
    page = 0
    # holder for simple rps limiter timestamp
    last_ts_holder = [None]
    
    # 메트릭 초기화
    api_calls = 0
    retry_count = 0
    start_time = time.time()

    # Estimate pages: ceil( minutes_needed / (200*unit) )
    minutes_needed = int((to_anchor - start_utc).total_seconds() // 60)
    est_pages = (minutes_needed + (200*unit) - 1) // (200*unit)
    if verbose:
        print(f"[수집 계획] {now_kst.strftime('%Y-%m-%d %H:%M:%S KST')} 기준")
        print(f"[수집 범위] {start_kst.strftime('%Y-%m-%d %H:%M:%S KST')} ~ {now_kst.strftime('%Y-%m-%d %H:%M:%S KST')}")
        print(f"[예상 페이지] 약 {est_pages}개 (총 {minutes_needed:,}분, 페이지당 {200*unit}분)")
        # 첫 번째 요청 URL 표시
        print(f"[첫 요청 URL] https://api.upbit.com/v1/candles/minutes/{unit}?market={market}&count=200&to={to_utc}")

    while True:
        page += 1
        
        # 마지막 페이지 최적화: 남은 분 수 계산
        remaining_minutes = int((to_anchor - (page-1)*timedelta(minutes=200*unit) - start_utc).total_seconds() // 60)
        optimal_count = min(200, max(1, remaining_minutes // unit))
        
        try:
            api_calls += 1
            if optimal_count < 200 and verbose:
                print(f"[opt] page {page}: requesting only {optimal_count} candles")
            df = fetch_upbit_minutes(market, unit, optimal_count, to_utc, rps_limit, last_ts_holder)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            retry_count += 1
            if verbose:
                print(f"[warn] page {page} fetch failed: {e}. retry in 1s... (retry #{retry_count})")
            time.sleep(1.0)
            page -= 1
            continue

        if df.empty:
            if verbose:
                print("[info] empty page, stopping.")
            break

        frames.append(df)
        # KST 시간으로 표시 및 비교
        oldest_kst = df["timestamp"].min()  # 이제 timestamp가 KST
        newest_kst = df["timestamp"].max()
        if verbose:
            # KST 시간으로 출력 (보기 편하게)
            oldest_str = oldest_kst.strftime("%Y-%m-%d %H:%M:%S KST")
            newest_str = newest_kst.strftime("%Y-%m-%d %H:%M:%S KST")
            print(f"[page {page}/{est_pages}] rows={len(df)} {oldest_str} ~ {newest_str}")
        
        # 비교도 KST로 (내부 로직)
        # start_utc를 KST로 변환해서 비교
        start_kst_tz = start_utc.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=9)))
        if oldest_kst.tz_localize(None) <= start_kst_tz.replace(tzinfo=None):
            break

        # Deterministic step: go exactly one page span back
        span = timedelta(minutes=200*unit)
        next_to = (to_anchor - page*span)
        to_utc = next_to.strftime("%Y-%m-%dT%H:%M:%SZ")

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    
    # KST 기준으로 필터링
    start_kst_tz = start_utc.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=9)))
    data = data[data["timestamp"].dt.tz_localize(None) >= start_kst_tz.replace(tzinfo=None)].reset_index(drop=True)
    
    # timestamp는 이미 KST (collect에서 설정됨)
    # timestamp_kst 컴럼은 호환성을 위해 추가 (동일한 값)
    data["timestamp_kst"] = data["timestamp"]
    
    # 메트릭 로깅
    if verbose:
        elapsed_time = time.time() - start_time
        avg_wait = elapsed_time / api_calls if api_calls > 0 else 0
        print(f"\n[수집 완료 메트릭]")
        print(f"- 총 API 호출: {api_calls}회")
        print(f"- 429 재시도: {retry_count}회")
        print(f"- 총 소요시간: {elapsed_time:.1f}초")
        print(f"- 평균 대기시간: {avg_wait:.3f}초/호출")
        print(f"- 수집 데이터: {len(data)}개 행")
    
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default=os.getenv("UPBIT_MARKET", "KRW-BTC"))
    ap.add_argument("--minutes", type=int, default=int(os.getenv("UPBIT_MINUTES", "1")))
    ap.add_argument("--days", type=int, default=int(os.getenv("UPBIT_DAYS", "1")))
    ap.add_argument("--out", default=os.getenv("UPBIT_OUT", "data/krw_btc_1m.parquet"))
    ap.add_argument("--quiet", action="store_true", help="suppress progress logs")
    ap.add_argument("--rps", type=float, default=None, help="cap requests per second (optional)")
    args = ap.parse_args()

    # 실행 시작 시간을 가장 먼저 표시 (초 단위까지)
    start_time_kst = datetime.now(timezone(timedelta(hours=9)))
    print(f"\n[실행 시간] {start_time_kst.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"[설정] market={args.market}, 봉={args.minutes}분, 기간={args.days}일")
    try:
        df = collect_days_fast(args.market, args.minutes, args.days, verbose=not args.quiet, rps_limit=args.rps)
    except KeyboardInterrupt:
        print("[중단] 사용자에 의해 중단됨")
        return

    print(f"[결과] 총 {len(df):,}개 행 수집")
    if len(df) == 0:
        print("[오류] 데이터가 수집되지 않음. 네트워크/프록시/방화벽 확인 또는 --days 값 줄이기")
        return
    
    # 수집된 데이터 범위 표시 (KST)
    if "timestamp" in df.columns:
        oldest_kst = df["timestamp"].min()  # timestamp가 이미 KST
        newest_kst = df["timestamp"].max()
        print(f"[데이터 범위] {oldest_kst.strftime('%Y-%m-%d %H:%M:%S KST')} ~ {newest_kst.strftime('%Y-%m-%d %H:%M:%S KST')}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Parquet 저장
    df.to_parquet(args.out, index=False)
    print(f"[저장 완료] {args.out}")
    
    # CSV 저장 (동일한 경로에 .csv 확장자로)
    csv_path = args.out.replace('.parquet', '.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[CSV 저장] {csv_path}")

if __name__ == "__main__":
    main()
