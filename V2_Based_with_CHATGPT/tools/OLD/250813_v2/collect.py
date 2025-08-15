
"""
collect.py - 업비트 OHLCV 데이터 수집 도구 (최적화)

CLI 사용 예시:

python tools/collect.py `
  --market KRW-BTC `
  --minutes 1 `
  --days 180 `
  --out data/raw/krw_btc_1m_180d.parquet

사용법:
• 업비트 공개 API에서 암호화폐 1분봉 OHLCV 데이터를 수집합니다
• API 키 불필요: 공개 데이터 수집이므로 인증 없이 사용 가능
• 자동 재시도: 네트워크 오류 시 자동으로 재시도 (5회)
• 안정적 수집: rate limit 준수로 안정적 데이터 수집

매개변수:
• --market: 거래 마켓 (KRW-BTC, KRW-ETH, KRW-ADA, KRW-SOL 등)
• --minutes: 분봉 단위 (현재 1분봉만 지원)
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
            time.sleep(0.12)
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
    out = pd.DataFrame({
        "timestamp_kst": pd.to_datetime(df["candle_date_time_kst"]).dt.tz_localize("Asia/Seoul"),
        "open": df["opening_price"],
        "high": df["high_price"],
        "low": df["low_price"],
        "close": df["trade_price"],
        "volume": df["candle_acc_trade_volume"],
        "value": df["candle_acc_trade_price"],
    })
    out["timestamp"] = pd.to_datetime(df["candle_date_time_utc"], format="%Y-%m-%dT%H:%M:%S").dt.tz_localize("UTC")
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out

def ceil_minute(dt_utc: datetime) -> datetime:
    # Upbit candle boundaries are minute-based; align to next minute to be safe
    return dt_utc.replace(second=0, microsecond=0) + timedelta(minutes=1) if (dt_utc.second or dt_utc.microsecond) else dt_utc

def collect_days_fast(market="KRW-BTC", unit=1, days=1, verbose=True, rps_limit: Optional[float]=None) -> pd.DataFrame:
    """
    Deterministic stepping:
      to_utc starts at the *next minute* after now (exclusive).
      Each page, subtract exactly 200*unit minutes from `to_utc`.
    """
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=days)
    # Start from the next minute boundary to avoid edge duplication
    to_anchor = ceil_minute(now_utc)
    to_utc = to_anchor.strftime("%Y-%m-%dT%H:%M:%SZ")

    frames = []
    page = 0
    # holder for simple rps limiter timestamp
    last_ts_holder = [None]

    # Estimate pages: ceil( minutes_needed / (200*unit) )
    minutes_needed = int((to_anchor - start_utc).total_seconds() // 60)
    est_pages = (minutes_needed + (200*unit) - 1) // (200*unit)
    if verbose:
        print(f"[plan] need about ~{est_pages} pages (minutes={minutes_needed}, step={200*unit})")

    while True:
        page += 1
        try:
            df = fetch_upbit_minutes(market, unit, 200, to_utc, rps_limit, last_ts_holder)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if verbose:
                print(f"[warn] page {page} fetch failed: {e}. retry in 1s...")
            time.sleep(1.0)
            page -= 1
            continue

        if df.empty:
            if verbose:
                print("[info] empty page, stopping.")
            break

        frames.append(df)
        oldest_utc = df["timestamp"].min()
        newest_utc = df["timestamp"].max()
        if verbose:
            print(f"[page {page}/{est_pages}] rows={len(df)} {oldest_utc} ~ {newest_utc}")

        if oldest_utc <= start_utc:
            break

        # Deterministic step: go exactly one page span back
        span = timedelta(minutes=200*unit)
        next_to = (to_anchor - page*span)
        to_utc = next_to.strftime("%Y-%m-%dT%H:%M:%SZ")

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    data = data[data["timestamp"] >= start_utc].reset_index(drop=True)
    try:
        data["timestamp_kst"] = data["timestamp"].dt.tz_convert("Asia/Seoul")
    except Exception:
        pass
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

    print(f"[start-fast] market={args.market} unit={args.minutes}m days={args.days}")
    try:
        df = collect_days_fast(args.market, args.minutes, args.days, verbose=not args.quiet, rps_limit=args.rps)
    except KeyboardInterrupt:
        print("[abort] interrupted by user.")
        return

    print("total rows:", len(df))
    if len(df) == 0:
        print("[error] no data fetched. Check network/proxy/firewall or try smaller --days")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print("saved ->", args.out)

if __name__ == "__main__":
    main()
