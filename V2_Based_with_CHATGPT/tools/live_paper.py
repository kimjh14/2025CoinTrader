# tools/live_paper.py
import os, sys, time, json
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import numpy as np
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from tools.collect import throttle_from_header, make_session, ceil_minute
from tools.build_dataset import (
    ema, rsi, macd, bollinger, atr,
    resample_ohlcv, add_features_ohlcv
)

# =========================
# Indicator / MTF params
# =========================
IND = {
    "EMA_FAST": 12,
    "EMA_SLOW": 26,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIG": 9,
    "RSI": 14,
    "BB_P": 20,
    "BB_STD": 2.0,
    "ATR": 14,
    "SMA50": 50,  # 메타가 요구할 때만 사용
}
MTFS = [3, 5]

# =========================
# Utils
# =========================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)

def fmt_both(ts_utc: datetime) -> str:
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    kst = ts.tz_convert("Asia/Seoul")
    return f"{ts.strftime('%Y-%m-%d %H:%M:%S %Z')}  |  KST {kst.strftime('%Y-%m-%d %H:%M:%S %Z')}"

def minutes_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return int(abs((b - a).total_seconds()) // 60)

# =========================
# Persistence (paper state)
# =========================
def load_state(path:str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"position": 0, "entry_time_utc": None, "entry_price": None, "cum_equity": 1.0}

def save_state(path:str, state:dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# =========================
# Upbit fetch (completed candles only)
# =========================
SESSION = make_session()

def fetch_last_n_minutes_completed(market: str="KRW-BTC", unit:int=1, minutes:int=240,
                                   rps_limit: Optional[float]=8.0) -> pd.DataFrame:
    """
    현재 진행 중인 분을 제외하고 마지막 완성 분봉까지 수신.
    to=floor(now) 로 요청 → 마지막 레코드는 방금-1분 봉.
    """
    to_anchor = floor_minute(now_utc())
    to_utc = to_anchor.strftime("%Y-%m-%dT%H:%M:%SZ")

    frames = []
    need = minutes
    page_span = 200 * unit
    last_ts_holder = [None]
    while need > 0:
        count = min(200, (need + unit - 1) // unit * unit)
        resp = SESSION.get(
            f"https://api.upbit.com/v1/candles/minutes/{unit}",
            params={"market": market, "count": count, "to": to_utc},
            timeout=SESSION.request_timeout,
        )
        resp.raise_for_status()
        throttle_from_header(resp, rps_limit, last_ts_holder)
        js = resp.json()
        if not js:
            break
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
        out["timestamp"] = pd.to_datetime(df["candle_date_time_utc"]).dt.tz_localize("UTC")
        frames.append(out.sort_values("timestamp"))
        to_anchor = to_anchor - timedelta(minutes=page_span)
        to_utc = to_anchor.strftime("%Y-%m-%dT%H:%M:%SZ")
        need -= page_span

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    current_floor = floor_minute(now_utc())
    data = data[data["timestamp"] < current_floor].reset_index(drop=True)
    return data

# =========================
# Feature build (1m + MTF)
# =========================
def build_features_classic(df_1m: pd.DataFrame, mtfs: List[int]=MTFS) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m

    d = df_1m.copy().sort_values("timestamp").reset_index(drop=True)

    # Returns/Vols
    d["ret_1"]    = d["close"].pct_change()
    d["logret_1"] = np.log(d["close"]).diff()
    d["ret_5"]    = d["close"].pct_change(5)
    d["ret_15"]   = d["close"].pct_change(15)
    d["vol_20"]   = d["ret_1"].rolling(20, min_periods=5).std(ddof=0)
    d["vol_60"]   = d["ret_1"].rolling(60, min_periods=10).std(ddof=0)

    # EMA / MACD / RSI
    d["ema_12"]   = ema(d["close"], IND["EMA_FAST"])
    d["ema_26"]   = ema(d["close"], IND["EMA_SLOW"])
    macd_line, macd_sig, macd_hist = macd(d["close"], IND["MACD_FAST"], IND["MACD_SLOW"], IND["MACD_SIG"])
    d["macd"]     = macd_line
    d["macd_sig"] = macd_sig
    d["macd_hist"]= macd_hist
    d["rsi_14"]   = rsi(d["close"], IND["RSI"])

    # Bollinger
    bb_ma, bb_up, bb_dn, bb_width = bollinger(d["close"], IND["BB_P"], IND["BB_STD"])
    d["bb_ma"]     = bb_ma
    d["bb_up"]     = bb_up
    d["bb_dn"]     = bb_dn
    d["bb_width"]  = bb_width

    # ATR
    d["atr_14"]    = atr(d["high"], d["low"], d["close"], IND["ATR"])

    # SMA50 (조건부 사용)
    d["sma_50"]    = d["close"].rolling(IND["SMA50"], min_periods=1).mean()

    # Position-like
    d["dist_bb_up"] = (d["bb_up"] - d["close"]) / d["bb_up"].replace(0, np.nan)
    d["dist_bb_dn"] = (d["close"] - d["bb_dn"]) / d["bb_dn"].replace(0, np.nan)
    d["ema_gap"]    = (d["ema_12"] - d["ema_26"]) / d["ema_26"].replace(0, np.nan)

    # MTF 3/5
    if mtfs:
        base_1m = d.set_index(pd.to_datetime(d["timestamp"], utc=True))[["open","high","low","close","volume","value"]]
        left_df = d[["timestamp"]].copy()
        merged = d.copy()
        for mmin in mtfs:
            tf = resample_ohlcv(base_1m, mmin)
            tf_feat = add_features_ohlcv(tf, prefix=f"m{mmin}_").copy()
            right_df = tf_feat.copy()
            right_df["ts"] = right_df.index.tz_convert("UTC")
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
        d = merged

    return d

# =========================
# Trade log (Markdown)
# =========================
def ensure_md_header(md_path: str):
    if not os.path.exists(md_path):
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Live Paper Trades (Classic)\n\n")
            f.write("| Time (UTC) | Time (KST) | Action | Price | p_long | Thr | Hold (m) | Gross | Net |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")

def append_trade(md_path: str, ts_utc: pd.Timestamp, action:str, price: float,
                 p_long: float, thr: float, hold_minutes: Optional[int],
                 gross: Optional[float], net: Optional[float]):
    ensure_md_header(md_path)
    utc_str = ts_utc.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
    kst_str = ts_utc.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")
    def pct(x):
        return f"{x*100:+.4f}%" if x is not None else ""
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"| {utc_str} | {kst_str} | {action} | {price:,.0f} | {p_long:.6f} | {thr:.2f} | "
                f"{(hold_minutes if hold_minutes is not None else '')} | {pct(gross)} | {pct(net)} |\n")

# =========================
# Inference once (+ paper trade)
# =========================
def load_meta(meta_path: str):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_once(model_path: str, meta_path: str, scaler_path: Optional[str],
               state_path: str, log_md: str,
               market:str="KRW-BTC", lookback:int=240, fee:float=0.0005):
    # data
    df = fetch_last_n_minutes_completed(market=market, unit=1, minutes=lookback, rps_limit=8.0)
    if df.empty:
        print("[live] no data fetched.")
        return None

    # 마지막 완성봉 (원시 데이터)
    last_row = df.iloc[-1]
    ts_last = pd.Timestamp(last_row["timestamp"]).tz_convert("UTC")
    try:
        o = int(last_row["open"]); h = int(last_row["high"]); l = int(last_row["low"]); c = int(last_row["close"])
    except Exception:
        o = float(last_row["open"]); h = float(last_row["high"]); l = float(last_row["low"]); c = float(last_row["close"])
    print(f"[live] last completed candle: {fmt_both(ts_last)}")
    print(f"[live] O/H/L/C: {o} / {h} / {l} / {c}")

    # features
    feat = build_features_classic(df, mtfs=MTFS)
    current_floor = floor_minute(now_utc())
    feat = feat[feat["timestamp"] < current_floor].reset_index(drop=True)
    if len(feat) == 0:
        print("[live] no completed rows after feature filter.")
        return None

    # model/meta
    clf = joblib.load(model_path)
    meta = load_meta(meta_path)
    feats = meta["features"]
    th_long = float(meta.get("optimized_long_p", 0.6))

    row = feat.iloc[-1].copy()
    X = row[feats].values.reshape(1, -1)
    proba = clf.predict_proba(X)[0]
    classes = list(getattr(clf, "classes_", [-1, 0, 1]))
    idx_short = classes.index(-1) if -1 in classes else None
    idx_flat  = classes.index(0)  if  0 in classes else None
    idx_long  = classes.index(1)  if  1 in classes else None

    p_short = float(proba[idx_short]) if idx_short is not None else 0.0
    p_flat  = float(proba[idx_flat])  if idx_flat  is not None else 0.0
    p_long  = float(proba[idx_long])  if idx_long  is not None else 0.0

    ts_feat = pd.to_datetime(row["timestamp"]).tz_convert("UTC")
    print(f"[live] feature row used:     {fmt_both(ts_feat)}")
    print(f"[live] probs  short={p_short:.6f}  flat={p_flat:.6f}  long={p_long:.6f}  (thr_long={th_long:.2f})")


    # ---- Paper trading logic (long-only) ----
    state = load_state(state_path)
    action = None
    price = float(c)  # 체결가: 마지막 완성 봉 종가

    if state["position"] == 0:
        if p_long >= th_long:
            # BUY
            state["position"] = 1
            state["entry_time_utc"] = str(ts_feat)
            state["entry_price"] = price
            action = "BUY"
            append_trade(log_md, ts_feat, "BUY", price, p_long, th_long, None, None, None)
            print(f"[live] action => BUY @ {price:,.0f}")
        else:
            action = "FLAT"
            print("[live] action => FLAT")
    else:
        # currently long
        if p_long < th_long:
            # EXIT
            entry_ts = pd.Timestamp(state["entry_time_utc"]).tz_convert("UTC")
            entry_px = float(state["entry_price"])
            gross = (price / entry_px) - 1.0
            net = (1 + gross) * (1 - fee) * (1 - fee) - 1.0
            hold_m = minutes_between(entry_ts, ts_feat)
            state["position"] = 0
            state["entry_time_utc"] = None
            state["entry_price"] = None
            action = "SELL"
            append_trade(log_md, ts_feat, "SELL", price, p_long, th_long, hold_m, gross, net)
            print(f"[live] action => SELL @ {price:,.0f} | gross={gross:+.4%} net={net:+.4%} hold={hold_m}m")
        else:
            action = "HOLD"
            print("[live] action => HOLD")

    save_state(state_path, state)

    return {
        "timestamp_utc": str(ts_feat),
        "timestamp_kst": str(ts_feat.tz_convert("Asia/Seoul")),
        "p_long": p_long,
        "threshold_long": th_long,
        "last_close": price,
        "action": action,
        "position": state["position"],
    }

# =========================
# Main loop
# =========================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="KRW-BTC")
    ap.add_argument("--lookback", type=int, default=240)
    ap.add_argument("--model", default="artifacts/model_mtf_h20_p03.joblib")
    ap.add_argument("--meta",  default="artifacts/meta_mtf_h20_p03.json")
    ap.add_argument("--scaler", default="artifacts/scaler_mtf_h20_p03.joblib")
    ap.add_argument("--fee", type=float, default=0.0005)
    ap.add_argument("--poll_buffer_sec", type=float, default=1.2)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--state", default="artifacts/live_state.json")
    ap.add_argument("--log_md", default="tools/live_trades.md")
    args = ap.parse_args()

    print(f"[live] market={args.market} lookback={args.lookback} model={args.model}")
    print("[live] behavior: waits for the CURRENT minute to finish, then infers on the last completed candle (KST/UTC safe).")

    while True:
        # wait for current minute to finish
        now = now_utc()
        next_minute = ceil_minute(now)
        wait_s = (next_minute - now).total_seconds() + args.poll_buffer_sec
        print(f"\n[live] tick at: {fmt_both(now)}  |  waiting {wait_s:.2f}s for minute close ...")
        if wait_s > 0:
            time.sleep(wait_s)

        try:
            infer_once(args.model, args.meta, args.scaler,
                       state_path=args.state, log_md=args.log_md,
                       market=args.market, lookback=args.lookback, fee=args.fee)
        except KeyboardInterrupt:
            print("[live] interrupted by user.")
            break
        except Exception as e:
            print(f"[live][warn] inference failed: {e}")

        if args.once:
            break

if __name__ == "__main__":
    main()
