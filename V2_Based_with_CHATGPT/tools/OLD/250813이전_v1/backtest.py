r"""
tools/backtest.py (final)
- Calibrated classifier + meta 임계치 사용
- 스케일러(Optional, None-safe)
- 거래 재구성(언제/왜 진입·청산했는지) + CSV/콘솔 출력
- 자동 Markdown 리포트 생성(--auto_md --md_out 옵션)

사용 예(PowerShell):
  python -m tools.backtest `
    --data data/dataset_mtf.parquet `
    --model artifacts/model_mtf.joblib `
    --meta artifacts/meta_mtf.json `
    --fee 0.0005 `
    --report artifacts/backtest_report_mtf.json `
    --log_csv artifacts/trades_mtf.csv `
    --print_trades 20 `
    --auto_md `
    --md_out artifacts/trades_mtf.md

    python tools/backtest.py --data data/dataset_mtf.parquet --model artifacts/model_mtf.joblib --meta artifacts/meta_mtf.json --scaler artifacts/scaler_mtf.joblib --fee 0.0005 --report artifacts/backtest_report_mtf.json --log_csv artifacts/trades_mtf.csv --print_trades 20 --auto_md --md_out artifacts/trades_mtf.md

"""

import argparse
import os
import json
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import joblib


# -----------------------
# Helpers
# -----------------------
def load_for_infer(dataset_path: str, features: list) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, np.ndarray]:
    df = pd.read_parquet(dataset_path).sort_values("timestamp").reset_index(drop=True)
    X = df[features].values
    ts = pd.to_datetime(df["timestamp"])
    close = df["close"].values
    return df, X, ts, close


def maybe_transform(X: np.ndarray, scaler_path: Optional[str]) -> np.ndarray:
    if not scaler_path or not os.path.exists(scaler_path):
        return X
    scaler = joblib.load(scaler_path)
    if scaler is None:
        return X
    return scaler.transform(X)


def class_indices(clf) -> Dict[str, Optional[int]]:
    classes = list(getattr(clf, "classes_", [-1, 0, 1]))
    return {
        "short": classes.index(-1) if -1 in classes else None,
        "flat": classes.index(0) if 0 in classes else None,
        "long": classes.index(1) if 1 in classes else None,
    }


def positions_from_proba(proba: np.ndarray, meta: dict, idx: dict):
    allow_short = bool(meta.get("allow_short", False))
    long_p = float(meta.get("optimized_long_p", 0.6))
    short_p = float(meta.get("optimized_short_p", 0.6))
    p_long = proba[:, idx["long"]] if idx["long"] is not None else np.zeros(len(proba))
    p_short = proba[:, idx["short"]] if (allow_short and idx["short"] is not None) else None

    if allow_short and p_short is not None:
        pos = np.where(p_long >= long_p, 1, np.where(p_short >= short_p, -1, 0))
    else:
        pos = (p_long >= long_p).astype(int)
    return pos.astype(int), p_long, (p_short if p_short is not None else None), {
        "long_p": long_p,
        "short_p": short_p,
        "allow_short": allow_short,
    }


def equity_curve(close: np.ndarray, pos: np.ndarray, fee_per_side: float):
    # next-bar execution
    ret = pd.Series(close).pct_change().fillna(0.0).values
    pos_shift = np.roll(pos, 1)
    pos_shift[0] = 0
    gross = (1 + ret * pos_shift).cumprod()
    trades = np.abs(np.diff(pos_shift, prepend=0))
    fee_curve = np.cumprod(1 - fee_per_side * trades)
    equity = gross * fee_curve
    return equity, trades, ret, pos_shift


def reconstruct_trades(
    ts: pd.Series,
    close: np.ndarray,
    pos_shift: np.ndarray,
    p_long: np.ndarray,
    p_short: Optional[np.ndarray],
    th: dict,
    fee: float,
) -> List[dict]:
    allow_short = th["allow_short"]
    long_p = th["long_p"]
    short_p = th["short_p"]
    trades = []
    cur_dir = 0  # -1/0/1

    def side_from(v): return "LONG" if v == 1 else ("SHORT" if v == -1 else "FLAT")

    for i in range(len(pos_shift)):
        new_dir = int(pos_shift[i])

        # open
        if cur_dir == 0 and new_dir != 0:
            reason = f"enter {side_from(new_dir)}: " + (
                f"p_long={p_long[i]:.2f}>= {long_p:.2f}" if new_dir == 1
                else f"p_short={(p_short[i] if p_short is not None else 0):.2f}>= {short_p:.2f}"
            )
            trades.append({
                "side": side_from(new_dir),
                "entry_idx": i,
                "entry_time": ts[i],
                "entry_price": float(close[i]),
                "entry_reason": reason,
                "exit_idx": None,
                "exit_time": None,
                "exit_price": None,
                "exit_reason": None,
                "hold_minutes": None,
                "gross_ret": None,
                "net_ret": None,
                "cum_equity_after": None,
            })
            cur_dir = new_dir
            continue

        # exit/flip
        if cur_dir != 0 and new_dir != cur_dir:
            last = trades[-1] if trades else None
            if last and last["exit_idx"] is None:
                last["exit_idx"] = i
                last["exit_time"] = ts[i]
                last["exit_price"] = float(close[i])
                if cur_dir == 1:
                    reason = f"exit LONG: p_long={p_long[i]:.2f} < {long_p:.2f}"
                    gross = (close[i] / last["entry_price"]) - 1.0
                else:
                    sprob = (p_short[i] if p_short is not None else 0.0)
                    reason = f"exit SHORT: p_short={sprob:.2f} < {short_p:.2f}"
                    gross = (last["entry_price"] / close[i]) - 1.0
                if new_dir != 0:
                    reason += f" & flip to {side_from(new_dir)}"
                net = (1 + gross) * (1 - fee) * (1 - fee) - 1
                last["exit_reason"] = reason
                last["gross_ret"] = float(gross)
                last["net_ret"] = float(net)
                last["hold_minutes"] = int((ts[i] - last["entry_time"]).total_seconds() // 60)

            # flip open
            if new_dir != 0:
                reason = f"enter {side_from(new_dir)}: " + (
                    f"p_long={p_long[i]:.2f}>= {long_p:.2f}" if new_dir == 1
                    else f"p_short={(p_short[i] if p_short is not None else 0):.2f}>= {short_p:.2f}"
                )
                trades.append({
                    "side": side_from(new_dir),
                    "entry_idx": i,
                    "entry_time": ts[i],
                    "entry_price": float(close[i]),
                    "entry_reason": reason,
                    "exit_idx": None,
                    "exit_time": None,
                    "exit_price": None,
                    "exit_reason": None,
                    "hold_minutes": None,
                    "gross_ret": None,
                    "net_ret": None,
                    "cum_equity_after": None,
                })
            cur_dir = new_dir

    # last open -> force close at end
    if cur_dir != 0 and trades:
        last = trades[-1]
        if last["exit_idx"] is None:
            i = len(pos_shift) - 1
            last["exit_idx"] = i
            last["exit_time"] = ts[i]
            last["exit_price"] = float(close[i])
            if cur_dir == 1:
                gross = (close[i] / last["entry_price"]) - 1.0
                last["exit_reason"] = "final close LONG (end of test)"
            else:
                gross = (last["entry_price"] / close[i]) - 1.0
                last["exit_reason"] = "final close SHORT (end of test)"
            net = (1 + gross) * (1 - fee) * (1 - fee) - 1
            last["gross_ret"] = float(gross)
            last["net_ret"] = float(net)
            last["hold_minutes"] = int((ts[i] - last["entry_time"]).total_seconds() // 60)

    return trades


def attach_cum_equity(trades: List[dict], close: np.ndarray, pos_shift: np.ndarray, fee: float):
    equity, trades_arr, ret, pos = equity_curve(close, pos_shift, fee)
    for t in trades:
        if t["exit_idx"] is not None:
            t["cum_equity_after"] = float(equity[t["exit_idx"]])
    return trades, equity


# -----------------------
# Markdown pretty printer (내장)
# -----------------------
def _pct(x):
    return f"{x*100:+.4f}%" if pd.notna(x) else "N/A"


def _comma(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)


def _format_trade_block(idx, row: dict) -> str:
    side = row.get("side", "NA")
    entry_time = row.get("entry_time", "NA")
    exit_time = row.get("exit_time", "NA")
    hold_minutes = int(row.get("hold_minutes", 0)) if pd.notna(row.get("hold_minutes")) else 0
    entry_price = _comma(row.get("entry_price", "NA"))
    exit_price = _comma(row.get("exit_price", "NA"))
    gross = _pct(row.get("gross_ret"))
    net = _pct(row.get("net_ret"))
    entry_reason = row.get("entry_reason", "")
    exit_reason = row.get("exit_reason", "")
    cum_eq = row.get("cum_equity_after", None)
    cum_eq_txt = f"{cum_eq:.4f}배" if pd.notna(cum_eq) else "N/A"

    title = f"[{idx}] {side} | {entry_time} → {exit_time} (보유 {hold_minutes}분)"
    lines = [
        f"**{title}**",
        f"- 진입가: {entry_price} | 청산가: {exit_price}",
        f"- 수익률: {gross} (순수익률 {net})",
        f"- 진입조건: {entry_reason}",
        f"- 청산조건: {exit_reason}",
        f"- 누적자본배수: {cum_eq_txt}",
        "",
    ]
    return "\n".join(lines)


def write_markdown(trades: List[dict], summary: dict, meta: dict, md_out: str, limit: Optional[int] = None):
    lines = []
    lines.append("# 거래 일지 (자동 생성)\n")
    if summary:
        lines.append("## 백테스트 요약")
        lines.append(f"- 관측수: {summary.get('n_obs', 'N/A')}")
        lines.append(f"- 거래수: {summary.get('n_trades', 'N/A')}")
        fe = summary.get("final_equity", "N/A")
        hf = summary.get("hold_final", "N/A")
        lines.append(f"- 최종 자본배수: {fe:.4f}" if isinstance(fe, (int, float)) else f"- 최종 자본배수: {fe}")
        lines.append(f"- 홀딩(무전략) 배수: {hf:.4f}" if isinstance(hf, (int, float)) else f"- 홀딩(무전략) 배수: {hf}")
        lines.append(f"- 기간: {summary.get('start', 'N/A')} ~ {summary.get('end', 'N/A')}")
        if "thresholds" in summary:
            thr = summary["thresholds"]
            lines.append(f"- 임계치: long_p={thr.get('long_p','N/A')}, short_p={thr.get('short_p','N/A')}, allow_short={thr.get('allow_short','N/A')}")
        lines.append("")

    if meta:
        lines.append("## 모델/임계치 정보")
        if "optimized_long_p" in meta:
            lines.append(f"- 최적 롱 확률 임계치: {meta['optimized_long_p']:.2f}")
        if meta.get("allow_short", False) and "optimized_short_p" in meta:
            lines.append(f"- 최적 숏 확률 임계치: {meta['optimized_short_p']:.2f}")
        if "fee_used" in meta:
            lines.append(f"- 수수료(한쪽): {meta['fee_used']}")
        if "algo" in meta:
            lines.append(f"- 알고리즘: {meta['algo']}")
        lines.append("")

    lines.append("## 거래 상세")
    rows = trades[: limit] if limit else trades
    for i, row in enumerate(rows, start=1):
        lines.append(_format_trade_block(i, row))

    os.makedirs(os.path.dirname(md_out), exist_ok=True)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved markdown -> {md_out}")


# -----------------------
# Main backtest
# -----------------------
def backtest(
    dataset_path: str,
    model_path: str,
    meta_path: str,
    scaler_path: Optional[str],
    fee: float,
    report_path: str,
    log_csv: Optional[str],
    print_trades: int,
    auto_md: bool = False,
    md_out: Optional[str] = None,
    md_limit: Optional[int] = None,
):
    # load
    clf = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta["features"]

    df, X, ts, close = load_for_infer(dataset_path, features)
    X = maybe_transform(X, scaler_path)

    # proba -> position
    idx = class_indices(clf)
    proba = clf.predict_proba(X)
    pos, p_long, p_short, th = positions_from_proba(proba, meta, idx)

    # equity + trades
    equity, trades_arr, ret, pos_shift = equity_curve(close, pos, fee)
    trades = reconstruct_trades(ts, close, pos_shift, p_long, p_short, th, fee)
    trades, equity = attach_cum_equity(trades, close, pos_shift, fee)

    # summary
    result = {
        "n_obs": int(len(df)),
        "n_trades": int(sum(1 for t in trades if t["exit_idx"] is not None)),
        "final_return": float(equity[-1] - 1),
        "final_equity": float(equity[-1]),
        "hold_final": float((1 + pd.Series(close).pct_change().fillna(0.0)).cumprod().iloc[-1]),
        "start": str(ts.iloc[0]) if len(ts) else None,
        "end": str(ts.iloc[-1]) if len(ts) else None,
        "fee_per_side": float(fee),
        "thresholds": th,
    }

    # save json
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # save csv
    if log_csv:
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        pd.DataFrame(trades).to_csv(log_csv, index=False, encoding="utf-8-sig")

    # print summary
    print("Backtest summary:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    # print sample trades
    if print_trades and len(trades):
        print(f"\n=== Sample trades (first {min(print_trades, len(trades))} of {len(trades)}) ===")
        for t in trades[:print_trades]:
            et = t['entry_time']
            xt = t['exit_time']
            hm = t['hold_minutes']
            ep = t['entry_price']
            xp = t['exit_price']
            gr = t['gross_ret']
            nr = t['net_ret']
            erea = t['entry_reason']
            exre = t['exit_reason']
            ceq = t['cum_equity_after']
            print(
                f"[{t['side']}] {et} -> {xt} "
                f"hold={hm}m entry={ep:.0f} exit={xp:.0f} "
                f"gross={(gr if gr is not None else 0):+0.4%} net={(nr if nr is not None else 0):+0.4%} "
                f"| {erea} | {exre} | cumEq={(ceq if ceq is not None else 1.0):.4f}"
            )

    # auto markdown
    if auto_md and md_out:
        try:
            write_markdown(trades, result, meta, md_out, limit=md_limit)
        except Exception as e:
            print(f"[warn] markdown export failed: {e}")

    return result, trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True, help="meta json saved during training")
    ap.add_argument("--scaler", default=None, help="optional scaler path (ignored if missing or None)")
    ap.add_argument("--fee", type=float, default=0.0005, help="per side fee (e.g., 0.0005 = 0.05%)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--log_csv", default=None, help="optional path to save detailed trade log CSV")
    ap.add_argument("--print_trades", type=int, default=20, help="print first N trades to console (0 to disable)")
    ap.add_argument("--auto_md", action="store_true", help="export pretty markdown without external script")
    ap.add_argument("--md_out", default=None, help="markdown output path (requires --auto_md)")
    ap.add_argument("--md_limit", type=int, default=None, help="limit number of trades in markdown")
    args = ap.parse_args()

    backtest(
        args.data,
        args.model,
        args.meta,
        args.scaler,
        args.fee,
        args.report,
        args.log_csv,
        args.print_trades,
        auto_md=args.auto_md,
        md_out=args.md_out,
        md_limit=args.md_limit,
    )


if __name__ == "__main__":
    main()
