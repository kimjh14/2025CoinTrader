# tools/pretty_trades_md.py
import argparse
import os
import json
import pandas as pd

def pct(x):
    return f"{x*100:+.4f}%" if pd.notna(x) else "N/A"

def comma(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)

def load_backtest_summary(report_path: str):
    if report_path and os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

def load_meta(meta_path: str):
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def format_trade_block(idx, row):
    side = row.get("side", "NA")
    entry_time = row.get("entry_time", "NA")
    exit_time = row.get("exit_time", "NA")
    hold_minutes = int(row.get("hold_minutes", 0)) if pd.notna(row.get("hold_minutes")) else 0
    entry_price = comma(row.get("entry_price", "NA"))
    exit_price = comma(row.get("exit_price", "NA"))
    gross = pct(row.get("gross_ret"))
    net = pct(row.get("net_ret"))
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
        ""
    ]
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="trades CSV (e.g., artifacts/trades_mtf.csv)")
    ap.add_argument("--out", required=True, help="output markdown path (e.g., artifacts/trades_mtf.md)")
    ap.add_argument("--report", default=None, help="optional backtest report json (e.g., artifacts/backtest_report_mtf.json)")
    ap.add_argument("--meta", default=None, help="optional meta json with thresholds")
    ap.add_argument("--limit", type=int, default=None, help="limit number of trades to export")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # 정렬 보장 (entry_idx가 있으면 그걸 우선)
    if "entry_idx" in df.columns:
        df = df.sort_values("entry_idx").reset_index(drop=True)

    if args.limit:
        df = df.head(args.limit)

    # 요약정보
    summary = load_backtest_summary(args.report) or {}
    meta = load_meta(args.meta)
    n_trades = len(df)
    final_equity = df["cum_equity_after"].iloc[-1] if n_trades > 0 and "cum_equity_after" in df.columns else None
    gross_sum = df["gross_ret"].sum() if "gross_ret" in df.columns else None
    net_sum = df["net_ret"].sum() if "net_ret" in df.columns else None

    # 헤더 만들기
    lines = []
    lines.append(f"# 거래 일지 (자동 생성)\n")
    if summary:
        lines.append("## 백테스트 요약")
        lines.append(f"- 관측수: {summary.get('n_obs', 'N/A')}")
        lines.append(f"- 거래수: {summary.get('n_trades', 'N/A')}")
        lines.append(f"- 최종 자본배수: {summary.get('final_equity', 'N/A'):.4f}" if isinstance(summary.get('final_equity'), (int,float)) else f"- 최종 자본배수: {summary.get('final_equity', 'N/A')}")
        lines.append(f"- 홀딩(무전략) 배수: {summary.get('hold_final', 'N/A'):.4f}" if isinstance(summary.get('hold_final'), (int,float)) else f"- 홀딩(무전략) 배수: {summary.get('hold_final', 'N/A')}")
        lines.append(f"- 기간: {summary.get('start', 'N/A')} ~ {summary.get('end', 'N/A')}")
        if "thresholds" in summary:
            thr = summary["thresholds"]
            lines.append(f"- 임계치: long_p={thr.get('long_p','N/A')}, short_p={thr.get('short_p','N/A')}, allow_short={thr.get('allow_short','N/A')}")
        lines.append("")
    else:
        lines.append("## 간단 요약")
        if final_equity is not None:
            lines.append(f"- 마지막 누적자본배수(거래 로그 기준): {final_equity:.4f}")
        if gross_sum is not None and net_sum is not None:
            lines.append(f"- 총 합산 수익률(로그 합계): 총 {gross_sum*100:+.2f}% / 순 {net_sum*100:+.2f}%")
        lines.append(f"- 거래수: {n_trades}\n")

    if meta:
        lines.append("## 모델/임계치 정보")
        if "optimized_long_p" in meta:
            lines.append(f"- 최적 롱 확률 임계치: {meta['optimized_long_p']:.2f}")
        if "optimized_short_p" in meta and meta.get("allow_short", False):
            lines.append(f"- 최적 숏 확률 임계치: {meta['optimized_short_p']:.2f}")
        if "fee_used" in meta:
            lines.append(f"- 수수료(한쪽): {meta['fee_used']}")
        if "algo" in meta:
            lines.append(f"- 알고리즘: {meta['algo']}")
        lines.append("")

    lines.append("## 거래 상세")
    for i, row in df.iterrows():
        lines.append(format_trade_block(i+1, row))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved markdown -> {args.out}")

if __name__ == "__main__":
    main()
