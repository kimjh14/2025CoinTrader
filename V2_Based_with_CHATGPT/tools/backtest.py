"""
tools/backtest.py (final)
- Calibrated classifier + meta 임계치 사용
- 스케일러(Optional, None-safe)
- 거래 재구성(언제/왜 진입·청산했는지) + CSV/콘솔 출력
- 자동 Markdown 리포트 생성(--auto_md --md_out 옵션)
- 시가 기준 거래 실행 (신호 다음 봉의 시가에서 체결)

CLI 사용 예시:
python tools/backtest.py `
    --data data/classic/dataset_mtf_5d_h2_0.001.parquet `
    --model artifacts/train/model_classic_10d_h2_0.001.joblib `
    --meta artifacts/train/meta_classic_10d_h2_0.001.json `
    --scaler artifacts/train/scaler_classic_10d_h2_0.001.joblib `
    --fee 0.0005 `
    --report artifacts/backtest/bt_classic_5d_h2_0.001.json `
    --log_csv artifacts/backtest/trades_classic_5d_h2_0.001.csv `
    --auto_md `
    --md_out artifacts/backtest/trades_classic_5d_h2_0.001.md

사용법:
• train.py로 훈련된 모델의 백테스팅을 수행합니다
• --data: 테스트용 데이터셋 (훈련과 동일한 데이터셋 사용)
• --model: 훈련된 모델 파일 (.joblib)
• --meta: 모델 메타데이터 (.json) - 임계값과 피처명 포함
• --scaler: 스케일러 파일 (V2에서는 사용 안함, 호환용)
• --fee: 거래 수수료 (0.0005 = 편도 0.05%)
• --report: 백테스트 결과 요약 (.json)
• --log_csv: 개별 거래 상세 로그 (.csv)
• --auto_md: 자동 마크다운 리포트 생성
• --md_out: 마크다운 리포트 저장 경로
• 시가 기준 거래: 신호 발생 다음 봉의 시가에서 체결
"""

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import joblib

# 프로젝트 레이아웃 기준: tools/ 에서 ml/ 모듈 import
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.append(str(ROOT_DIR))  # V2_Based_with_CHATGPT/ 를 모듈 루트로

# 공용 PnL 및 리스크 지표 사용
from ml.pnl_utils import calculate_next_open_pnl, calculate_risk_metrics


# -----------------------
# Helpers
# -----------------------
def load_for_infer(dataset_path: str, features: list) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, np.ndarray, np.ndarray]:
    df = pd.read_parquet(dataset_path).sort_values("timestamp").reset_index(drop=True)
    X = df[features].values
    ts = pd.to_datetime(df["timestamp"])
    close = df["close"].values
    open_prices = df["open"].values  # 시가 데이터 추가
    return df, X, ts, close, open_prices


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


def equity_curve(close: np.ndarray, open_prices: np.ndarray, pos: np.ndarray, fee_per_side: float):
    """공용 PnL 함수를 사용한 에쿼티 커브 계산"""
    # 가격 데이터프레임 생성
    prices_df = pd.DataFrame({
        'open': open_prices,
        'close': close
    })
    
    # 공용 PnL 함수 사용
    equity, returns, trades_arr = calculate_next_open_pnl(
        prices=prices_df,
        positions=pos,
        fee=fee_per_side,
        initial_capital=1.0
    )
    
    # 거래 발생 시점 찾기
    trades = []
    pos_shift = np.roll(pos, 1)
    pos_shift[0] = 0
    
    for i in range(1, len(pos)):
        if pos_shift[i] != pos_shift[i-1]:
            trades.append(i)
    
    # ret 계산 (기존 호환성)
    ret = pd.Series(close).pct_change().fillna(0.0).values
    
    return equity, trades, ret, pos_shift


def reconstruct_trades(
    ts: pd.Series,
    close: np.ndarray,
    open_prices: np.ndarray,
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

        # open - 시가로 진입
        if cur_dir == 0 and new_dir != 0:
            # pos_shift 때문에 실제 조건은 이전 시점에서 결정됨
            actual_idx = i - 1 if i > 0 else i
            # 시가로 진입
            entry_price = open_prices[i] if i < len(open_prices) else close[i]
            
            reason = f"enter {side_from(new_dir)}: " + (
                f"p_long={p_long[actual_idx]:.3f} >= {long_p:.3f}" if new_dir == 1
                else f"p_short={(p_short[actual_idx] if p_short is not None else 0):.3f} >= {short_p:.3f}"
            )
            trades.append({
                "side": side_from(new_dir),
                "entry_idx": i,
                "entry_time": ts[i],
                "entry_price": float(entry_price),  # 시가로 진입
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

        # exit/flip - 시가로 청산
        if cur_dir != 0 and new_dir != cur_dir:
            last = trades[-1] if trades else None
            if last and last["exit_idx"] is None:
                last["exit_idx"] = i
                last["exit_time"] = ts[i]
                # 시가로 청산
                exit_price = open_prices[i] if i < len(open_prices) else close[i]
                last["exit_price"] = float(exit_price)
                
                # 간단한 청산 이유 - 신호 시점 다음 봉에서 무조건 실행
                actual_idx = i - 1 if i > 0 else i
                if cur_dir == 1:
                    reason = f"exit LONG: p_long={p_long[actual_idx]:.3f} < {long_p:.3f}"
                    gross = (last["exit_price"] / last["entry_price"]) - 1.0
                else:
                    sprob = (p_short[actual_idx] if p_short is not None else 0.0)
                    reason = f"exit SHORT: p_short={sprob:.3f} < {short_p:.3f}"
                    gross = (last["entry_price"] / last["exit_price"]) - 1.0
                    
                if new_dir != 0:
                    reason += f" & flip to {side_from(new_dir)}"
                net = (1 + gross) * (1 - fee) * (1 - fee) - 1
                last["exit_reason"] = reason
                last["gross_ret"] = float(gross)
                last["net_ret"] = float(net)
                last["hold_minutes"] = int((ts[i] - last["entry_time"]).total_seconds() // 60)

            # flip open - 시가로 진입
            if new_dir != 0:
                # pos_shift 때문에 실제 조건은 이전 시점에서 결정됨
                actual_idx = i - 1 if i > 0 else i
                entry_price = open_prices[i] if i < len(open_prices) else close[i]
                
                reason = f"flipped: " + (
                    f"p_long={p_long[actual_idx]:.3f} >= {long_p:.3f}" if new_dir == 1
                    else f"p_short={(p_short[actual_idx] if p_short is not None else 0):.3f} >= {short_p:.3f}"
                )
                trades.append({
                    "side": side_from(new_dir),
                    "entry_idx": i,
                    "entry_time": ts[i],
                    "entry_price": float(entry_price),  # 시가로 진입
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

    # final
    if trades and trades[-1]["exit_idx"] is None:
        i = len(pos_shift) - 1
        last = trades[-1]
        last["exit_idx"] = i
        last["exit_time"] = ts[i]
        last["exit_price"] = float(close[i])  # 마지막은 종가로
        if last["side"] == "LONG":
            gross = (close[i] / last["entry_price"]) - 1.0
            last["exit_reason"] = "final close LONG (end of test)"
        else:
            gross = (last["entry_price"] / close[i]) - 1.0
            last["exit_reason"] = "final close SHORT (end of test)"
        net = (1 + gross) * (1 - fee) * (1 - fee) - 1
        last["gross_ret"] = float(gross)
        last["net_ret"] = float(net)
        last["hold_minutes"] = int((ts[i] - last["entry_time"]).total_seconds() // 60)

    # equity after
    eq = 1.0
    for t in trades:
        nr = t.get("net_ret")
        if nr is not None:
            eq *= (1 + nr)
        t["cum_equity_after"] = eq

    return trades


def run_backtest(args):
    # paths
    required = ["data", "model", "meta"]
    for r in required:
        if not getattr(args, r) or not os.path.exists(getattr(args, r)):
            print(f"파일을 찾을 수 없습니다: {getattr(args, r)}")
            return
    
    # meta
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta.get("features", [])
    allow_short = bool(meta.get("allow_short", False))
    
    # load
    df, X, ts, close, open_prices = load_for_infer(args.data, features)
    X = maybe_transform(X, args.scaler)
    model = joblib.load(args.model)
    
    # predict
    try:
        proba = model.predict_proba(X)
    except AttributeError:
        if hasattr(model, "estimator"):
            proba = model.estimator.predict_proba(X)
        else:
            raise
    
    # positions
    idx = class_indices(model if not hasattr(model, "estimator") else model.estimator)
    pos, p_long, p_short, th = positions_from_proba(proba, meta, idx)
    
    # equity curve (시가 기준)
    equity, _, ret, pos_shift = equity_curve(close, open_prices, pos, args.fee)
    
    # trades (시가 기준)
    trades = reconstruct_trades(ts, close, open_prices, pos_shift, p_long, p_short, th, args.fee)
    
    # metrics - 올바른 누적 수익률 사용 (거래 기준)
    # 마지막 거래의 누적자본배수를 사용
    final_equity = trades[-1].get('cum_equity_after', 1.0) if trades else 1.0
    cum_ret = final_equity - 1  # 올바른 수익률
    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t.get("net_ret", 0) > 0)
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    
    # 홀딩(무전략) 수익률 계산
    hold_return = (close[-1] / close[0] - 1) if len(close) > 0 else 0
    
    # 리스크 지표 계산
    risk_metrics = calculate_risk_metrics(equity, periods_per_year=365*24*60)
    
    long_trades = [t for t in trades if t["side"] == "LONG"]
    short_trades = [t for t in trades if t["side"] == "SHORT"]
    
    print(f"\n=== 백테스트 결과 (시가 기준 거래) ===")
    print(f"수익률: {cum_ret*100:.2f}%")
    print(f"거래 수: {n_trades}")
    print(f"승률: {win_rate*100:.1f}%")
    print(f"LONG 거래: {len(long_trades)}")
    print(f"SHORT 거래: {len(short_trades)}")
    print(f"\n=== 리스크 지표 ===")
    print(f"MDD: {risk_metrics['mdd']*100:.2f}%")
    print(f"Sharpe Ratio: {risk_metrics['sharpe']:.2f}")
    print(f"Calmar Ratio: {risk_metrics['calmar']:.2f}")
    print(f"최대 연속 손실: {risk_metrics['max_consecutive_losses']}건")
    print(f"최대 연속 이익: {risk_metrics['max_consecutive_wins']}건")
    
    # CSV
    if args.log_csv and trades:
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
        pd.DataFrame(trades).to_csv(args.log_csv, index=False, encoding="utf-8-sig")
        print(f"거래 로그 저장: {args.log_csv}")
    
    # Report
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        report_data = {
            "cumulative_return": float(cum_ret),
            "num_trades": n_trades,
            "win_rate": float(win_rate),
            "num_long": len(long_trades),
            "num_short": len(short_trades),
            "fee": args.fee,
            "allow_short": allow_short,
            "thresholds": th,
            "mdd": risk_metrics['mdd'],
            "sharpe": risk_metrics['sharpe'],
            "calmar": risk_metrics['calmar'],
            "max_consecutive_losses": risk_metrics['max_consecutive_losses'],
            "max_consecutive_wins": risk_metrics['max_consecutive_wins'],
            "hold_return": float(hold_return),
            "execution_rule": "next_open"
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"리포트 저장: {args.report}")
    
    # Markdown
    if args.auto_md and args.md_out and trades:
        os.makedirs(os.path.dirname(args.md_out), exist_ok=True)
        # cum_ret은 이미 올바르게 계산됨 (마지막 거래의 누적자본 사용)
        summary_info = {
            'n_obs': len(df),
            'n_trades': n_trades,
            'cum_ret': cum_ret,  # 이미 올바른 값
            'hold_return': hold_return,
            'win_rate': win_rate,
            'start_time': ts.iloc[0] if len(ts) > 0 else None,
            'end_time': ts.iloc[-1] if len(ts) > 0 else None,
            'thresholds': th,
            'fee': args.fee
        }
        generate_markdown_report(trades, summary_info, args.md_out)
        print(f"마크다운 리포트 저장: {args.md_out}")


def generate_markdown_report(trades, summary_info, output_path):
    """마크다운 형식의 거래 리포트 생성 (상세 형식)"""
    
    def format_trade_block(i, trade):
        """거래 하나를 포맷팅"""
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time']) if trade['exit_time'] else None
        
        # KST 시간으로 표시
        entry_str = entry_time.strftime('%Y-%m-%d %H:%M KST') if entry_time else 'N/A'
        exit_str = exit_time.strftime('%Y-%m-%d %H:%M KST') if exit_time else '보유중'
        
        lines = []
        lines.append(f"**[{i}] {trade['side']} | {entry_str} → {exit_str} (보유 {trade.get('hold_minutes', 0)}분)**")
        lines.append(f"- 진입가: {trade['entry_price']:,.0f} | 청산가: {trade.get('exit_price', 0):,.0f}")
        lines.append(f"- 수익률: {trade.get('gross_ret', 0)*100:+.4f}% (순수익률 {trade.get('net_ret', 0)*100:+.4f}%)")
        lines.append(f"- 진입조건: {trade['entry_reason']}")
        lines.append(f"- 청산조건: {trade.get('exit_reason', '보유중')}")
        lines.append(f"- 누적자본배수: {trade.get('cum_equity_after', 1):.4f}배")
        return "\n".join(lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 거래 일지 (자동 생성)\n\n")
        
        # 백테스트 요약
        f.write("## 백테스트 요약\n")
        f.write(f"- 관측수: {summary_info['n_obs']}\n")
        f.write(f"- 거래수: {summary_info['n_trades']}\n")
        f.write(f"- 최종 자본배수: {1 + summary_info['cum_ret']:.4f}\n")
        f.write(f"- 홀딩(무전략) 배수: {1 + summary_info['hold_return']:.4f}\n")
        if summary_info['start_time'] and summary_info['end_time']:
            start_str = pd.to_datetime(summary_info['start_time']).strftime('%Y-%m-%d %H:%M KST')
            end_str = pd.to_datetime(summary_info['end_time']).strftime('%Y-%m-%d %H:%M KST')
            f.write(f"- 기간: {start_str} ~ {end_str}\n")
        th = summary_info['thresholds']
        f.write(f"- 임계치: long_p={th['long_p']:.3f}, short_p={th['short_p']:.3f}, allow_short={th['allow_short']}\n\n")
        
        # 모델/임계치 정보
        f.write("## 모델/임계치 정보\n")
        f.write(f"- 최적 롱 확률 임계치: {th['long_p']:.3f}\n")
        if th['allow_short']:
            f.write(f"- 최적 숏 확률 임계치: {th['short_p']:.3f}\n")
        f.write(f"- 수수료(한쪽): {summary_info['fee']}\n")
        f.write("- 알고리즘: HistGradientBoostingClassifier + IsotonicCalibration (FrozenEstimator)\n\n")
        
        # 거래 상세
        f.write("## 거래 상세\n")
        for i, trade in enumerate(trades, 1):
            f.write(format_trade_block(i, trade))
            f.write("\n\n")  # 각 거래 사이에 빈 줄 추가


def main():
    parser = argparse.ArgumentParser(description="백테스트 (시가 기준 거래)")
    parser.add_argument("--data", required=True, help="dataset path")
    parser.add_argument("--model", required=True, help="model path (.joblib)")
    parser.add_argument("--meta", required=True, help="meta path (.json)")
    parser.add_argument("--scaler", help="scaler path (optional)")
    parser.add_argument("--fee", type=float, default=0.0005, help="fee per side")
    parser.add_argument("--report", help="save report to this json")
    parser.add_argument("--log_csv", help="save trades to csv")
    parser.add_argument("--auto_md", action="store_true", help="auto generate markdown")
    parser.add_argument("--md_out", help="markdown output path")
    
    args = parser.parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()