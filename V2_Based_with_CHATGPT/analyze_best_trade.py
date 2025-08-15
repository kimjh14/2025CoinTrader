"""
analyze_best_trade.py - 백테스트 최고 수익 거래 상세 분석 도구

CLI 사용 예시:
python analyze_best_trade.py `
  --trades artifacts/backtest/trades_hgb_h20_0.001.csv `
  --data data/classic/dataset_mtf_h20_0.001.parquet `
  --model artifacts/train/hgb_model.joblib `
  --top 1

사용법:
• 백테스트 CSV에서 최고 수익 거래를 찾아 상세 분석합니다
• --trades: 백테스트 거래 로그 CSV 파일
• --data: 원본 데이터셋 (피처값 분석용)
• --model: 학습된 모델 (예측 확률 확인용)
• --top: 상위 N개 거래 분석 (기본값: 1)

분석 내용:
• 거래 정보: 진입/청산 시간, 가격, 수익률
• 신호 시점: 진입/청산 1분 전 피처값과 예측 확률
• 보유 기간: 5분 간격 가격/확률 변화 추적
• 거래 실행: 신호 발생 다음 봉의 시가에서 체결 (백테스트와 동일)
"""

import argparse
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timezone
import os
import sys
import warnings

# sklearn 경고 메시지 숨기기
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def analyze_top_trades(trades_file, data_file, model_file, top_n=1):
    """최고 수익 거래들을 분석"""
    
    # CSV 파일 읽기
    trades_df = pd.read_csv(trades_file)
    
    # net_ret 기준 상위 거래 선택
    top_trades = trades_df.nlargest(top_n, 'net_ret')
    
    # 데이터와 모델 로드
    df = pd.read_parquet(data_file)
    model = joblib.load(model_file)
    
    # 피처 컬럼 준비
    drop_cols = {"label", "fwd_ret", "timestamp", "timestamp_kst"}
    feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    for rank, (idx, trade) in enumerate(top_trades.iterrows(), 1):
        print(f"\n{'='*60}")
        print(f"TOP {rank} 수익 거래 분석")
        print(f"{'='*60}")
        
        # 거래 정보 파싱
        entry_time = pd.to_datetime(trade['entry_time'])
        exit_time = pd.to_datetime(trade['exit_time'])
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        net_ret = trade['net_ret']
        gross_ret = trade['gross_ret']
        side = trade['side']
        hold_minutes = trade['hold_minutes']
        
        print(f"\n거래 유형: {side}")
        print(f"거래 기간: {entry_time} → {exit_time}")
        print(f"진입가: {entry_price:,.0f}원")
        print(f"청산가: {exit_price:,.0f}원")
        print(f"총 수익률: {gross_ret*100:.2f}%")
        print(f"순 수익률: {net_ret*100:.2f}% (수수료 차감)")
        print(f"보유시간: {hold_minutes:.0f}분")
        
        # 진입 신호 시점 분석 (진입 1분 전 = 신호 발생 시점)
        signal_time = entry_time - pd.Timedelta(minutes=1)
        signal_idx = df[df['timestamp'] == signal_time].index
        if len(signal_idx) == 0:
            time_diff = abs(df['timestamp'] - signal_time)
            signal_idx = [time_diff.idxmin()]
        signal_idx = signal_idx[0]
        
        signal_row = df.iloc[signal_idx]
        
        # 진입 시점 봉의 시가 구하기 (다음 봉)
        entry_idx = df[df['timestamp'] == entry_time].index
        if len(entry_idx) == 0:
            time_diff = abs(df['timestamp'] - entry_time)
            entry_idx = [time_diff.idxmin()]
        entry_open = df.iloc[entry_idx[0]]['open']
        
        print(f"\n=== 진입 신호 시점 ({signal_time}) 분석 ===")
        print(f"(실제 진입은 {entry_time}에 실행)")
        print(f"신호 발생 시점 종가: {signal_row['close']:,.0f}원")
        print(f"다음 봉 시가 (실제 진입가): {entry_open:,.0f}원")
        print(f"백테스트 진입가: {entry_price:,.0f}원 (동일하게 다음 봉 시가)")
        print(f"MACD: {signal_row['macd']:.2f}")
        print(f"RSI: {signal_row['rsi_14']:.1f}")
        print(f"볼린저밴드 중심선: {signal_row['bb_ma']:,.0f}원")
        print(f"EMA12: {signal_row['ema_12']:,.0f}원")
        print(f"EMA26: {signal_row['ema_26']:,.0f}원")
        print(f"1분 수익률: {signal_row['ret_1']*100:.2f}%")
        print(f"변동성(20분): {signal_row['vol_20']:.6f}")
        print(f"ATR: {signal_row['atr_14']:,.0f}원")
        
        # 멀티 타임프레임 지표
        if 'm3_macd' in signal_row:
            print(f"\n--- 멀티 타임프레임 지표 ---")
            print(f"3분봉 MACD: {signal_row['m3_macd']:.2f}")
            print(f"3분봉 RSI: {signal_row['m3_rsi_14']:.1f}")
            if 'm5_macd' in signal_row:
                print(f"5분봉 MACD: {signal_row['m5_macd']:.2f}")
                print(f"5분봉 RSI: {signal_row['m5_rsi_14']:.1f}")
        
        # 모델 예측 (신호 시점 기준)
        X_signal = df[feature_cols].iloc[signal_idx:signal_idx+1]
        proba_signal = model.predict_proba(X_signal)
        
        print(f"\n=== 신호 시점 모델 예측 ===")
        print(f"P_SHORT: {proba_signal[0][0]:.6f} ({proba_signal[0][0]*100:.2f}%)")
        print(f"P_FLAT:  {proba_signal[0][1]:.6f} ({proba_signal[0][1]*100:.2f}%)")
        print(f"P_LONG:  {proba_signal[0][2]:.6f} ({proba_signal[0][2]*100:.2f}%)")
        
        if side == "LONG":
            print(f"\n진입 결정: P_LONG={proba_signal[0][2]:.6f} >= 임계값 → 다음 봉 시가에서 매수!")
        else:
            print(f"\n진입 결정: P_SHORT={proba_signal[0][0]:.6f} >= 임계값 → 다음 봉 시가에서 매도!")
        
        # 보유 기간 중 가격 변화 추적
        print(f"\n=== 보유 기간 중 가격 변화 ===")
        period_data = df[(df['timestamp'] >= entry_time) & (df['timestamp'] <= exit_time)].copy()
        
        # 5분 간격으로 샘플링
        sample_interval = max(1, len(period_data) // 10)
        sample_indices = range(0, len(period_data), sample_interval)
        
        for i in sample_indices:
            if i < len(period_data):
                row = period_data.iloc[i]
                timestamp = row['timestamp']
                price = row['close']
                
                if side == "LONG":
                    profit_pct = ((price / entry_price) - 1) * 100
                else:  # SHORT
                    profit_pct = ((entry_price / price) - 1) * 100
                
                # 해당 시점의 예측
                row_idx = df[df['timestamp'] == timestamp].index[0]
                X_sample = df[feature_cols].iloc[row_idx:row_idx+1]
                proba_sample = model.predict_proba(X_sample)
                
                if side == "LONG":
                    p_signal = proba_sample[0][2]
                    signal_name = "P_LONG"
                else:
                    p_signal = proba_sample[0][0]
                    signal_name = "P_SHORT"
                
                print(f"{timestamp.strftime('%H:%M')} | {price:,.0f}원 ({profit_pct:+.2f}%) | {signal_name}={p_signal:.6f}")
        
        # 청산 신호 시점 분석
        # backtest.py에서 pos_shift 때문에 실제 신호는 청산 시점에 발생
        exit_signal_time = exit_time
        exit_signal_idx = df[df['timestamp'] == exit_signal_time].index
        if len(exit_signal_idx) == 0:
            time_diff = abs(df['timestamp'] - exit_signal_time)
            exit_signal_idx = [time_diff.idxmin()]
        exit_signal_idx = exit_signal_idx[0]
        
        exit_signal_row = df.iloc[exit_signal_idx]
        
        # 청산 시점 봉의 시가 구하기 (다음 봉)
        exit_idx = df[df['timestamp'] == exit_time].index
        if len(exit_idx) == 0:
            time_diff = abs(df['timestamp'] - exit_time)
            exit_idx = [time_diff.idxmin()]
        exit_open = df.iloc[exit_idx[0]]['open']
        
        print(f"\n=== 청산 신호 시점 ({exit_signal_time}) 분석 ===")
        print(f"(청산 신호와 실행이 동시에 발생 - pos_shift 효과)")
        print(f"신호 발생 시점 종가: {exit_signal_row['close']:,.0f}원")
        print(f"다음 봉 시가 (실제 청산가): {exit_open:,.0f}원")
        print(f"백테스트 청산가: {exit_price:,.0f}원 (동일하게 다음 봉 시가)")
        print(f"MACD: {exit_signal_row['macd']:.2f}")
        print(f"RSI: {exit_signal_row['rsi_14']:.1f}")
        
        # 청산 신호 시점 예측
        X_exit_signal = df[feature_cols].iloc[exit_signal_idx:exit_signal_idx+1]
        proba_exit_signal = model.predict_proba(X_exit_signal)
        
        print(f"\n=== 청산 신호 시점 모델 예측 ===")
        print(f"P_SHORT: {proba_exit_signal[0][0]:.6f} ({proba_exit_signal[0][0]*100:.2f}%)")
        print(f"P_FLAT:  {proba_exit_signal[0][1]:.6f} ({proba_exit_signal[0][1]*100:.2f}%)")
        print(f"P_LONG:  {proba_exit_signal[0][2]:.6f} ({proba_exit_signal[0][2]*100:.2f}%)")
        
        if side == "LONG":
            print(f"\n청산 결정: P_LONG={proba_exit_signal[0][2]:.6f} < 임계값 → 다음 봉 시가에서 매도!")
        else:
            print(f"\n청산 결정: P_SHORT={proba_exit_signal[0][0]:.6f} < 임계값 → 다음 봉 시가에서 매수!")
        
        print(f"\n=== 거래 요약 ===")
        print(f"진입 이유: {trade['entry_reason']}")
        print(f"청산 이유: {trade['exit_reason']}")
        print(f"실제 결과: {hold_minutes:.0f}분 만에 {net_ret*100:+.2f}% 수익!")
        
        # CSV의 exit_reason에 있는 p_long 값과 비교
        if 'p_long=' in trade['exit_reason']:
            csv_p_long = trade['exit_reason'].split('p_long=')[1].split(' ')[0]
            print(f"\n[검증] CSV의 청산 시점 P_LONG: {csv_p_long}")
            print(f"[검증] 분석 도구의 청산 시점 P_LONG: {proba_exit_signal[0][2]:.6f}")
            print(f"[검증] 일치 여부: {'일치' if abs(float(csv_p_long) - proba_exit_signal[0][2]) < 0.001 else '불일치'}")

def main():
    parser = argparse.ArgumentParser(description="백테스트 최고 수익 거래 분석")
    parser.add_argument("--trades", required=True, help="백테스트 거래 CSV 파일")
    parser.add_argument("--data", required=True, help="데이터셋 파일 (.parquet)")
    parser.add_argument("--model", required=True, help="모델 파일 (.joblib)")
    parser.add_argument("--top", type=int, default=1, help="상위 N개 거래 분석 (기본: 1)")
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.trades):
        print(f"거래 파일을 찾을 수 없습니다: {args.trades}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"데이터 파일을 찾을 수 없습니다: {args.data}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"모델 파일을 찾을 수 없습니다: {args.model}")
        sys.exit(1)
    
    analyze_top_trades(args.trades, args.data, args.model, args.top)

if __name__ == "__main__":
    main()