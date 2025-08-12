"""
check_exact_timing.py - 특정 시점 피처값 비교 분석 도구

CLI 사용 예시:
python check_exact_timing.py `
  --data data/classic/dataset_mtf_h20_0.001.parquet `
  --times "2025-03-02 15:46:00" "2025-03-02 15:47:00" "2025-03-02 16:40:00" "2025-03-02 16:41:00"

사용법:
• 지정된 시점들의 OHLCV와 모든 기술적 지표를 비교 분석합니다
• 매수/매도 신호와 실행 시점 간 차이를 확인할 때 유용합니다
• 특정 거래의 진입/청산 타이밍을 검증할 때 사용합니다

분석 내용:
• OHLCV: 시가, 고가, 저가, 종가, 거래량
• 주요 지표: MACD, RSI, EMA, 볼린저밴드, ATR, 변동성
• 멀티 타임프레임: 3분봉, 5분봉 MACD/RSI
• 라벨 정보: fwd_ret과 label (신호 시점만)
• 가격 변화 요약: 시점별 수익률 계산
"""

import pandas as pd
import numpy as np

df = pd.read_parquet("data/classic/dataset_mtf_h20_0.001.parquet")
df['timestamp'] = pd.to_datetime(df['timestamp'])

times = [
    pd.Timestamp("2025-03-02 15:46:00+00:00"),
    pd.Timestamp("2025-03-02 15:47:00+00:00"), 
    pd.Timestamp("2025-03-02 16:40:00+00:00"),
    pd.Timestamp("2025-03-02 16:41:00+00:00")
]

labels = ["15:46(매수신호)", "15:47(매수실행)", "16:40(매도신호)", "16:41(매도실행)"]

print("=== 4개 시점 전체 피처값 비교 ===")

# 핵심 컬럼 정의
core_cols = ['open', 'high', 'low', 'close', 'volume', 'ret_1', 'macd', 'rsi_14', 
            'bb_ma', 'bb_up', 'bb_dn', 'ema_12', 'ema_26', 'atr_14', 'vol_20']
multi_cols = ['m3_macd', 'm3_rsi_14', 'm5_macd', 'm5_rsi_14']

for i, time in enumerate(times):
    idx = df[df['timestamp'] == time].index[0]
    row = df.iloc[idx]
    
    print(f"\n=== {labels[i]} ===")
    
    # OHLCV
    print("OHLCV:")
    print(f"  Open: {row['open']:,.0f}")
    print(f"  High: {row['high']:,.0f}")  
    print(f"  Low: {row['low']:,.0f}")
    print(f"  Close: {row['close']:,.0f}")
    print(f"  Volume: {row['volume']:,.0f}")
    print(f"  1분수익률: {row['ret_1']:.6f}")
    
    # 주요 지표
    print("주요지표:")
    print(f"  MACD: {row['macd']:.2f}")
    print(f"  RSI: {row['rsi_14']:.2f}")
    print(f"  EMA12: {row['ema_12']:,.0f}")
    print(f"  EMA26: {row['ema_26']:,.0f}")
    print(f"  BB상단: {row['bb_up']:,.0f}")
    print(f"  BB중심: {row['bb_ma']:,.0f}")
    print(f"  BB하단: {row['bb_dn']:,.0f}")
    print(f"  ATR: {row['atr_14']:,.0f}")
    print(f"  변동성: {row['vol_20']:.6f}")
    
    # 멀티 타임프레임
    print("멀티TF:")
    print(f"  3분MACD: {row['m3_macd']:.2f}")
    print(f"  3분RSI: {row['m3_rsi_14']:.2f}")
    print(f"  5분MACD: {row['m5_macd']:.2f}")
    print(f"  5분RSI: {row['m5_rsi_14']:.2f}")
    
    # 신호 시점만 라벨 정보
    if i in [0, 2]:
        print(f"라벨정보:")
        print(f"  fwd_ret: {row['fwd_ret']:.6f}")
        print(f"  label: {row['label']}")

print(f"\n=== 가격 변화 요약 ===")
prices = [df.iloc[df[df['timestamp'] == t].index[0]]['close'] for t in times]
for i in range(len(prices)):
    if i == 0:
        print(f"{labels[i]}: {prices[i]:,.0f}원")
    else:
        change = ((prices[i]/prices[0])-1)*100
        print(f"{labels[i]}: {prices[i]:,.0f}원 ({change:+.2f}%)")

final_return = ((prices[3]/prices[1])-1)*100
print(f"매수→매도 수익률: {final_return:+.2f}%")