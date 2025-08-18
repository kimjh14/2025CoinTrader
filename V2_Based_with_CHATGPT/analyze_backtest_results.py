"""
백테스트 결과 분석 스크립트
artifacts/backtest 폴더의 모든 JSON 파일을 분석하여 최고/최저 수익률 조합 찾기
"""
import json
import glob
import os
import re

# ========== 분석할 날짜 필터 설정 (사용자 수정 가능) ==========
# None = 모든 날짜 분석
# [10, 181] = 10일과 181일만 분석
# [10] = 10일만 분석
DAYS_TO_ANALYZE = [10]  # 예: [10, 181] 또는 None (모두)
# ========================================================

# 모든 백테스트 JSON 파일 읽기
import pathlib
current_dir = pathlib.Path(__file__).parent.resolve()
json_files = glob.glob(str(current_dir / 'artifacts/backtest/bt_*.json'))

results = []
for file in json_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 파일명에서 조건 추출
            filename = os.path.basename(file)
            cum_return = data.get('cumulative_return', 0)
            
            # 파일명 파싱 개선 (bt_classic_200d_h3_0.002.json 또는 bt_seq_181d_h3_n20_0.002.json 형식)
            if 'bt_seq_' in filename:
                # SEQ 모드 파싱
                match = re.search(r'bt_seq_(\d+)d_h(\d+)_n(\d+)_([\d.]+)\.json', filename)
                if match:
                    days = match.group(1)
                    horizon = match.group(2)
                    n_steps = match.group(3)
                    threshold = match.group(4)
                    mode = 'seq'
                else:
                    days, horizon, threshold, n_steps, mode = 'unknown', 'unknown', 'unknown', 'unknown', 'unknown'
            else:
                # CLASSIC 모드 파싱
                match = re.search(r'bt_classic_(\d+)d_h(\d+)_([\d.]+)\.json', filename)
                if match:
                    days = match.group(1)
                    horizon = match.group(2)
                    threshold = match.group(3)
                    n_steps = 'N/A'
                    mode = 'classic'
                else:
                    parts = filename.replace('bt_classic_', '').replace('.json', '').split('_')
                    if len(parts) >= 3:
                        days = parts[0].replace('d', '')
                        horizon = parts[1].replace('h', '')
                        threshold = parts[2]
                        n_steps = 'N/A'
                        mode = 'classic'
                    else:
                        days, horizon, threshold, n_steps, mode = 'unknown', 'unknown', 'unknown', 'N/A', 'unknown'
            
            # 날짜 필터 적용
            if DAYS_TO_ANALYZE is not None:
                try:
                    if int(days) not in DAYS_TO_ANALYZE:
                        continue  # 이 파일은 건너뛰기
                except ValueError:
                    continue  # days가 숫자가 아니면 건너뛰기
            
            results.append({
                'file': filename,
                'days': days,
                'horizon': horizon,
                'threshold': threshold,
                'n_steps': n_steps,
                'mode': mode,
                'return': cum_return,
                'trades': data.get('total_trades', 0),
                'win_rate': data.get('win_rate', 0),
                'sharpe': data.get('sharpe_ratio', 0),
                'max_dd': data.get('max_drawdown', 0)
            })
    except Exception as e:
        print(f'Error reading {file}: {e}')

# 수익률 기준 정렬
results.sort(key=lambda x: x['return'], reverse=True)

print('=' * 70)
print('TOP 15 최고 수익률 조합')
print('=' * 70)
print(f'{"순위":^4} {"일수":^5} {"H":^3} {"Threshold":^10} {"수익률":>10} {"거래":>6} {"승률":>7} {"샤프":>7}')
print('-' * 70)
for i, r in enumerate(results[:15], 1):
    print(f'{i:3d}. {r["days"]:>4}d h{r["horizon"]:<2} {r["threshold"]:^10} {r["return"]*100:>9.2f}% {r["trades"]:>5}회 {r["win_rate"]*100:>6.1f}% {r["sharpe"]:>7.2f}')

print('\n' + '=' * 70)
print('BOTTOM 15 최저 수익률 조합')
print('=' * 70)
print(f'{"순위":^4} {"일수":^5} {"H":^3} {"Threshold":^10} {"수익률":>10} {"거래":>6} {"승률":>7} {"MDD":>7}')
print('-' * 70)
for i, r in enumerate(results[-15:], 1):
    print(f'{i:3d}. {r["days"]:>4}d h{r["horizon"]:<2} {r["threshold"]:^10} {r["return"]*100:>9.2f}% {r["trades"]:>5}회 {r["win_rate"]*100:>6.1f}% {r["max_dd"]*100:>6.1f}%')

print(f'\n총 {len(results)}개 백테스트 결과 분석 완료')

# 통계 분석
if results:
    positive_results = [r for r in results if r['return'] > 0]
    negative_results = [r for r in results if r['return'] < 0]
    
    print('\n' + '=' * 70)
    print('통계 요약')
    print('=' * 70)
    if DAYS_TO_ANALYZE:
        print(f'분석 대상: {DAYS_TO_ANALYZE}일 데이터만')
    else:
        print(f'분석 대상: 모든 날짜')
    print(f'양수 수익률: {len(positive_results)}개 ({len(positive_results)/len(results)*100:.1f}%)')
    print(f'음수 수익률: {len(negative_results)}개 ({len(negative_results)/len(results)*100:.1f}%)')
    print(f'평균 수익률: {sum(r["return"] for r in results)/len(results)*100:.2f}%')
    
    # Horizon별 분석
    horizon_stats = {}
    for r in results:
        h = r['horizon']
        if h not in horizon_stats:
            horizon_stats[h] = []
        horizon_stats[h].append(r['return'])
    
    print('\nHorizon별 평균 수익률:')
    for h in sorted(horizon_stats.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        avg_return = sum(horizon_stats[h])/len(horizon_stats[h])*100
        print(f'  H{h}: {avg_return:>7.2f}% ({len(horizon_stats[h])}개 결과)')
    
    # Threshold별 분석
    threshold_stats = {}
    for r in results:
        t = r['threshold']
        if t not in threshold_stats:
            threshold_stats[t] = []
        threshold_stats[t].append(r['return'])
    
    print('\nThreshold별 평균 수익률:')
    for t in sorted(threshold_stats.keys()):
        if t != 'unknown':
            avg_return = sum(threshold_stats[t])/len(threshold_stats[t])*100
            print(f'  {t:>8}: {avg_return:>7.2f}% ({len(threshold_stats[t])}개 결과)')

# 최고/최저 상세 분석
if results:
    print('\n' + '=' * 70)
    print('최고 수익 조합 상세 분석')
    print('=' * 70)
    best = results[0]
    print(f'파일: {best["file"]}')
    print(f'조건:')
    print(f'  - 데이터 기간: {best["days"]}일')
    print(f'  - 예측 Horizon: {best["horizon"]}분 후')
    print(f'  - Threshold: ±{best["threshold"]} (라벨링 임계값)')
    print(f'성과:')
    print(f'  - 누적 수익률: {best["return"]*100:.2f}%')
    print(f'  - 총 거래: {best["trades"]}회')
    print(f'  - 승률: {best["win_rate"]*100:.1f}%')
    print(f'  - 샤프 비율: {best["sharpe"]:.2f}')
    print(f'  - 최대 낙폭: {best["max_dd"]*100:.1f}%')
    
    print('\n' + '=' * 70)
    print('최저 수익 조합 상세 분석')
    print('=' * 70)
    worst = results[-1]
    print(f'파일: {worst["file"]}')
    print(f'조건:')
    print(f'  - 데이터 기간: {worst["days"]}일')
    print(f'  - 예측 Horizon: {worst["horizon"]}분 후')
    print(f'  - Threshold: ±{worst["threshold"]} (라벨링 임계값)')
    print(f'성과:')
    print(f'  - 누적 수익률: {worst["return"]*100:.2f}%')
    print(f'  - 총 거래: {worst["trades"]}회')
    print(f'  - 승률: {worst["win_rate"]*100:.1f}%')
    print(f'  - 최대 낙폭: {worst["max_dd"]*100:.1f}%')