"""
batch_data.py - 30개 조합 일괄 데이터셋 생성 도구

CLI 사용 예시:
python tools/batch_data.py `
  --input data/raw/krw_btc_1m_180d.parquet `
  --output_dir data/seq `
  --up 0.0005 `
  --dn -0.0005

사용법:
• N_STEPS x HORIZON 30개 조합으로 데이터셋을 일괄 생성합니다
• N_STEPS: [5, 10, 15, 20, 30, 60] - 입력 패턴 길이
• HORIZON: [1, 3, 5, 10, 15] - 예측 시점 (분)
• 각 조합마다 dataset_seq_n{N}_h{H}.parquet 파일 생성
• 실행 시간: 약 15-30분 소요

출력:
• 30개 데이터셋 파일
• 생성 결과 요약 리포트
• 실패한 조합 목록
"""

import os
import subprocess
import pandas as pd
import time
from pathlib import Path

# 그리드 서치 파라미터
N_STEPS_CANDIDATES = [5, 10, 15, 20, 30, 60]
HORIZON_CANDIDATES = [1, 3, 5, 10, 15]

# 고정 파라미터
INPUT_FILE = "data/raw/krw_btc_1m_5d.parquet"
UP_THRESHOLD = 0.0005  # ±0.05%
DN_THRESHOLD = -0.0005
BASE_OUTPUT_DIR = "data/seq"  # build_dataset.py와 동일한 출력 폴더

def run_build_dataset(n_steps, horizon, output_path):
    """단일 데이터셋 생성"""
    cmd = [
        "python", "tools/build_dataset.py",
        "--mode", "seq",
        "--in", INPUT_FILE,
        "--out", output_path,
        "--n_steps", str(n_steps),
        "--horizon", str(horizon),
        "--up", str(UP_THRESHOLD),
        "--dn", str(DN_THRESHOLD),
        "--ta"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout (5분 초과)"
    except Exception as e:
        return False, str(e)

def analyze_dataset_distribution(file_path):
    """데이터셋 클래스 분포 분석"""
    try:
        df = pd.read_parquet(file_path)
        if len(df) == 0:
            return None, "빈 데이터셋"
        
        # 클래스 분포 계산
        label_counts = df['label'].value_counts(normalize=True).sort_index()
        
        # 균형도 계산 (표준편차가 낮을수록 균형잡힘)
        balance_score = label_counts.std()
        
        # 최소 클래스 비율 (너무 작으면 학습 어려움)
        min_class_ratio = label_counts.min()
        
        return {
            'total_rows': len(df),
            'flat_ratio': label_counts.get('flat', 0),
            'long_ratio': label_counts.get('long', 0), 
            'short_ratio': label_counts.get('short', 0),
            'balance_score': balance_score,
            'min_class_ratio': min_class_ratio
        }, "성공"
    except Exception as e:
        return None, str(e)

def main():
    print("30개 조합 일괄 데이터셋 생성 시작")
    print(f"입력 파일: {INPUT_FILE}")
    print(f"N_STEPS 후보: {N_STEPS_CANDIDATES}")
    print(f"HORIZON 후보: {HORIZON_CANDIDATES}")
    print(f"총 조합 수: {len(N_STEPS_CANDIDATES) * len(HORIZON_CANDIDATES)}개")
    print()
    
    # 입력 파일 존재 확인
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] 입력 파일이 없습니다: {INPUT_FILE}")
        print("먼저 다음 명령으로 5일 데이터를 수집하세요:")
        print("python tools/collect.py --market KRW-BTC --minutes 1 --days 5 --out data/raw/krw_btc_1m_5d.parquet")
        return
    
    # 출력 디렉터리 생성
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    results = []
    total_combinations = len(N_STEPS_CANDIDATES) * len(HORIZON_CANDIDATES)
    current_combination = 0
    total_start_time = time.time()
    
    for n_steps in N_STEPS_CANDIDATES:
        for horizon in HORIZON_CANDIDATES:
            current_combination += 1
            
            # 출력 파일명 (build_dataset.py 스타일)
            output_file = f"{BASE_OUTPUT_DIR}/dataset_seq_n{n_steps}_h{horizon}.parquet"
            
            print(f"[{current_combination:2d}/{total_combinations}] N_STEPS={n_steps:2d}, HORIZON={horizon:2d} 처리 중...")
            
            start_time = time.time()
            
            # 데이터셋 생성
            success, message = run_build_dataset(n_steps, horizon, output_file)
            
            if success:
                # 클래스 분포 분석
                stats, error = analyze_dataset_distribution(output_file)
                
                if stats:
                    elapsed = time.time() - start_time
                    total_elapsed = time.time() - total_start_time
                    remaining = total_combinations - current_combination
                    avg_time = total_elapsed / current_combination
                    eta = remaining * avg_time
                    
                    print(f"    [OK] 완료 ({elapsed:.1f}s) | "
                          f"총 {stats['total_rows']:,}행 | "
                          f"균형도: {stats['balance_score']:.3f}")
                    print(f"       분포 - Flat: {stats['flat_ratio']:.1%}, "
                          f"Long: {stats['long_ratio']:.1%}, "
                          f"Short: {stats['short_ratio']:.1%}")
                    print(f"       진행률: {current_combination}/{total_combinations} | "
                          f"ETA: {eta/60:.1f}분")
                    
                    results.append({
                        'n_steps': n_steps,
                        'horizon': horizon,
                        'file_path': output_file,
                        'success': True,
                        **stats
                    })
                else:
                    print(f"    [FAIL] 분석 실패: {error}")
                    results.append({
                        'n_steps': n_steps,
                        'horizon': horizon, 
                        'file_path': output_file,
                        'success': False,
                        'error': error
                    })
            else:
                print(f"    [FAIL] 생성 실패: {message}")
                results.append({
                    'n_steps': n_steps,
                    'horizon': horizon,
                    'file_path': output_file,
                    'success': False,
                    'error': message
                })
            
            print()
    
    # 결과 요약
    results_df = pd.DataFrame(results)
    successful_results = results_df[results_df['success'] == True]
    
    total_time = time.time() - total_start_time
    
    print("=" * 70)
    print("일괄 생성 결과 요약")
    print("=" * 70)
    print(f"총 시도: {len(results)}개")
    print(f"성공: {len(successful_results)}개")
    print(f"실패: {len(results) - len(successful_results)}개")
    print(f"총 소요 시간: {total_time/60:.1f}분")
    print()
    
    if len(successful_results) > 0:
        # 가장 균형잡힌 조합 찾기 (balance_score가 낮을수록 좋음)
        best_balanced = successful_results.loc[successful_results['balance_score'].idxmin()]
        
        # 최소 클래스 비율이 가장 높은 조합 (모든 클래스가 충분히 존재)
        best_min_class = successful_results.loc[successful_results['min_class_ratio'].idxmax()]
        
        print("TOP 추천 조합:")
        print()
        print(f"[1] 가장 균형잡힌 분포:")
        print(f"   N_STEPS={best_balanced['n_steps']}, HORIZON={best_balanced['horizon']}")
        print(f"   균형도: {best_balanced['balance_score']:.3f}")
        print(f"   분포 - Flat: {best_balanced['flat_ratio']:.1%}, "
              f"Long: {best_balanced['long_ratio']:.1%}, "
              f"Short: {best_balanced['short_ratio']:.1%}")
        print(f"   파일: {best_balanced['file_path']}")
        print()
        
        print(f"[2] 최소 클래스 비율이 가장 높음:")
        print(f"   N_STEPS={best_min_class['n_steps']}, HORIZON={best_min_class['horizon']}")
        print(f"   최소 클래스: {best_min_class['min_class_ratio']:.1%}")
        print(f"   분포 - Flat: {best_min_class['flat_ratio']:.1%}, "
              f"Long: {best_min_class['long_ratio']:.1%}, "
              f"Short: {best_min_class['short_ratio']:.1%}")
        print(f"   파일: {best_min_class['file_path']}")
        print()
        
        # 상위 5개 조합 표시
        top5_balanced = successful_results.nsmallest(5, 'balance_score')
        print("균형도 기준 TOP 5:")
        print("순위  N_STEPS  HORIZON  균형도    Flat    Long   Short   최소클래스  데이터수")
        print("-" * 80)
        for i, (idx, row) in enumerate(top5_balanced.iterrows(), 1):
            print(f"{i:2d}   {row['n_steps']:7d}  {row['horizon']:7d}  "
                  f"{row['balance_score']:6.3f}  {row['flat_ratio']:6.1%}  "
                  f"{row['long_ratio']:6.1%}  {row['short_ratio']:6.1%}  "
                  f"{row['min_class_ratio']:8.1%}  {row['total_rows']:7,d}")
    
    # 결과를 CSV로 저장
    results_csv = f"{BASE_OUTPUT_DIR}/batch_results.csv"
    results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
    print(f"\n상세 결과 저장: {results_csv}")
    
    print("\n일괄 데이터셋 생성 완료!")
    print(f"생성된 파일 위치: {BASE_OUTPUT_DIR}/")
    print("파일명 형식: dataset_seq_n{N_STEPS}_h{HORIZON}.parquet")

if __name__ == "__main__":
    main()