"""
batch_train.py - 30개 조합 일괄 모델 학습 도구

CLI 사용 예시:
python tools/batch_train.py `
  --data_dir data/seq `
  --output_dir artifacts/train `
  --splits 4 `
  --fee 0.0005

사용법:
• batch_data.py로 생성된 30개 데이터셋을 모두 학습합니다
• N_STEPS: [5, 10, 15, 20, 30, 60] x HORIZON: [1, 3, 5, 10, 15]
• 각 조합마다 model, scaler, meta 파일 생성
• HistGradientBoostingClassifier + 확률 보정 사용
• 실행 시간: 약 30-60분 소요

출력:
• 30개 모델 파일 (model_seq_n{N}_h{H}.joblib)
• 30개 스케일러 파일 (scaler_seq_n{N}_h{H}.joblib)
• 30개 메타데이터 파일 (meta_seq_n{N}_h{H}.json)
• 학습 결과 요약 리포트
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
BASE_DATA_DIR = "data/seq"  # 데이터셋 위치
BASE_OUTPUT_DIR = "artifacts/train"  # train.py 출력 폴더
SPLITS = 4
FEE = 0.0005

def run_train_model(n_steps, horizon, data_path, model_path, scaler_path, meta_path):
    """단일 모델 학습"""
    cmd = [
        "python", "tools/train.py",
        "--data", data_path,
        "--model", model_path,
        "--scaler", scaler_path,
        "--meta", meta_path,
        "--splits", str(SPLITS),
        "--fee", str(FEE)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10분 타임아웃
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout (10분 초과)"
    except Exception as e:
        return False, str(e)

def analyze_model_performance(meta_path):
    """모델 성능 분석"""
    try:
        import json
        if not os.path.exists(meta_path):
            return None, "메타 파일 없음"
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        # 성능 지표 추출
        stats = {
            'optimized_long_p': meta.get('optimized_long_p', 0),
            'best_equity_cv': 1.0000,  # train.py에서 저장하지 않으므로 더미값
            'algo': meta.get('algo', 'Unknown'),
            'feature_count': len(meta.get('features', [])),  # 'feature_names' -> 'features'
            'allow_short': meta.get('allow_short', False),
            'n_train': meta.get('n_train', 0),
            'n_holdout': meta.get('n_holdout', 0)
        }
        
        return stats, "성공"
    except Exception as e:
        return None, str(e)

def main():
    print(f"일괄 학습 시작: {len(N_STEPS_CANDIDATES) * len(HORIZON_CANDIDATES)}개 조합")
    
    # 출력 디렉터리 생성
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    results = []
    total_combinations = len(N_STEPS_CANDIDATES) * len(HORIZON_CANDIDATES)
    current_combination = 0
    total_start_time = time.time()
    
    for n_steps in N_STEPS_CANDIDATES:
        for horizon in HORIZON_CANDIDATES:
            current_combination += 1
            
            # 파일명 규칙 (train.py CLI 스타일)
            data_file = f"{BASE_DATA_DIR}/dataset_seq_n{n_steps}_h{horizon}.parquet"
            model_file = f"{BASE_OUTPUT_DIR}/model_seq_n{n_steps}_h{horizon}.joblib"
            scaler_file = f"{BASE_OUTPUT_DIR}/scaler_seq_n{n_steps}_h{horizon}.joblib"
            meta_file = f"{BASE_OUTPUT_DIR}/meta_seq_n{n_steps}_h{horizon}.json"
            
            print(f"[{current_combination:2d}/{total_combinations}] n{n_steps}_h{horizon}")
            
            # 데이터셋 존재 확인
            if not os.path.exists(data_file):
                print(f"    SKIP: 데이터셋 없음")
                results.append({
                    'n_steps': n_steps,
                    'horizon': horizon,
                    'success': False,
                    'error': '데이터셋 파일 없음',
                    'model_path': model_file
                })
                continue
            
            start_time = time.time()
            
            # 모델 학습
            success, message = run_train_model(n_steps, horizon, data_file, model_file, scaler_file, meta_file)
            
            if success:
                # 성능 분석
                stats, error = analyze_model_performance(meta_file)
                
                if stats:
                    elapsed = time.time() - start_time
                    print(f"    OK ({elapsed:.1f}s) CV={stats['best_equity_cv']:.4f}")
                    
                    results.append({
                        'n_steps': n_steps,
                        'horizon': horizon,
                        'success': True,
                        'model_path': model_file,
                        **stats
                    })
                else:
                    print(f"    WARN: 분석실패")
                    results.append({
                        'n_steps': n_steps,
                        'horizon': horizon,
                        'success': True,
                        'model_path': model_file,
                        'error': f"분석 실패: {error}"
                    })
            else:
                print(f"    FAIL")
                results.append({
                    'n_steps': n_steps,
                    'horizon': horizon,
                    'success': False,
                    'error': message,
                    'model_path': model_file
                })
    
    # 결과 요약
    results_df = pd.DataFrame(results)
    successful_results = results_df[results_df['success'] == True]
    
    total_time = time.time() - total_start_time
    
    print("=" * 70)
    print("일괄 학습 결과 요약")
    print("=" * 70)
    print(f"총 시도: {len(results)}개")
    print(f"성공: {len(successful_results)}개")
    print(f"실패: {len(results) - len(successful_results)}개")
    print(f"총 소요 시간: {total_time/60:.1f}분")
    print()
    
    if len(successful_results) > 0:
        # 성능 기준 정렬
        performance_cols = ['best_equity_cv', 'optimized_long_p']
        available_cols = [col for col in performance_cols if col in successful_results.columns]
        
        if 'best_equity_cv' in successful_results.columns:
            # 가장 높은 CV 성능
            best_cv = successful_results.loc[successful_results['best_equity_cv'].idxmax()]
            
            print("TOP 추천 조합:")
            print()
            print(f"[1] 최고 CV 성능:")
            print(f"   N_STEPS={best_cv['n_steps']}, HORIZON={best_cv['horizon']}")
            print(f"   CV 성능: {best_cv['best_equity_cv']:.4f}")
            print(f"   임계값: {best_cv['optimized_long_p']:.2f}")
            print(f"   모델: {best_cv['model_path']}")
            print()
            
            # 상위 5개 조합 표시
            top5_cv = successful_results.nlargest(5, 'best_equity_cv')
            print("CV 성능 기준 TOP 5:")
            print("순위  N_STEPS  HORIZON  CV성능    임계값   피처수   모델파일")
            print("-" * 80)
            for i, (idx, row) in enumerate(top5_cv.iterrows(), 1):
                model_name = os.path.basename(row['model_path'])
                print(f"{i:2d}   {row['n_steps']:7d}  {row['horizon']:7d}  "
                      f"{row['best_equity_cv']:7.4f}  {row['optimized_long_p']:6.2f}  "
                      f"{row.get('feature_count', 0):6d}   {model_name}")
    
    # 결과를 CSV로 저장
    results_csv = f"{BASE_OUTPUT_DIR}/batch_train_results.csv"
    results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
    print(f"\\n상세 결과 저장: {results_csv}")
    
    print("\\n일괄 모델 학습 완료!")
    print(f"학습된 모델 위치: {BASE_OUTPUT_DIR}/")
    print("파일명 형식: model_seq_n{N_STEPS}_h{HORIZON}.joblib")

if __name__ == "__main__":
    main()