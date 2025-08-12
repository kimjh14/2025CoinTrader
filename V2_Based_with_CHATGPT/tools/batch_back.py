"""
batch_back.py - 30개 조합 일괄 백테스트 도구

CLI 사용 예시:
python tools/batch_back.py `
  --data_dir data/seq `
  --model_dir artifacts/train `
  --output_dir artifacts/backtest `
  --fee 0.0005

사용법:
• batch_train.py로 학습된 30개 모델을 모두 백테스트합니다
• N_STEPS: [5, 10, 15, 20, 30, 60] x HORIZON: [1, 3, 5, 10, 15]
• 각 조합마다 성과 리포트와 거래 로그 생성
• 자동 마크다운 리포트 생성 포함
• 실행 시간: 약 20-40분 소요

출력:
• 30개 JSON 리포트 (bt_seq_n{N}_h{H}.json)
• 30개 CSV 거래 로그 (trades_seq_n{N}_h{H}.csv)
• 30개 마크다운 리포트 (trades_seq_n{N}_h{H}.md)
• 종합 성과 비교 표 (summary_results.csv)
"""

import os
import subprocess
import pandas as pd
import time
import json
from pathlib import Path

# 그리드 서치 파라미터
N_STEPS_CANDIDATES = [5, 10, 15, 20, 30, 60]
HORIZON_CANDIDATES = [1, 3, 5, 10, 15]

# 고정 파라미터
BASE_DATA_DIR = "data/seq"  # 데이터셋 위치
BASE_MODEL_DIR = "artifacts/train"  # 학습된 모델 위치
BASE_OUTPUT_DIR = "artifacts/backtest"  # backtest.py 출력 폴더
FEE = 0.0005

def run_backtest_model(n_steps, horizon, data_path, model_path, meta_path, scaler_path, 
                      report_path, csv_path, md_path):
    """단일 모델 백테스트"""
    cmd = [
        "python", "tools/backtest.py",
        "--data", data_path,
        "--model", model_path,
        "--meta", meta_path,
        "--scaler", scaler_path,
        "--fee", str(FEE),
        "--report", report_path,
        "--log_csv", csv_path,
        "--auto_md",
        "--md_out", md_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5분 타임아웃
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout (5분 초과)"
    except Exception as e:
        return False, str(e)

def analyze_backtest_performance(report_path):
    """백테스트 성능 분석"""
    try:
        if not os.path.exists(report_path):
            return None, "리포트 파일 없음"
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        # 성능 지표 추출
        stats = {
            'final_equity': report.get('final_equity', 0),
            'hold_final': report.get('hold_final', 0),
            'n_trades': report.get('n_trades', 0),
            'n_obs': report.get('n_obs', 0),
            'start': report.get('start', ''),
            'end': report.get('end', ''),
            'outperformance': report.get('final_equity', 0) - report.get('hold_final', 0)
        }
        
        # 임계값 정보
        if 'thresholds' in report:
            th = report['thresholds']
            stats['long_p'] = th.get('long_p', 0)
            stats['short_p'] = th.get('short_p', 0)
            stats['allow_short'] = th.get('allow_short', False)
        
        return stats, "성공"
    except Exception as e:
        return None, str(e)

def main():
    print(f"일괄 백테스트 시작: {len(N_STEPS_CANDIDATES) * len(HORIZON_CANDIDATES)}개 조합")
    
    # 출력 디렉터리 생성
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    results = []
    total_combinations = len(N_STEPS_CANDIDATES) * len(HORIZON_CANDIDATES)
    current_combination = 0
    total_start_time = time.time()
    
    for n_steps in N_STEPS_CANDIDATES:
        for horizon in HORIZON_CANDIDATES:
            current_combination += 1
            
            # 파일명 규칙 (backtest.py CLI 스타일)
            data_file = f"{BASE_DATA_DIR}/dataset_seq_n{n_steps}_h{horizon}.parquet"
            model_file = f"{BASE_MODEL_DIR}/model_seq_n{n_steps}_h{horizon}.joblib"
            meta_file = f"{BASE_MODEL_DIR}/meta_seq_n{n_steps}_h{horizon}.json"
            scaler_file = f"{BASE_MODEL_DIR}/scaler_seq_n{n_steps}_h{horizon}.joblib"
            
            report_file = f"{BASE_OUTPUT_DIR}/bt_seq_n{n_steps}_h{horizon}.json"
            csv_file = f"{BASE_OUTPUT_DIR}/trades_seq_n{n_steps}_h{horizon}.csv"
            md_file = f"{BASE_OUTPUT_DIR}/trades_seq_n{n_steps}_h{horizon}.md"
            
            print(f"[{current_combination:2d}/{total_combinations}] n{n_steps}_h{horizon}")
            
            # 필수 파일 존재 확인
            missing_files = []
            required_files = {
                '데이터': data_file,
                '모델': model_file, 
                '메타': meta_file,
                '스케일러': scaler_file
            }
            
            for name, path in required_files.items():
                if not os.path.exists(path):
                    missing_files.append(name)
            
            if missing_files:
                print(f"    SKIP: {','.join(missing_files)} 없음")
                results.append({
                    'n_steps': n_steps,
                    'horizon': horizon,
                    'success': False,
                    'error': f"파일 없음: {', '.join(missing_files)}",
                    'report_path': report_file
                })
                continue
            
            start_time = time.time()
            
            # 백테스트 실행
            success, message = run_backtest_model(
                n_steps, horizon, data_file, model_file, meta_file, scaler_file,
                report_file, csv_file, md_file
            )
            
            if success:
                # 성능 분석
                stats, error = analyze_backtest_performance(report_file)
                
                if stats:
                    elapsed = time.time() - start_time
                    print(f"    OK ({elapsed:.1f}s) 배수={stats['final_equity']:.4f} 거래={stats['n_trades']}")
                    
                    results.append({
                        'n_steps': n_steps,
                        'horizon': horizon,
                        'success': True,
                        'report_path': report_file,
                        **stats
                    })
                else:
                    print(f"    WARN: 분석실패")
                    results.append({
                        'n_steps': n_steps,
                        'horizon': horizon,
                        'success': True,
                        'report_path': report_file,
                        'error': f"분석 실패: {error}"
                    })
            else:
                print(f"    FAIL")
                results.append({
                    'n_steps': n_steps,
                    'horizon': horizon,
                    'success': False,
                    'error': message,
                    'report_path': report_file
                })
    
    # 결과 요약
    results_df = pd.DataFrame(results)
    successful_results = results_df[results_df['success'] == True]
    
    total_time = time.time() - total_start_time
    
    print("=" * 70)
    print("일괄 백테스트 결과 요약")
    print("=" * 70)
    print(f"총 시도: {len(results)}개")
    print(f"성공: {len(successful_results)}개")
    print(f"실패: {len(results) - len(successful_results)}개")
    print(f"총 소요 시간: {total_time/60:.1f}분")
    print()
    
    if len(successful_results) > 0:
        # 성능 기준 분석
        performance_cols = ['final_equity', 'outperformance', 'n_trades']
        available_cols = [col for col in performance_cols if col in successful_results.columns]
        
        if 'final_equity' in successful_results.columns:
            # 가장 높은 최종 수익률
            best_equity = successful_results.loc[successful_results['final_equity'].idxmax()]
            
            # 가장 높은 아웃퍼폼
            if 'outperformance' in successful_results.columns:
                best_outperf = successful_results.loc[successful_results['outperformance'].idxmax()]
            else:
                best_outperf = best_equity
            
            print("TOP 추천 조합:")
            print()
            print(f"[1] 최고 수익률:")
            print(f"   N_STEPS={best_equity['n_steps']}, HORIZON={best_equity['horizon']}")
            print(f"   최종 배수: {best_equity['final_equity']:.4f}")
            print(f"   vs 홀딩: {best_equity.get('hold_final', 0):.4f}")
            print(f"   거래 수: {best_equity.get('n_trades', 0)}")
            print(f"   리포트: {best_equity['report_path']}")
            print()
            
            if best_outperf['n_steps'] != best_equity['n_steps'] or best_outperf['horizon'] != best_equity['horizon']:
                print(f"[2] 최고 아웃퍼폼:")
                print(f"   N_STEPS={best_outperf['n_steps']}, HORIZON={best_outperf['horizon']}")
                print(f"   아웃퍼폼: {best_outperf.get('outperformance', 0):+.4f}")
                print(f"   최종 배수: {best_outperf['final_equity']:.4f}")
                print(f"   거래 수: {best_outperf.get('n_trades', 0)}")
                print()
            
            # 상위 5개 조합 표시
            top5_equity = successful_results.nlargest(5, 'final_equity')
            print("최종 수익률 기준 TOP 5:")
            print("순위  N_STEPS  HORIZON  최종배수  vs홀딩   아웃퍼폼   거래수")
            print("-" * 65)
            for i, (idx, row) in enumerate(top5_equity.iterrows(), 1):
                outperf = row.get('outperformance', 0)
                print(f"{i:2d}   {row['n_steps']:7d}  {row['horizon']:7d}  "
                      f"{row['final_equity']:7.4f}  {row.get('hold_final', 0):6.4f}  "
                      f"{outperf:+7.4f}  {row.get('n_trades', 0):6d}")
    
    # 결과를 CSV로 저장
    results_csv = f"{BASE_OUTPUT_DIR}/batch_backtest_results.csv"
    results_df.to_csv(results_csv, index=False, encoding='utf-8-sig')
    print(f"\\n상세 결과 저장: {results_csv}")
    
    print("\\n일괄 백테스트 완료!")
    print(f"백테스트 결과 위치: {BASE_OUTPUT_DIR}/")
    print("파일명 형식: bt_seq_n{N_STEPS}_h{HORIZON}.json")

if __name__ == "__main__":
    main()