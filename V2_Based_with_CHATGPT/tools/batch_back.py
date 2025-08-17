"""
batch_back.py - 일괄 백테스트 도구

CLI 사용 예시:

# SEQ 모드
python tools/batch_back.py `
  --mode seq `
  --data_dir data/seq `
  --model_dir artifacts/train `
  --output_dir artifacts/backtest `
  --fee 0.0005

# CLASSIC 모드



사용법:
• batch_train.py로 학습된 모델을 모두 백테스트합니다
• 파일명 패턴을 자동으로 감지하여 처리
• 각 조합마다 성과 리포트와 거래 로그 생성
• 자동 마크다운 리포트 생성 포함

출력:
• JSON 리포트 (bt_{mode}_{params}.json)
• CSV 거래 로그 (trades_{mode}_{params}.csv)
• 마크다운 리포트 (trades_{mode}_{params}.md)
• 종합 성과 비교 표 (summary_results_{mode}.csv)
"""

import os
import sys
import subprocess
import glob
import re
import argparse
from pathlib import Path

# 현재 스크립트의 절대 경로 기준으로 프로젝트 루트 찾기
SCRIPT_DIR = Path(__file__).resolve().parent  # tools 폴더
ROOT_DIR = SCRIPT_DIR.parent  # V2_Based_with_CHATGPT 폴더

# batch_data.py에서 설정 임포트
sys.path.insert(0, str(SCRIPT_DIR))
from batch_data import (
    DAYS_TO_PROCESS,
    HORIZON_CANDIDATES,
    THRESHOLD_CANDIDATES,
    N_STEPS_CANDIDATES
)

# ========== 배치 파라미터 설정 (batch_data.py의 값 사용) ==========

# 백테스트할 조합 선택 (batch_data.py의 설정 따름)
DAYS_TO_BACKTEST = DAYS_TO_PROCESS  # batch_data.py의 DAYS_TO_PROCESS 사용
HORIZON_TO_BACKTEST = HORIZON_CANDIDATES  # batch_data.py의 HORIZON_CANDIDATES 사용
THRESHOLD_TO_BACKTEST = THRESHOLD_CANDIDATES  # batch_data.py의 THRESHOLD_CANDIDATES 사용

# SEQ 모드 전용
N_STEPS_TO_BACKTEST = N_STEPS_CANDIDATES  # batch_data.py의 N_STEPS_CANDIDATES 사용

# 기본 파라미터
DEFAULT_FEE = 0.0005

# ========================================================

def extract_params_from_model_filename(filename, mode):
    """모델 파일명에서 파라미터 추출"""
    if mode == "seq":
        # model_seq_181d_h1_n20_0.001.joblib (one_click_pipeline.py 형식)
        match = re.search(r'model_seq_(\d+)d_h(\d+)_n(\d+)_([\d.]+)\.joblib', filename)
        if match:
            return {
                'days': int(match.group(1)),
                'horizon': int(match.group(2)),
                'n_steps': int(match.group(3)),
                'threshold': float(match.group(4)),
                'threshold_str': match.group(4)
            }
    else:  # classic
        # model_classic_181d_h1_0.001.joblib
        match = re.search(r'model_classic_(\d+)d_h(\d+)_([\d.]+)\.joblib', filename)
        if match:
            return {
                'days': int(match.group(1)),
                'horizon': int(match.group(2)),
                'threshold': float(match.group(3)),
                'threshold_str': match.group(3)
            }
    return None

def should_process_model(params, mode):
    """파라미터 기반으로 모델 처리 여부 결정"""
    # Days 필터
    if DAYS_TO_BACKTEST is not None and params['days'] not in DAYS_TO_BACKTEST:
        return False
    
    # Horizon 필터
    if HORIZON_TO_BACKTEST is not None and params['horizon'] not in HORIZON_TO_BACKTEST:
        return False
    
    # Threshold 필터
    if THRESHOLD_TO_BACKTEST is not None:
        # 부동소수점 비교를 위해 근사값 비교
        threshold_match = any(abs(params['threshold'] - t) < 0.0001 for t in THRESHOLD_TO_BACKTEST)
        if not threshold_match:
            return False
    
    # SEQ 모드 전용 N_STEPS 필터
    if mode == "seq" and N_STEPS_TO_BACKTEST is not None:
        if params['n_steps'] not in N_STEPS_TO_BACKTEST:
            return False
    
    return True

def find_corresponding_files(model_file, params, mode, data_dir, model_dir):
    """모델 파일에 대응하는 데이터셋, 메타, 스케일러 파일 찾기"""
    base_dir = os.path.dirname(model_file)
    base_name = os.path.basename(model_file).replace('model_', '').replace('.joblib', '')
    
    # 스케일러와 메타 파일
    scaler_file = os.path.join(base_dir, f"scaler_{base_name}.joblib")
    meta_file = os.path.join(base_dir, f"meta_{base_name}.json")
    
    # 데이터셋 파일 (one_click_pipeline.py 형식)
    if mode == "seq":
        data_filename = f"dataset_seq_{params['days']}d_h{params['horizon']}_n{params['n_steps']}_{params['threshold_str']}.parquet"
    else:
        data_filename = f"dataset_mtf_{params['days']}d_h{params['horizon']}_{params['threshold_str']}.parquet"
    
    data_file = os.path.join(data_dir, data_filename)
    
    return {
        'data': data_file,
        'model': model_file,
        'scaler': scaler_file,
        'meta': meta_file
    }

def run_backtest_model(data_path, model_path, meta_path, scaler_path, 
                      report_path, csv_path, md_path, fee):
    """단일 모델 백테스트"""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "backtest.py"),
        "--data", data_path,
        "--model", model_path,
        "--meta", meta_path,
        "--scaler", scaler_path,
        "--fee", str(fee),
        "--report", report_path,
        "--log_csv", csv_path,
        "--auto_md",
        "--md_out", md_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT_DIR))  # 5분 타임아웃
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout (5분 초과)"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="일괄 백테스트 도구")
    parser.add_argument("--mode", type=str, required=True, choices=["seq", "classic"],
                      help="백테스트 모드: seq 또는 classic")
    parser.add_argument("--data_dir", type=str, required=True,
                      help="데이터셋 디렉토리")
    parser.add_argument("--model_dir", type=str, default="artifacts/train",
                      help="모델 디렉토리")
    parser.add_argument("--output_dir", type=str, default="artifacts/backtest",
                      help="출력 디렉토리")
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE,
                      help="거래 수수료")
    
    args = parser.parse_args()
    
    
    # 출력 디렉터리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 파일 검색
    if args.mode == "seq":
        pattern = os.path.join(args.model_dir, "model_seq_*.joblib")
    else:
        pattern = os.path.join(args.model_dir, "model_classic_*.joblib")
    
    all_model_files = glob.glob(pattern)
    # 필터링
    model_files = []
    for file in all_model_files:
        filename = os.path.basename(file)
        params = extract_params_from_model_filename(filename, args.mode)
        if params and should_process_model(params, args.mode):
            model_files.append(file)
    
    print(f"{args.mode.upper()} 모드: {len(model_files)}개 모델 백테스트 시작")
    
    if not model_files:
        print("[ERROR] 모델 파일이 없습니다.")
        print(f"검색 패턴: {pattern}")
        return
    
    
    for idx, model_file in enumerate(sorted(model_files), 1):
        filename = os.path.basename(model_file)
        params = extract_params_from_model_filename(filename, args.mode)
        
        if not params:
            print(f"[{idx}/{len(model_files)}] {filename} - 파일명 파싱 실패")
            continue
        
        # 대응 파일들 찾기
        files = find_corresponding_files(model_file, params, args.mode, args.data_dir, args.model_dir)
        
        # 필수 파일 존재 확인
        missing_files = []
        for name, path in files.items():
            if not os.path.exists(path):
                missing_files.append(name)
        
        if missing_files:
            print(f"[{idx:3d}/{len(model_files)}] 처리 중... [SKIP - 파일 누락]")
            continue
        
        # 출력 파일명 생성 (one_click_pipeline.py 형식)
        if args.mode == "seq":
            base_name = f"seq_{params['days']}d_h{params['horizon']}_n{params['n_steps']}_{params['threshold_str']}"
            display_name = f"d{params['days']}_h{params['horizon']}_n{params['n_steps']}_t{params['threshold_str']}"
        else:
            base_name = f"classic_{params['days']}d_h{params['horizon']}_{params['threshold_str']}"
            display_name = f"d{params['days']}_h{params['horizon']}_t{params['threshold_str']}"
        
        report_file = os.path.join(args.output_dir, f"bt_{base_name}.json")
        csv_file = os.path.join(args.output_dir, f"trades_{base_name}.csv")
        md_file = os.path.join(args.output_dir, f"trades_{base_name}.md")
        
        print(f"[{idx:3d}/{len(model_files)}] 처리 중...", end=" ")
        
        # 백테스트 실행
        success, message = run_backtest_model(
            files['data'], files['model'], files['meta'], files['scaler'],
            report_file, csv_file, md_file, args.fee
        )
        
        if success:
            print(f"[OK]")
        else:
            print(f"[FAIL]")
    
    print(f"\n배치 완료")

if __name__ == "__main__":
    main()