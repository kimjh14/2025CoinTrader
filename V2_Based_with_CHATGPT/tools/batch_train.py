"""
batch_train.py - 일괄 모델 학습 도구

CLI 사용 예시:

# SEQ 모드
python tools/batch_train.py `
  --mode seq `
  --data_dir data/seq `
  --output_dir artifacts/train `
  --splits 4 `
  --fee 0.0005

# CLASSIC 모드
python tools/batch_train.py `
  --mode classic `
  --data_dir data/classic `
  --output_dir artifacts/train `
  --splits 4 `
  --fee 0.0005 `
  --allow_short

사용법:
• batch_data.py로 생성된 데이터셋 중 선택된 조합만 학습합니다
• 상단 변수 설정으로 원하는 조합만 선택 가능
• 각 조합마다 model, scaler, meta 파일 생성
• HistGradientBoostingClassifier + 확률 보정 사용

출력:
• 모델 파일 (model_{mode}_{params}.joblib)
• 스케일러 파일 (scaler_{mode}_{params}.joblib)
• 메타데이터 파일 (meta_{mode}_{params}.json)
• 학습 결과 요약 리포트
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

# 학습할 조합 선택 (batch_data.py의 설정 따름)
DAYS_TO_TRAIN = DAYS_TO_PROCESS  # batch_data.py의 DAYS_TO_PROCESS 사용
HORIZON_TO_TRAIN = HORIZON_CANDIDATES  # batch_data.py의 HORIZON_CANDIDATES 사용
THRESHOLD_TO_TRAIN = THRESHOLD_CANDIDATES  # batch_data.py의 THRESHOLD_CANDIDATES 사용

# SEQ 모드 전용
N_STEPS_TO_TRAIN = N_STEPS_CANDIDATES  # batch_data.py의 N_STEPS_CANDIDATES 사용

# 기본 파라미터
DEFAULT_SPLITS = 4
DEFAULT_FEE = 0.0005

# ========================================================

def extract_params_from_filename(filename, mode):
    """파일명에서 파라미터 추출"""
    if mode == "seq":
        # dataset_seq_181d_h1_n20_0.001.parquet (one_click_pipeline.py 형식)
        match = re.search(r'dataset_seq_(\d+)d_h(\d+)_n(\d+)_([\d.]+)\.parquet', filename)
        if match:
            return {
                'days': int(match.group(1)),
                'horizon': int(match.group(2)),
                'n_steps': int(match.group(3)),
                'threshold': float(match.group(4)),
                'threshold_str': match.group(4)
            }
    else:  # classic
        # dataset_mtf_181d_h1_0.001.parquet
        match = re.search(r'dataset_mtf_(\d+)d_h(\d+)_([\d.]+)\.parquet', filename)
        if match:
            return {
                'days': int(match.group(1)),
                'horizon': int(match.group(2)),
                'threshold': float(match.group(3)),
                'threshold_str': match.group(3)
            }
    return None

def should_process_file(params, mode):
    """파라미터 기반으로 파일 처리 여부 결정"""
    # Days 필터
    if DAYS_TO_TRAIN is not None and params['days'] not in DAYS_TO_TRAIN:
        return False
    
    # Horizon 필터
    if HORIZON_TO_TRAIN is not None and params['horizon'] not in HORIZON_TO_TRAIN:
        return False
    
    # Threshold 필터
    if THRESHOLD_TO_TRAIN is not None:
        # 부동소수점 비교를 위해 근사값 비교
        threshold_match = any(abs(params['threshold'] - t) < 0.0001 for t in THRESHOLD_TO_TRAIN)
        if not threshold_match:
            return False
    
    # SEQ 모드 전용 N_STEPS 필터
    if mode == "seq" and N_STEPS_TO_TRAIN is not None:
        if params['n_steps'] not in N_STEPS_TO_TRAIN:
            return False
    
    return True

def run_train_model(data_path, model_path, scaler_path, meta_path, splits, fee, allow_short=False):
    """단일 모델 학습"""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "train.py"),
        "--data", data_path,
        "--model", model_path,
        "--scaler", scaler_path,
        "--meta", meta_path,
        "--splits", str(splits),
        "--fee", str(fee)
    ]
    
    if allow_short:
        cmd.append("--allow_short")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT_DIR))  # 10분 타임아웃
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout (10분 초과)"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="일괄 모델 학습 도구")
    parser.add_argument("--mode", type=str, required=True, choices=["seq", "classic"],
                      help="학습 모드: seq 또는 classic")
    parser.add_argument("--data_dir", type=str, required=True,
                      help="데이터셋 디렉토리")
    parser.add_argument("--output_dir", type=str, default="artifacts/train",
                      help="출력 디렉토리")
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS,
                      help="교차 검증 분할 수")
    parser.add_argument("--fee", type=float, default=DEFAULT_FEE,
                      help="거래 수수료")
    parser.add_argument("--allow_short", action="store_true",
                      help="숏 포지션 허용")
    parser.add_argument("--force", action="store_true",
                      help="기존 모델이 있어도 재학습")
    
    args = parser.parse_args()
    
    
    # 출력 디렉터리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 데이터셋 파일 검색
    if args.mode == "seq":
        pattern = os.path.join(args.data_dir, "dataset_seq_*.parquet")
    else:
        pattern = os.path.join(args.data_dir, "dataset_mtf_*.parquet")
    
    all_files = glob.glob(pattern)
    # 필터링
    dataset_files = []
    for file in all_files:
        filename = os.path.basename(file)
        params = extract_params_from_filename(filename, args.mode)
        if params and should_process_file(params, args.mode):
            dataset_files.append(file)
    
    print(f"{args.mode.upper()} 모드: {len(dataset_files)}개 파일 학습 시작")
    
    if not dataset_files:
        print("[ERROR] 필터 조건에 맞는 데이터셋 파일이 없습니다.")
        return
    
    
    for idx, data_file in enumerate(sorted(dataset_files), 1):
        filename = os.path.basename(data_file)
        params = extract_params_from_filename(filename, args.mode)
        
        if not params:
            print(f"[{idx}/{len(dataset_files)}] {filename} - 파일명 파싱 실패")
            continue
        
        # 출력 파일명 생성 (one_click_pipeline.py 형식)
        if args.mode == "seq":
            base_name = f"seq_{params['days']}d_h{params['horizon']}_n{params['n_steps']}_{params['threshold_str']}"
            display_name = f"d{params['days']}_h{params['horizon']}_n{params['n_steps']}_t{params['threshold_str']}"
        else:
            base_name = f"classic_{params['days']}d_h{params['horizon']}_{params['threshold_str']}"
            display_name = f"d{params['days']}_h{params['horizon']}_t{params['threshold_str']}"
        
        model_file = os.path.join(args.output_dir, f"model_{base_name}.joblib")
        scaler_file = os.path.join(args.output_dir, f"scaler_{base_name}.joblib")
        meta_file = os.path.join(args.output_dir, f"meta_{base_name}.json")
        
        # 이미 모델이 존재하면 건너뛰기 (--force 옵션이 없을 때만)
        if not args.force and os.path.exists(model_file) and os.path.exists(meta_file):
            print(f"[{idx:3d}/{len(dataset_files)}] {display_name} 처리 중... [SKIP] (이미 존재)")
            continue
        
        print(f"[{idx:3d}/{len(dataset_files)}] {display_name} 처리 중...", end=" ")
        
        # 모델 학습
        success, message = run_train_model(data_file, model_file, scaler_file, meta_file, args.splits, args.fee, args.allow_short)
        
        if success:
            print(f"[OK]")
        else:
            print(f"[FAIL]")
    
    print(f"\n배치 완료")

if __name__ == "__main__":
    main()