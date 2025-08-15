"""
batch_data.py - 다중 파라미터 조합 일괄 데이터셋 생성 도구

CLI 사용 예시:

# SEQ 모드 (패턴 기반)
python tools/batch_data.py `
  --mode seq `
  --input data/raw/krw_btc_1m_181d.parquet `
  --output_dir data/seq

# CLASSIC 모드 (기술적 지표 기반)  
python tools/batch_data.py `
  --mode classic `
  --input data/raw/krw_btc_1m_181d.parquet `
  --output_dir data/classic

사용법:
• 공통 파라미터:
  - HORIZON: [1, 3, 5, 10, 15] - 예측 시점 (분)
  - THRESHOLD: [0.0003, 0.0005, 0.001, 0.002] - 라벨링 임계값 (±값)

• SEQ 모드: N_STEPS x HORIZON x THRESHOLD 조합으로 데이터셋 생성
  - N_STEPS: [5, 10, 15, 20, 30, 60] - 입력 패턴 길이
  
• CLASSIC 모드: HORIZON x TIMEFRAMES x THRESHOLD 조합으로 데이터셋 생성
  - TIMEFRAMES: ["", "3", "3,5", "3,5,15"] - 멀티 타임프레임

출력:
• 데이터셋 파일들
• 생성 결과 요약 리포트
• 실패한 조합 목록
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

# 현재 스크립트의 절대 경로 기준으로 프로젝트 루트 찾기
SCRIPT_DIR = Path(__file__).resolve().parent  # tools 폴더
ROOT_DIR = SCRIPT_DIR.parent  # V2_Based_with_CHATGPT 폴더

# ========== 배치 파라미터 설정 (사용자 수정 가능) ==========

# 공통 파라미터
HORIZON_CANDIDATES = [1, 3, 5, 10, 15, 20]  # 예측 시점 (분)
THRESHOLD_CANDIDATES = [0.001, 0.002, 0.003]  # 라벨링 임계값 (±0.03%, ±0.05%, ±0.1%, ±0.2%)

# SEQ 모드 전용 파라미터
N_STEPS_CANDIDATES = [5, 10, 15, 20, 30, 60]  # 입력 패턴 길이

# CLASSIC 모드 전용 파라미터  
TIMEFRAMES_CANDIDATES = ["3,5"]  # 멀티 타임프레임 ("" = 1분봉만)

# ========================================================

# 기본 설정 - 입력 파일은 필수로 지정해야 함 (일수는 파일명에서 자동 추출)

def run_build_dataset_seq(input_file, n_steps, horizon, threshold, output_path):
    """SEQ 모드 데이터셋 생성"""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "build_dataset.py"),
        "--mode", "seq",
        "--in", input_file,
        "--out", output_path,
        "--n_steps", str(n_steps),
        "--horizon", str(horizon),
        "--up", str(threshold),
        "--dn", str(-threshold),  # DN은 UP의 음수값
        "--ta"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT_DIR))
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout (5분 초과)"
    except Exception as e:
        return False, str(e)

def run_build_dataset_classic(input_file, horizon, timeframes, threshold, output_path):
    """CLASSIC 모드 데이터셋 생성"""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "build_dataset.py"),
        "--mode", "classic",
        "--in", input_file,
        "--out", output_path,
        "--horizon", str(horizon),
        "--up", str(threshold),
        "--dn", str(-threshold)  # DN은 UP의 음수값
    ]
    
    # 타임프레임이 있으면 추가
    if timeframes:
        cmd.extend(["--tfs", timeframes])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT_DIR))
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout (5분 초과)"
    except Exception as e:
        return False, str(e)


def extract_days_from_filename(input_file):
    """입력 파일명에서 일수 추출 (예: krw_btc_1m_180d.parquet -> 180)"""
    import re
    match = re.search(r'_(\d+)d\.parquet', input_file)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"파일명에서 일수를 추출할 수 없습니다: {input_file}\n"
                        f"올바른 형식: krw_btc_1m_XXXd.parquet (예: krw_btc_1m_180d.parquet)")

def run_seq_batch(input_file, output_dir):
    """SEQ 모드 배치 실행"""
    # 입력 파일명에서 일수 추출
    days = extract_days_from_filename(input_file)
    
    total_combinations = len(N_STEPS_CANDIDATES) * len(HORIZON_CANDIDATES) * len(THRESHOLD_CANDIDATES)
    print(f"SEQ 모드: {total_combinations}개 조합 처리 시작")
    
    current_combination = 0
    
    for n_steps in N_STEPS_CANDIDATES:
        for horizon in HORIZON_CANDIDATES:
            for threshold in THRESHOLD_CANDIDATES:
                current_combination += 1
                
                # 출력 파일명 (one_click_pipeline.py 규칙)
                threshold_str = f"{threshold:.3f}".rstrip("0").rstrip(".")
                output_file = f"{output_dir}/dataset_seq_{days}d_h{horizon}_n{n_steps}_{threshold_str}.parquet"
                
                print(f"[{current_combination}/{total_combinations}] 처리 중...", end=" ")
                
                # 데이터셋 생성
                success, message = run_build_dataset_seq(input_file, n_steps, horizon, threshold, output_file)
                
                if success:
                    print(f"    [OK]")
                else:
                    print(f"    [FAIL]")

def run_classic_batch(input_file, output_dir):
    """CLASSIC 모드 배치 실행"""
    # 입력 파일명에서 일수 추출
    days = extract_days_from_filename(input_file)
    
    total_combinations = len(HORIZON_CANDIDATES) * len(TIMEFRAMES_CANDIDATES) * len(THRESHOLD_CANDIDATES)
    print(f"CLASSIC 모드: {total_combinations}개 조합 처리 시작")
    
    current_combination = 0
    
    for horizon in HORIZON_CANDIDATES:
        for timeframes in TIMEFRAMES_CANDIDATES:
            for threshold in THRESHOLD_CANDIDATES:
                current_combination += 1
                
                # 출력 파일명 (one_click_pipeline.py 규칙)
                threshold_str = f"{threshold:.3f}".rstrip("0").rstrip(".")
                # classic 모드는 mtf 사용
                output_file = f"{output_dir}/dataset_mtf_{days}d_h{horizon}_{threshold_str}.parquet"
                
                tf_display = f"TF={timeframes if timeframes else '1m':6s}"
                print(f"[{current_combination}/{total_combinations}] 처리 중...", end=" ")
                
                # 데이터셋 생성
                success, message = run_build_dataset_classic(input_file, horizon, timeframes, threshold, output_file)
                
                if success:
                    print(f"    [OK]")
                else:
                    print(f"    [FAIL]")


def main():
    parser = argparse.ArgumentParser(description="배치 데이터셋 생성 도구")
    parser.add_argument("--mode", type=str, required=True, choices=["seq", "classic"],
                      help="데이터셋 모드: seq(시퀀스) 또는 classic(기술지표)")
    parser.add_argument("--input", type=str, required=True,
                      help="입력 파일 경로 (필수, 형식: krw_btc_1m_XXXd.parquet)")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="출력 디렉토리 (기본: data/seq 또는 data/classic)")
    args = parser.parse_args()
    
    # 출력 디렉토리 자동 설정
    if args.output_dir is None:
        args.output_dir = f"data/{args.mode}"
    
    print(f"배치 데이터셋 생성 시작 - {args.mode.upper()} 모드")
    print(f"실행 위치: {Path.cwd()}")
    print(f"프로젝트 루트: {ROOT_DIR}")
    print(f"입력 파일: {args.input}")
    print(f"출력 디렉토리: {args.output_dir}")
    print()
    
    # 입력 파일 존재 확인
    if not os.path.exists(args.input):
        print(f"[ERROR] 입력 파일이 없습니다: {args.input}")
        print("먼저 다음 명령으로 데이터를 수집하세요:")
        print(f"python tools/collect.py --market KRW-BTC --minutes 1 --days 5 --out {args.input}")
        return
    
    # 출력 디렉터리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모드별 배치 실행
    if args.mode == "seq":
        run_seq_batch(args.input, args.output_dir)
    else:  # classic
        run_classic_batch(args.input, args.output_dir)
    
    print(f"\n배치 완료")

if __name__ == "__main__":
    main()