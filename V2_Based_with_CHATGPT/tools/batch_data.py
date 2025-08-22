"""
batch_data.py - 다중 파라미터 조합 일괄 데이터셋 생성 도구

CLI 사용 예시:

# SEQ 모드 - 자동으로 data/raw에서 파일 선택
python tools/batch_data.py --mode seq

# CLASSIC 모드 - 자동으로 data/raw에서 파일 선택  
python tools/batch_data.py --mode classic

# 특정 입력 파일 지정 (선택사항)
python tools/batch_data.py `
  --mode seq `
  --input data/raw/krw_btc_1m_30d.parquet

# CSV 파일도 함께 생성 (모든 조합에 대해)
python tools/batch_data.py --mode seq --save_csv

사용법:
• 공통 파라미터:
  - DAYS_TO_PROCESS: 처리할 일수 목록 (상단 설정에서 수정)
  - HORIZON: [1, 3, 5, 10, 15] - 예측 시점 (분)
  - THRESHOLD: [0.001, 0.002, 0.003] - 라벨링 임계값 (±값)
  - --save_csv: CSV 파일도 함께 저장 (선택사항)

• SEQ 모드: N_STEPS x HORIZON x THRESHOLD 조합으로 데이터셋 생성
  - N_STEPS: [5, 10, 15, 20, 30, 60] - 입력 패턴 길이
  
• CLASSIC 모드: HORIZON x TIMEFRAMES x THRESHOLD 조합으로 데이터셋 생성
  - TIMEFRAMES: ["3,5"] - 멀티 타임프레임

출력:
• 데이터셋 파일들 (자동 명명)
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

# 처리할 일수 목록 (None = data/raw에서 모든 파일, 리스트 = 선택된 일수만)
DAYS_TO_PROCESS = [200]  # 예: [181, 200] 또는 None (모두)

# 공통 파라미터
HORIZON_CANDIDATES = [1, 2, 3, 5, 7, 10]  # 예측 시점 (분)
THRESHOLD_CANDIDATES = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001]  # 라벨링 임계값

# SEQ 모드 전용 파라미터
N_STEPS_CANDIDATES = [5, 10, 15, 20, 30, 60]  # 입력 패턴 길이

# CLASSIC 모드 전용 파라미터  
TIMEFRAMES_CANDIDATES = ["3,5"]  # 멀티 타임프레임 ("" = 1분봉만)

# ========================================================

def run_build_dataset_seq(input_file, n_steps, horizon, threshold, output_path, save_csv=False):
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
    
    # CSV 저장 옵션 추가
    if save_csv:
        cmd.append("--save_csv")
    
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

def run_build_dataset_classic(input_file, horizon, timeframes, threshold, output_path, save_csv=False):
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
    
    # CSV 저장 옵션 추가
    if save_csv:
        cmd.append("--save_csv")
    
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

def find_input_files(days_list=None):
    """data/raw 디렉토리에서 입력 파일 자동 검색
    
    Args:
        days_list: 처리할 일수 목록 (None = 모든 파일)
        
    Returns:
        찾은 파일 경로 리스트
    """
    import glob
    import re
    
    raw_dir = ROOT_DIR / "data" / "raw"
    pattern = str(raw_dir / "krw_btc_1m_*d.parquet")
    all_files = glob.glob(pattern)
    
    if not all_files:
        raise FileNotFoundError(f"data/raw 디렉토리에 입력 파일이 없습니다.\n"
                               f"먼저 데이터를 수집하세요:\n"
                               f"python tools/collect.py --market KRW-BTC --minutes 1 --days 200")
    
    if days_list is None:
        # 모든 파일 반환
        return sorted(all_files)
    
    # 특정 일수만 필터링
    filtered_files = []
    for file_path in all_files:
        match = re.search(r'_(\d+)d\.parquet', file_path)
        if match and int(match.group(1)) in days_list:
            filtered_files.append(file_path)
    
    if not filtered_files:
        available_days = []
        for file_path in all_files:
            match = re.search(r'_(\d+)d\.parquet', file_path)
            if match:
                available_days.append(int(match.group(1)))
        raise FileNotFoundError(f"요청한 일수 {days_list}에 해당하는 파일이 없습니다.\n"
                               f"사용 가능한 일수: {sorted(available_days)}")
    
    return sorted(filtered_files)

def run_seq_batch(input_file, output_dir, save_csv=False):
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
                # 더 정확한 threshold 문자열 생성
                if threshold < 0.0001:
                    threshold_str = f"{threshold:.6f}".rstrip("0").rstrip(".")
                elif threshold < 0.001:
                    threshold_str = f"{threshold:.5f}".rstrip("0").rstrip(".")
                else:
                    threshold_str = f"{threshold:.3f}".rstrip("0").rstrip(".")
                output_file = f"{output_dir}/dataset_seq_{days}d_h{horizon}_n{n_steps}_{threshold_str}.parquet"
                
                # 파라미터 정보를 간단히 표시
                param_info = f"n{n_steps}_h{horizon}_t{threshold_str}"
                print(f"[{current_combination:3d}/{total_combinations}] {param_info} 처리 중...", end=" ")
                
                # 데이터셋 생성
                success, message = run_build_dataset_seq(input_file, n_steps, horizon, threshold, output_file, save_csv)
                
                if success:
                    print(f"[OK]")
                else:
                    print(f"[FAIL]")

def run_classic_batch(input_file, output_dir, save_csv=False):
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
                # 더 정확한 threshold 문자열 생성
                if threshold < 0.0001:
                    threshold_str = f"{threshold:.6f}".rstrip("0").rstrip(".")
                elif threshold < 0.001:
                    threshold_str = f"{threshold:.5f}".rstrip("0").rstrip(".")
                else:
                    threshold_str = f"{threshold:.3f}".rstrip("0").rstrip(".")
                # classic 모드는 mtf 사용
                output_file = f"{output_dir}/dataset_mtf_{days}d_h{horizon}_{threshold_str}.parquet"
                
                # 파라미터 정보를 간단히 표시
                tf_info = f"tf{timeframes}" if timeframes else "tf1m"
                param_info = f"h{horizon}_{tf_info}_t{threshold_str}"
                print(f"[{current_combination:3d}/{total_combinations}] {param_info} 처리 중...", end=" ")
                
                # 데이터셋 생성
                success, message = run_build_dataset_classic(input_file, horizon, timeframes, threshold, output_file, save_csv)
                
                if success:
                    print(f"[OK]")
                else:
                    print(f"[FAIL]")


def main():
    parser = argparse.ArgumentParser(description="배치 데이터셋 생성 도구")
    parser.add_argument("--mode", type=str, required=True, choices=["seq", "classic"],
                      help="데이터셋 모드: seq(시퀀스) 또는 classic(기술지표)")
    parser.add_argument("--input", type=str, default=None,
                      help="입력 파일 경로 (선택, 미지정시 DAYS_TO_PROCESS 기준 자동 선택)")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="출력 디렉토리 (기본: data/seq 또는 data/classic)")
    parser.add_argument("--save_csv", action="store_true",
                      help="결과를 CSV 파일로도 저장 (디버깅/분석용)")
    args = parser.parse_args()
    
    # 출력 디렉토리 자동 설정
    if args.output_dir is None:
        args.output_dir = f"data/{args.mode}"
    
    print(f"배치 데이터셋 생성 시작 - {args.mode.upper()} 모드")
    print(f"실행 위치: {Path.cwd()}")
    print(f"프로젝트 루트: {ROOT_DIR}")
    print(f"출력 디렉토리: {args.output_dir}")
    
    # 입력 파일 결정
    if args.input:
        # 명시적으로 지정된 파일 사용
        input_files = [args.input]
        print(f"입력 파일: {args.input}")
    else:
        # DAYS_TO_PROCESS 기준으로 자동 선택
        try:
            input_files = find_input_files(DAYS_TO_PROCESS)
            print(f"처리할 일수: {DAYS_TO_PROCESS if DAYS_TO_PROCESS else '모든 파일'}")
            print(f"자동 검색된 파일: {len(input_files)}개")
            for f in input_files:
                print(f"  - {os.path.basename(f)}")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return
    
    print()
    
    # 출력 디렉터리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모든 입력 파일에 대해 배치 실행
    for input_file in input_files:
        if len(input_files) > 1:
            print(f"\n{'='*60}")
            print(f"처리 중: {os.path.basename(input_file)}")
            print(f"{'='*60}\n")
        
        # 입력 파일 존재 확인
        if not os.path.exists(input_file):
            print(f"[ERROR] 입력 파일이 없습니다: {input_file}")
            continue
        
        # 모드별 배치 실행
        if args.mode == "seq":
            run_seq_batch(input_file, args.output_dir, args.save_csv)
        else:  # classic
            run_classic_batch(input_file, args.output_dir, args.save_csv)
    
    print(f"\n모든 배치 작업 완료")

if __name__ == "__main__":
    main()