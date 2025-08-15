#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2025CoinTrader V2 - 원클릭 파이프라인 최종
- 수집 → 데이터셋 → 학습 → 백테스트를 한 번에 수행
- 실시간 로그 스트리밍(Popen + -u + PYTHONUNBUFFERED=1)
- 모든 스크립트 경로는 절대경로로 호출(중첩/상대경로 이슈 방지)
- build_dataset.py 인자: --in / --up / --dn 사용
- 기본 전제: 1분봉 수집, PnL 규칙은 다음 봉 시가 체결 + 수수료
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# === 경로 설정 =================================================================
TOOLS_DIR = Path(__file__).resolve().parent           # .../V2_Based_with_CHATGPT/tools
ROOT_DIR = TOOLS_DIR.parent                            # .../V2_Based_with_CHATGPT
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# === 기본 상수 =================================================================
DEFAULT_MARKET = "KRW-BTC"
DEFAULT_MINUTES = 1
DEFAULT_DAYS = 180
DEFAULT_MODE = "classic"      # classic | seq
DEFAULT_HORIZON = 1           # 분
DEFAULT_THR_PCT = 0.1         # %
DEFAULT_FEE = 0.0005          # 업비트 0.05%
DEFAULT_ALLOW_SHORT = True
DEFAULT_N_STEPS = 20           # 시퀀스 길이 (seq 모드용)

# === 유틸 ======================================================================
def ensure_dirs():
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "classic").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "seq").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "train").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "backtest").mkdir(parents=True, exist_ok=True)

def yn_to_bool(s: str | None, default=True) -> bool:
    if not s or not s.strip():
        return default
    return s.strip().lower() in ("y", "yes", "1", "true", "t")

def pct_to_float(pct_str: str | None, default_pct: float) -> float:
    if not pct_str or not pct_str.strip():
        return default_pct / 100.0
    try:
        return float(pct_str.strip()) / 100.0
    except Exception:
        return default_pct / 100.0

def run_streaming(cmd, step_name, timeout=None, extra_env=None) -> bool:
    """실시간 로그 스트리밍 실행"""
    print(f"\n[실행중] {step_name} ...")
    print("명령어:", " ".join(str(c) for c in cmd))
    print("작업 디렉토리:", ROOT_DIR)
    print("-" * 50)

    patched_cmd = cmd[:]
    if Path(patched_cmd[0]).name.lower().startswith("python"):
        patched_cmd.insert(1, "-u")  # 언버퍼드

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    with subprocess.Popen(
        patched_cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    ) as p:
        try:
            for line in p.stdout:
                print(line, end="")
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            print(f"\n[실패] {step_name} - 타임아웃({timeout}s)")
            return False
        if p.returncode != 0:
            print(f"\n[실패] {step_name} - 종료코드 {p.returncode}")
            return False

    print(f"[완료] {step_name}")
    return True

def summarize_report(report_path: Path):
    if not report_path.exists():
        print(f"[경고] 백테스트 리포트 미존재: {report_path}")
        return
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            rep = json.load(f)
    except Exception as e:
        print(f"[경고] 리포트 읽기 실패: {e}")
        return

    final_equity = rep.get("final_equity")
    hold_final = rep.get("hold_final")
    n_trades = rep.get("n_trades", rep.get("num_trades"))
    win_rate = rep.get("win_rate")
    mdd = rep.get("mdd")
    sharpe = rep.get("sharpe")
    calmar = rep.get("calmar")

    cum_ret = rep.get("cumulative_return", 0.0) if final_equity is None else (final_equity - 1.0)

    print("\n================ 백테스트 요약 ================")
    print(f"최종 수익률: {cum_ret*100:.2f}%")
    if final_equity is not None:
        print(f"최종 에쿼티배수: {final_equity:.4f}배")
    if hold_final is not None:
        print(f"단순 보유 대비: {hold_final:.4f}배")
    if n_trades is not None:
        print(f"거래 수: {n_trades}")
    if win_rate is not None:
        print(f"승률: {win_rate*100:.2f}%")
    if mdd is not None:
        print(f"MDD: {mdd*100:.2f}%")
    if sharpe is not None:
        print(f"Sharpe: {sharpe:.3f}")
    if calmar is not None:
        print(f"Calmar: {calmar:.3f}")
    print("===============================================")

# === 경로 생성 =================================================================
def make_paths(days: int, mode: str, horizon: int, thr: float, allow_short: bool, n_steps: int = None):
    thr_token = f"{thr:.3f}".rstrip("0").rstrip(".")  # 0.001 -> "0.001"
    # short_token 제거 - 숏 허용 여부와 관계없이 동일한 파일명 사용

    raw_path = DATA_DIR / "raw" / f"krw_btc_1m_{days}d.parquet"

    if mode == "classic":
        ds_name = f"dataset_mtf_{days}d_h{horizon}_{thr_token}.parquet"
        ds_path = DATA_DIR / "classic" / ds_name
        model_name = f"model_classic_{days}d_h{horizon}_{thr_token}.joblib"
        scaler_name = f"scaler_classic_{days}d_h{horizon}_{thr_token}.joblib"
        meta_name = f"meta_classic_{days}d_h{horizon}_{thr_token}.json"
        bt_json = f"bt_classic_{days}d_h{horizon}_{thr_token}.json"
        bt_csv  = f"trades_classic_{days}d_h{horizon}_{thr_token}.csv"
        bt_md   = f"trades_classic_{days}d_h{horizon}_{thr_token}.md"
    else:
        # seq 모드에서는 n_steps를 파일명에 포함
        steps_token = f"_s{n_steps}" if n_steps else ""
        ds_name = f"dataset_seq_{days}d_h{horizon}_n{n_steps}_{thr_token}.parquet"
        ds_path = DATA_DIR / "seq" / ds_name
        model_name = f"model_seq_{days}d_h{horizon}_n{n_steps}_{thr_token}.joblib"
        scaler_name = f"scaler_seq_{days}d_h{horizon}_n{n_steps}_{thr_token}.joblib"
        meta_name = f"meta_seq_{days}d_h{horizon}_n{n_steps}_{thr_token}.json"
        bt_json = f"bt_seq_{days}d_h{horizon}_n{n_steps}_{thr_token}.json"
        bt_csv  = f"trades_seq_{days}d_h{horizon}_n{n_steps}_{thr_token}.csv"
        bt_md   = f"trades_seq_{days}d_h{horizon}_n{n_steps}_{thr_token}.md"

    return {
        "raw": raw_path,
        "dataset": ds_path,
        "model": ARTIFACTS_DIR / "train" / model_name,
        "scaler": ARTIFACTS_DIR / "train" / scaler_name,
        "meta": ARTIFACTS_DIR / "train" / meta_name,
        "bt_report": ARTIFACTS_DIR / "backtest" / bt_json,
        "bt_trades": ARTIFACTS_DIR / "backtest" / bt_csv,
        "bt_md": ARTIFACTS_DIR / "backtest" / bt_md,
    }

# === 인터랙티브 입력 ============================================================
def interactive_inputs():
    print("=" * 60)
    print("2025CoinTrader V2 - 원클릭 파이프라인")
    print("=" * 60)
    print()
    print("실행 위치:", os.getcwd())
    print("프로젝트 루트:", ROOT_DIR)
    print()
    print("파이프라인 설정을 시작합니다.")
    print("Enter를 누르면 기본값이 사용됩니다.\n")

    days_txt = input(f"1. 데이터 수집 기간 (일수) [기본값: {DEFAULT_DAYS}]: ").strip()
    days = int(days_txt) if days_txt else DEFAULT_DAYS

    print("2. 데이터셋 모드 선택")
    print("   1: classic (단순 가격 예측)")
    print("   2: seq (시퀀스 데이터)")
    sel = input(f"   [기본값: 1]: ").strip()
    mode = DEFAULT_MODE if (not sel or sel == "1") else "seq"
    
    # seq 모드인 경우 시퀀스 길이 입력받기
    n_steps = None
    if mode == "seq":
        n_steps_txt = input(f"   2-1. 시퀀스 길이 (스텝 수) [기본값: {DEFAULT_N_STEPS}]: ").strip()
        n_steps = int(n_steps_txt) if n_steps_txt else DEFAULT_N_STEPS

    horizon_txt = input(f"3. 예측 시점 (분 단위) [기본값: {DEFAULT_HORIZON}]: ").strip()
    horizon = int(horizon_txt) if horizon_txt else DEFAULT_HORIZON

    thr_txt = input(f"4. 가격 변화 임계값 (%) [기본값: {DEFAULT_THR_PCT}]: ").strip()
    thr = pct_to_float(thr_txt, DEFAULT_THR_PCT)

    short_inp = input(f"7. 숏 포지션 허용 (y/n) [기본값: {'y' if DEFAULT_ALLOW_SHORT else 'n'}]: ").strip()
    allow_short = yn_to_bool(short_inp, default=DEFAULT_ALLOW_SHORT)

    print("\n============================================================")
    print("설정 요약")
    print("============================================================")
    print(f"데이터 수집 기간: {days}일")
    print(f"데이터셋 모드: {mode}")
    if mode == "seq" and n_steps:
        print(f"시퀀스 길이: {n_steps} 스텝")
    print(f"예측 시점: {horizon}분")
    print(f"가격 변화 임계값: ±{thr*100:.3f}%")
    print(f"숏 포지션: {'허용' if allow_short else '미허용'}")
    print(f"거래 수수료: {DEFAULT_FEE*100:.2f}% (업비트 고정)")

    paths = make_paths(days, mode, horizon, thr, allow_short, n_steps)

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT_DIR))
        except Exception:
            return str(p)

    print("\n생성될 파일들:")
    print("  raw_data:", rel(paths["raw"]))
    print("  dataset:", rel(paths["dataset"]))
    print("  model:", rel(paths["model"]))
    print("  scaler:", rel(paths["scaler"]))
    print("  meta:", rel(paths["meta"]))
    print("  bt_report:", rel(paths["bt_report"]))
    print("  bt_trades:", rel(paths["bt_trades"]))
    print("  bt_markdown:", rel(paths["bt_md"]))

    cont = input("\n계속 진행하시겠습니까? (y/n) [기본값: y]: ").strip()
    if cont and cont.lower() not in ("y", "yes"):
        print("사용자에 의해 취소되었습니다.")
        sys.exit(0)

    return {
        "days": days,
        "mode": mode,
        "n_steps": n_steps,
        "horizon": horizon,
        "thr": thr,
        "allow_short": allow_short,
        "paths": paths,
    }

# === 커맨드 빌드(절대경로) ======================================================
def build_commands(cfg):
    py = sys.executable
    days = cfg["days"]
    mode = cfg["mode"]
    n_steps = cfg.get("n_steps")
    horizon = cfg["horizon"]
    thr = cfg["thr"]
    allow_short = cfg["allow_short"]
    paths = cfg["paths"]

    cmd_collect = [
        py, "tools/collect.py",  # 상대 경로 사용
        "--market", DEFAULT_MARKET,
        "--minutes", str(DEFAULT_MINUTES),
        "--days", str(days),
        "--out", str(paths["raw"].relative_to(ROOT_DIR)),  # 상대 경로로 변환
    ]

    if mode == "classic":
        cmd_build = [
            py, "tools/build_dataset.py",  # 상대 경로 사용
            "--mode", "classic",
            "--in", str(paths["raw"].relative_to(ROOT_DIR)),          # 상대 경로로 변환
            "--out", str(paths["dataset"].relative_to(ROOT_DIR)),     # 상대 경로로 변환
            "--horizon", str(horizon),
            "--up", str(thr),                   # <-- 중요: --thr 대신 --up/--dn
            "--dn", str(-thr),                  # <-- 음수로 변경
            # "--warmup_minutes", "420",        # 필요 시 사용
        ]
    else:
        cmd_build = [
            py, "tools/build_dataset.py",  # 상대 경로 사용
            "--mode", "seq",
            "--in", str(paths["raw"].relative_to(ROOT_DIR)),          # 상대 경로로 변환
            "--out", str(paths["dataset"].relative_to(ROOT_DIR)),     # 상대 경로로 변환
            "--horizon", str(horizon),
            "--up", str(thr),
            "--dn", str(-thr),                  # <-- 음수로 변경
            # "--tfs", "1,3,5,15,60",
        ]
        if n_steps:
            cmd_build.extend(["--n_steps", str(n_steps)])
        # 선택적 파라미터들
        # cmd_build.extend(["--ta"])  # 기술지표 사용시
        # cmd_build.extend(["--warmup_minutes", "420"])  # 워밍업 필요시

    cmd_train = [
        py, "tools/train.py",  # 상대 경로 사용
        "--data", str(paths["dataset"].relative_to(ROOT_DIR)),
        "--model", str(paths["model"].relative_to(ROOT_DIR)),
        "--scaler", str(paths["scaler"].relative_to(ROOT_DIR)),
        "--meta", str(paths["meta"].relative_to(ROOT_DIR)),
        "--fee", str(DEFAULT_FEE),
    ]
    if allow_short:
        cmd_train.append("--allow_short")

    cmd_back = [
        py, "tools/backtest.py",  # 상대 경로 사용
        "--data", str(paths["dataset"].relative_to(ROOT_DIR)),
        "--model", str(paths["model"].relative_to(ROOT_DIR)),
        "--meta", str(paths["meta"].relative_to(ROOT_DIR)),
        "--scaler", str(paths["scaler"].relative_to(ROOT_DIR)),
        "--fee", str(DEFAULT_FEE),
        "--report", str(paths["bt_report"].relative_to(ROOT_DIR)),
        "--log_csv", str(paths["bt_trades"].relative_to(ROOT_DIR)),
        "--auto_md",
        "--md_out", str(paths["bt_md"].relative_to(ROOT_DIR)),
    ]

    return cmd_collect, cmd_build, cmd_train, cmd_back

# === 메인 ======================================================================
def main():
    cfg = interactive_inputs()
    ensure_dirs()

    cmd_collect, cmd_build, cmd_train, cmd_back = build_commands(cfg)

    print("\n============================================================")
    print("파이프라인 실행 시작")
    print("============================================================")

    if not run_streaming(cmd_collect, "1단계: 데이터 수집", timeout=None):
        sys.exit(1)

    if not run_streaming(cmd_build, "2단계: 데이터셋 생성", timeout=None):
        sys.exit(1)

    if not run_streaming(cmd_train, "3단계: 모델 학습", timeout=None):
        sys.exit(1)

    if not run_streaming(cmd_back, "4단계: 백테스트", timeout=None):
        sys.exit(1)

    summarize_report(cfg["paths"]["bt_report"])

    print("\n모든 단계 완료 ✅")
    print("산출물 위치:")
    for k in ("raw", "dataset", "model", "scaler", "meta", "bt_report", "bt_trades", "bt_md"):
        p = cfg["paths"][k]
        try:
            rel = str(p.relative_to(ROOT_DIR))
        except Exception:
            rel = str(p)
        print(f" - {k:10s}: {rel}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n사용자 중단(Ctrl+C). 종료합니다.")
        sys.exit(130)
