# tools/sweep.py
import argparse, itertools, json, os, subprocess, sys, time
from pathlib import Path

def run(cmd):
    print("\n=== RUN:", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return p.stdout

def fmt_time(seconds):
    """초를 분:초로 변환"""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_1m", required=True, help="collect.py가 만든 1분봉 parquet (예: data/krw_btc_1m.parquet)")
    ap.add_argument("--outdir", default="artifacts/sweep", help="중간 산출물/리포트 저장 폴더")
    ap.add_argument("--n_steps", default="10,15,20,30,45,60", help="윈도우 길이 후보(쉼표구분)")
    ap.add_argument("--thresholds", default="0.0003,0.0005,0.0010", help="라벨 임계 후보(절대값)")
    ap.add_argument("--allow_short", default="0,1", help="숏 허용 여부 후보(0/1)")
    ap.add_argument("--fee", type=float, default=0.0005, help="백테스트 수수료(한쪽)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_steps_list = [int(x) for x in args.n_steps.split(",") if x.strip()]
    th_list = [float(x) for x in args.thresholds.split(",") if x.strip()]
    allow_short_list = [int(x) for x in args.allow_short.split(",") if x.strip()]

    results = []
    total = len(n_steps_list) * len(th_list) * len(allow_short_list)
    k = 0
    t0 = time.time()
    times = []  # 각 루프 소요 시간 기록

    for n_steps, th, as_int in itertools.product(n_steps_list, th_list, allow_short_list):
        k += 1
        tag = f"n{n_steps}_p{int(th*1e4):04d}_{'short' if as_int else 'long'}"

        # 진행률 & 예상 시간 표시
        if times:
            avg_time = sum(times) / len(times)
            elapsed = time.time() - t0
            remaining = avg_time * (total - (k - 1))
            print(f"\n##### [{k}/{total}] SWEEP {tag} | {k/total*100:.1f}% 완료 "
                  f"| 경과 {fmt_time(elapsed)} | 예상 남은 {fmt_time(remaining)} #####")
        else:
            print(f"\n##### [{k}/{total}] SWEEP {tag} | {k/total*100:.1f}% 완료 #####")

        t_start = time.time()

        # paths
        ds_path   = outdir / f"dataset_seq_{tag}.parquet"
        model     = outdir / f"model_{tag}.joblib"
        scaler    = outdir / f"scaler_{tag}.joblib"
        meta      = outdir / f"meta_{tag}.json"
        rpt_json  = outdir / f"bt_{tag}.json"
        tradescsv = outdir / f"trades_{tag}.csv"

        # 1) BUILD (seq)
        build_cmd = [
            sys.executable, "tools/build_dataset.py",
            "--mode", "seq",
            "--in", args.data_1m,
            "--out", str(ds_path),
            "--n_steps", str(n_steps),
            "--horizon", "1",
            "--up", str(th),
            "--dn", str(-th),
            "--ta",
        ]
        run(build_cmd)

        # 2) TRAIN
        train_cmd = [
            sys.executable, "tools/train.py",
            "--data", str(ds_path),
            "--model", str(model),
            "--scaler", str(scaler),
            "--meta", str(meta),
            "--splits", "4",
        ]
        if as_int:
            train_cmd.append("--allow_short")
        run(train_cmd)

        # 3) BACKTEST
        bt_cmd = [
            sys.executable, "tools/backtest.py",
            "--data", str(ds_path),
            "--model", str(model),
            "--meta", str(meta),
            "--scaler", str(scaler),
            "--fee", str(args.fee),
            "--report", str(rpt_json),
            "--log_csv", str(tradescsv),
            "--print_trades", "0",
        ]
        run(bt_cmd)

        # 요약 수집
        try:
            rpt = json.load(open(rpt_json, "r", encoding="utf-8"))
            final_equity = rpt.get("final_equity", 1.0)
            n_trades     = rpt.get("n_trades", 0)
            hold_final   = rpt.get("hold_final", 1.0)
        except Exception as e:
            final_equity, n_trades, hold_final = 1.0, 0, 1.0
            print(f"[warn] report parse failed for {tag}: {e}")

        results.append({
            "tag": tag,
            "n_steps": n_steps,
            "threshold": th,
            "allow_short": as_int,
            "final_equity": final_equity,
            "n_trades": n_trades,
            "hold_final": hold_final,
            "dataset": str(ds_path),
            "model": str(model),
            "meta": str(meta),
            "report": str(rpt_json),
            "trades_csv": str(tradescsv),
        })

        # 이번 루프 소요 시간 기록
        times.append(time.time() - t_start)

    # 정렬: 최종 자본배수 내림차순
    results.sort(key=lambda x: x["final_equity"], reverse=True)

    # 저장
    summary_path = outdir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 콘솔 Top 10
    print("\n==== TOP 10 by final_equity ====")
    for i, r in enumerate(results[:10], start=1):
        print(f"[{i}] tag={r['tag']}  eq={r['final_equity']:.4f}  trades={r['n_trades']}  n_steps={r['n_steps']}  th=±{r['threshold']*100:.2f}%  short={bool(r['allow_short'])}")

    print(f"\nSaved summary -> {summary_path}")
    print(f"Elapsed: {fmt_time(time.time()-t0)}, Total runs: {total}")

if __name__ == "__main__":
    main()
