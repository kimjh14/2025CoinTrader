# tools/train.py
"""
train.py - XGBoost(GPU) 기반 머신러닝 모델 학습 도구

CLI 예시:
python tools/train.py `
--data data/seq/dataset_seq_n20_h20.parquet `
--model artifacts/model_seq_n20_h20.joblib `
--scaler artifacts/scaler_seq_n20_h20.joblib `
--meta artifacts/meta_seq_n20_h20.json `
--splits 4 `
--fee 0.0005

요약:
- 데이터셋(.parquet)을 읽어 XGBoost 다중분류(softprob)로 학습합니다.
- TimeSeriesSplit 교차검증으로 간단한 파라미터 탐색 후, 베스트 파라미터로 재학습합니다.
- 확률 보정(CalibratedClassifierCV, isotonic)과 진입 임계치 탐색(롱/숏)을 수행합니다.
- 가능하면 GPU(device='cuda') 사용, 실패 시 CPU(device='cpu')로 자동 폴백합니다.
"""

import argparse, os, json, time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

# XGBoost 불러오기 (미설치 시 메시지)
try:
    import xgboost as xgb
except Exception as e:
    raise RuntimeError(
        "xgboost가 설치되어 있지 않습니다. 다음 중 하나로 설치하세요.\n"
        " - pip:  pip install xgboost  (※ Windows + Py3.13의 pip 빌드는 보통 GPU 미포함)\n"
        " - conda: conda install -c conda-forge xgboost cudatoolkit\n"
        f"원인: {e}"
    )

# ------------------------------------------------------------
# 데이터/유틸
# ------------------------------------------------------------
def load_dataset(path: str):
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    drop_cols = {"label", "fwd_ret", "timestamp", "timestamp_kst"}
    feats = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].values
    y = df["label"].astype(int).values
    ts = pd.to_datetime(df["timestamp"]).values
    fwd = df["fwd_ret"].values
    return df, X, y, feats, ts, fwd

def profit_weights(fwd_ret, alpha=8.0, target=0.003, cap=3.0):
    # 큰 수익/손실 구간일수록 가중 ↑
    mag = np.minimum(np.abs(fwd_ret) / target, cap)
    return 1.0 + alpha * mag

def quick_equity_from_proba(
    p_long, p_short=None, ret=None, fee=0.0005, allow_short=False, long_p=0.6, short_p=0.6
):
    # 확률 임계치로 진입/청산 판단 → 자본배수 계산(수수료 반영)
    if allow_short and p_short is not None:
        pred = np.where(p_long >= long_p, 1, np.where(p_short >= short_p, -1, 0))
    else:
        pred = (p_long >= long_p).astype(int)  # 롱 온리
    pos = np.roll(pred, 1)
    pos[0] = 0
    r = np.nan_to_num(ret, nan=0.0)
    gross = (1 + r * pos).cumprod()
    trades = np.abs(np.diff(pos, prepend=0))
    equity = gross * np.cumprod(1 - fee * trades)
    return float(equity[-1])

def pick_xgb_params_from_grid(ps: dict, n_classes: int, use_gpu: bool):
    """XGBoost 2.x 권장 파라미터 (GPU는 device='cuda', CPU는 device='cpu')."""
    lr = float(ps.get("learning_rate", 0.05))
    max_depth = int(ps.get("max_depth", 8))
    reg_lambda = float(ps.get("l2_regularization", 0.0))
    min_child_weight = max(1.0, float(ps.get("min_samples_leaf", 50)) / 10.0)

    return dict(
        n_estimators=800,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=reg_lambda,
        reg_alpha=0.0,
        min_child_weight=min_child_weight,
        objective="multi:softprob",
        num_class=n_classes,
        random_state=42,
        verbosity=1,
        tree_method="hist",                        # ✅ 2.x 권장
        device=("cuda" if use_gpu else "cpu"),    # ✅ GPU/CPU 선택
        eval_metric="mlogloss",
    )

def has_gpu_preference():
    # GPU만 사용, CPU 폴백 없음
    return True

# ------------------------------------------------------------
# 학습 메인
# ------------------------------------------------------------
def train(data_path, model_path, scaler_path, meta_path, n_splits=4, allow_short=False, fee=0.0005):
    t0_all = time.time()

    df, X, y, feats, ts, fwd = load_dataset(data_path)
    if X.size == 0:
        raise RuntimeError("No numeric features found in dataset.")

    # dtype 최적화: GPU 메모리 효율성을 위해 float32 사용
    X = X.astype(np.float32, copy=False)

    # --- 라벨 매핑 {-1,0,1} -> {0,1,2} (XGBoost 요구사항) ---
    y_raw = y.copy()
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {v: k for k, v in label_map.items()}
    y = np.array([label_map[int(v)] for v in y_raw], dtype=int)

    # 분할
    n = len(df)
    split_idx = int(n * 0.8)
    X_tr, X_ho = X[:split_idx], X[split_idx:]
    y_tr, y_ho = y[:split_idx], y[split_idx:]
    fwd_tr, fwd_ho = fwd[:split_idx], fwd[split_idx:]

    # 기본 정보
    n_classes = int(len(np.unique(y)))
    close_ret = pd.Series(df["close"]).pct_change().values
    r_ho = close_ret[split_idx:]

    # 수익가중 샘플 웨이트
    w_tr = profit_weights(fwd_tr, alpha=8.0, target=0.003, cap=3.0)

    # 교차검증 & 파라미터 그리드
    tscv = TimeSeriesSplit(n_splits=n_splits)
    params_grid = [
        dict(learning_rate=0.08, max_depth=6, l2_regularization=0.0, min_samples_leaf=50),
        dict(learning_rate=0.05, max_depth=8, l2_regularization=0.01, min_samples_leaf=80),
        dict(learning_rate=0.03, max_depth=10, l2_regularization=0.03, min_samples_leaf=120),
    ]

    # GPU만 사용, CPU 폴백 없음
    use_gpu = True
    print(f"[INFO] dataset rows={n}, train={len(X_tr)}, holdout={len(X_ho)}")
    print(f"[INFO] CV splits={n_splits}, param_sets={len(params_grid)}, total inner steps={len(params_grid)*n_splits}", flush=True)
    print(f"[INFO] GPU ONLY mode - CPU fallback disabled", flush=True)

    total_steps = len(params_grid) * n_splits
    done_steps = 0
    best_equity_cv, best_ps = -1.0, None

    for ps_i, ps in enumerate(params_grid, start=1):
        t_ps = time.time()
        print(f"\n[PARAM] {ps_i}/{len(params_grid)} -> {ps}", flush=True)
        eq_cv = []

        for fold_i, (tr_idx, va_idx) in enumerate(tscv.split(X_tr), start=1):
            t_fold = time.time()
            print(f"  [FOLD] {fold_i}/{n_splits} | train={len(tr_idx):,}, val={len(va_idx):,} ...", flush=True)

            X_tr_i, X_va_i = X_tr[tr_idx], X_tr[va_idx]
            y_tr_i, y_va_i = y_tr[tr_idx], y_tr[va_idx]
            w_tr_i = w_tr[tr_idx]

            # GPU 전용 파라미터
            params = pick_xgb_params_from_grid(ps, n_classes=n_classes, use_gpu=True)

            # GPU에서만 학습 (실패 시 에러 발생)
            try:
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr_i, y_tr_i, sample_weight=w_tr_i)
                used_device = "cuda"
            except Exception as e:
                # GPU 실패 시 즉시 중단
                raise RuntimeError(f"GPU 학습 실패! CPU 폴백 없음. 원인: {e}")

            print(f"    - fit(XGBoost) done | device={used_device}", flush=True)

            # 확률 보정 (GPU 모델이므로 GPU 데이터 사용)
            cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
            cal.fit(X_va_i, y_va_i)
            print("    - calibration(isotonic) done", flush=True)

            # 예측 (GPU 모델이므로 GPU 데이터 사용)
            proba = cal.predict_proba(X_va_i)

            # 임계치 탐색
            longs = np.linspace(0.60, 0.75, 6)
            shorts = np.linspace(0.60, 0.75, 6)
            best_eq_split = -1.0
            best_lp, best_sp = 0.6, 0.6

            if allow_short:
                for lp in longs:
                    for sp in shorts:
                        eq = quick_equity_from_proba(
                            proba[:, 2], proba[:, 0], ret=close_ret[va_idx],
                            fee=fee, allow_short=True, long_p=lp, short_p=sp
                        )
                        if eq > best_eq_split:
                            best_eq_split, best_lp, best_sp = eq, lp, sp
            else:
                for lp in longs:
                    eq = quick_equity_from_proba(
                        proba[:, 2], None, ret=close_ret[va_idx],
                        fee=fee, allow_short=False, long_p=lp
                    )
                    if eq > best_eq_split:
                        best_eq_split, best_lp = eq, lp

            eq_cv.append(best_eq_split)

            done_steps += 1
            dt_fold = time.time() - t_fold
            elapsed = time.time() - t0_all
            eta = (elapsed / max(done_steps, 1)) * (total_steps - done_steps)
            thr_str = f"long_p={best_lp:.2f}" + (f", short_p={best_sp:.2f}" if allow_short else "")
            print(
                f"    - fold best_eq={best_eq_split:.4f} ({thr_str}) | fold_time={dt_fold:.1f}s "
                f"| progress={done_steps}/{total_steps} | ETA≈{eta/60:.1f} min",
                flush=True,
            )

        avg_eq = float(np.mean(eq_cv))
        dt_ps = time.time() - t_ps
        print(f"[RESULT] param_set avg_eq={avg_eq:.4f} | set_time={dt_ps:.1f}s", flush=True)
        if avg_eq > best_equity_cv:
            best_equity_cv, best_ps = avg_eq, ps
            print(f"[BEST] updated best_equity_cv={best_equity_cv:.4f} with params={best_ps}", flush=True)

    # 베스트 파라미터로 전체 재학습 + 보정 (GPU 전용)
    print("\n[FINAL-TRAIN] best_params ->", best_ps, flush=True)
    params_final = pick_xgb_params_from_grid(best_ps, n_classes=n_classes, use_gpu=True)
    
    # GPU에서만 학습 (실패 시 에러 발생)
    try:
        model_final = xgb.XGBClassifier(**params_final)
        t_final = time.time()
        model_final.fit(X_tr, y_tr, sample_weight=w_tr)
        used_device_final = "cuda"
    except Exception as e:
        # GPU 실패 시 즉시 중단
        raise RuntimeError(f"GPU 최종 학습 실패! CPU 폴백 없음. 원인: {e}")
    
    print(f"[FINAL-TRAIN] fit done | device={used_device_final} | time={time.time()-t_final:.1f}s", flush=True)

    # 최종 보정 (GPU 모델이므로 GPU 데이터 사용)
    cal_final = CalibratedClassifierCV(model_final, method="isotonic", cv=3).fit(X_tr, y_tr)
    print("[FINAL-TRAIN] calibration done", flush=True)

    # 홀드아웃 예측 (GPU 모델이므로 GPU 데이터 사용)
    proba_ho = cal_final.predict_proba(X_ho)
    longs = np.linspace(0.60, 0.80, 7)
    shorts = np.linspace(0.60, 0.80, 7)
    best_eq, best_lp, best_sp = -1.0, 0.6, 0.6

    if allow_short:
        for lp in longs:
            for sp in shorts:
                eq = quick_equity_from_proba(proba_ho[:, 2], proba_ho[:, 0], r_ho, fee=fee, allow_short=True, long_p=lp, short_p=sp)
                if eq > best_eq:
                    best_eq, best_lp, best_sp = eq, lp, sp
    else:
        for lp in longs:
            eq = quick_equity_from_proba(proba_ho[:, 2], None, r_ho, fee=fee, allow_short=False, long_p=lp)
            if eq > best_eq:
                best_eq, best_lp = eq, lp

    # 최종 예측 라벨 → 원라벨 {-1,0,1} 복원
    y_pred_ho = cal_final.predict(X_ho)
    y_pred_ho_orig = np.array([inv_label_map[int(v)] for v in y_pred_ho], dtype=int)
    y_ho_orig = np.array([inv_label_map[int(v)] for v in y_ho], dtype=int)

    print("=== Classification report (holdout) ===")
    print(classification_report(y_ho_orig, y_pred_ho_orig, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_ho_orig, y_pred_ho_orig))
    print(
        f"Best holdout equity={best_eq:.4f} with thresholds: long_p={best_lp:.2f}"
        + (f", short_p={best_sp:.2f}" if allow_short else "")
    )

    # 저장
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(cal_final, model_path)  # 보정된 최종 모델
    joblib.dump(None, scaler_path)      # 호환용 더미
    meta = {
        "features": feats,
        "n_train": int(len(X_tr)),
        "n_holdout": int(len(X_ho)),
        "allow_short": bool(allow_short),
        "optimized_long_p": float(best_lp),
        "optimized_short_p": float(best_sp if allow_short else 0.6),
        "fee_used": float(fee),
        "algo": "XGBoost (multi:softprob) + IsotonicCalibration (cv=3) - GPU ONLY",
        "params": best_ps,
        "device_train": "cuda",
        "label_mapping": {"-1": 0, "0": 1, "1": 2},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total_time = time.time() - t0_all
    print(f"Saved model -> {model_path}\nSaved meta  -> {meta_path}")
    print(f"[DONE] total wall time = {total_time/60:.1f} min", flush=True)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--scaler", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--splits", type=int, default=4)
    ap.add_argument("--allow_short", action="store_true")
    ap.add_argument("--fee", type=float, default=0.0005)
    args = ap.parse_args()
    train(args.data, args.model, args.scaler, args.meta,
          n_splits=args.splits, allow_short=args.allow_short, fee=args.fee)

if __name__ == "__main__":
    main()
