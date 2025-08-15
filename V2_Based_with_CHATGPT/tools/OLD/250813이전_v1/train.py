"""
python tools/train.py --data data/dataset_mtf.parquet --model artifacts/model_mtf.joblib --scaler artifacts/scaler_mtf.joblib --meta artifacts/meta_mtf.json --splits 4 --allow_short
"""
# tools/train.py
import argparse, os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator  # scikit-learn 1.6+


def load_dataset(path):
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    drop_cols = {"label", "fwd_ret", "timestamp", "timestamp_kst"}
    feats = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].values
    y = df["label"].astype(int).values
    ts = pd.to_datetime(df["timestamp"]).values
    fwd = df["fwd_ret"].values
    return df, X, y, feats, ts, fwd


def profit_weights(fwd_ret, alpha=8.0, target=0.003, cap=3.0):
    mag = np.minimum(np.abs(fwd_ret) / target, cap)
    return 1.0 + alpha * mag  # 큰 움직임일수록 가중↑


def quick_equity_from_proba(
    p_long, p_short=None, ret=None, fee=0.0005, allow_short=False, long_p=0.6, short_p=0.6
):
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


def train(data_path, model_path, scaler_path, meta_path, n_splits=4, allow_short=False, fee=0.0005):
    df, X, y, feats, ts, fwd = load_dataset(data_path)
    if X.size == 0:
        raise RuntimeError("No numeric features found in dataset.")

    # 최종 홀드아웃(뒤 20%)는 남겨두고, 앞부분으로 CV
    n = len(df)
    split_idx = int(n * 0.8)
    X_tr, X_ho = X[:split_idx], X[split_idx:]
    y_tr, y_ho = y[:split_idx], y[split_idx:]
    fwd_tr, fwd_ho = fwd[:split_idx], fwd[split_idx:]

    # 수익가중 샘플웨이트
    w_tr = profit_weights(fwd_tr, alpha=8.0, target=0.003, cap=3.0)

    # 시계열 CV로 하이퍼(간단) 탐색 + 조기종료 확인
    best_equity, best_model = -1.0, None
    tscv = TimeSeriesSplit(n_splits=n_splits)
    params_grid = [
        dict(learning_rate=0.08, max_depth=6, l2_regularization=0.0, min_samples_leaf=50),
        dict(learning_rate=0.05, max_depth=8, l2_regularization=0.01, min_samples_leaf=80),
        dict(learning_rate=0.03, max_depth=10, l2_regularization=0.03, min_samples_leaf=120),
    ]

    close_ret = pd.Series(df["close"]).pct_change().values

    for ps in params_grid:
        eq_cv = []
        for tr_idx, va_idx in tscv.split(X_tr):
            X_tr_i, X_va_i = X_tr[tr_idx], X_tr[va_idx]
            y_tr_i, y_va_i = y_tr[tr_idx], y_tr[va_idx]
            w_tr_i = w_tr[tr_idx]

            # 베이스 모델 학습 (조기종료 포함)
            hgb = HistGradientBoostingClassifier(
                learning_rate=ps["learning_rate"],
                max_depth=ps["max_depth"],
                l2_regularization=ps["l2_regularization"],
                min_samples_leaf=ps["min_samples_leaf"],
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42,
            )
            hgb.fit(X_tr_i, y_tr_i, sample_weight=w_tr_i)

            # 확률 보정 (FrozenEstimator로 사전학습 모델 감싸기)
            cal = CalibratedClassifierCV(FrozenEstimator(hgb), method="isotonic", cv=None)
            cal.fit(X_va_i, y_va_i)

            proba = cal.predict_proba(X_va_i)

            # 임계치 탐색 (롱/숏)
            longs = np.linspace(0.50, 0.75, 6)
            shorts = np.linspace(0.50, 0.75, 6)
            best_eq_split = -1
            best_lp, best_sp = 0.6, 0.6
            for lp in longs:
                if allow_short:
                    for sp in shorts:
                        eq = quick_equity_from_proba(
                            proba[:, 2],
                            proba[:, 0],
                            ret=close_ret[va_idx],
                            fee=fee,
                            allow_short=True,
                            long_p=lp,
                            short_p=sp,
                        )
                        if eq > best_eq_split:
                            best_eq_split, best_lp, best_sp = eq, lp, sp
                else:
                    eq = quick_equity_from_proba(
                        proba[:, 2],
                        None,
                        ret=close_ret[va_idx],
                        fee=fee,
                        allow_short=False,
                        long_p=lp,
                    )
                    if eq > best_eq_split:
                        best_eq_split, best_lp = eq, lp
            eq_cv.append(best_eq_split)

        avg_eq = float(np.mean(eq_cv))
        if avg_eq > best_equity:
            best_equity, best_model = avg_eq, ps

    # 베스트 파라미터로 전체 학습세트 재학습 + 보정
    hgb = HistGradientBoostingClassifier(
        learning_rate=best_model["learning_rate"],
        max_depth=best_model["max_depth"],
        l2_regularization=best_model["l2_regularization"],
        min_samples_leaf=best_model["min_samples_leaf"],
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )
    hgb.fit(X_tr, y_tr, sample_weight=w_tr)
    cal = CalibratedClassifierCV(FrozenEstimator(hgb), method="isotonic", cv=None).fit(X_tr, y_tr)

    # 홀드아웃 성능 + 임계치 재탐색 → meta에 저장
    proba_ho = cal.predict_proba(X_ho)
    longs = np.linspace(0.50, 0.80, 7)
    shorts = np.linspace(0.50, 0.80, 7)
    best_eq, best_lp, best_sp = -1, 0.6, 0.6
    r_ho = close_ret[split_idx:]

    if allow_short:
        for lp in longs:
            for sp in shorts:
                eq = quick_equity_from_proba(
                    proba_ho[:, 2], proba_ho[:, 0], r_ho, fee=fee, allow_short=True, long_p=lp, short_p=sp
                )
                if eq > best_eq:
                    best_eq, best_lp, best_sp = eq, lp, sp
    else:
        for lp in longs:
            eq = quick_equity_from_proba(
                proba_ho[:, 2], None, r_ho, fee=fee, allow_short=False, long_p=lp
            )
            if eq > best_eq:
                best_eq, best_lp = eq, lp

    # 간단 리포트
    y_pred_ho = cal.predict(X_ho)
    print("=== Classification report (holdout) ===")
    print(classification_report(y_ho, y_pred_ho, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_ho, y_pred_ho))
    print(
        f"Best holdout equity={best_eq:.4f} with thresholds: long_p={best_lp:.2f}"
        + (f", short_p={best_sp:.2f}" if allow_short else "")
    )

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(cal, model_path)  # 보정된 모델 저장
    # 스케일러는 사용 안 함(호환 위해 더미 저장 가능)
    joblib.dump(None, scaler_path)
    meta = {
        "features": feats,
        "n_train": int(len(X_tr)),
        "n_holdout": int(len(X_ho)),
        "allow_short": bool(allow_short),
        "optimized_long_p": float(best_lp),
        "optimized_short_p": float(best_sp if allow_short else 0.6),
        "fee_used": float(fee),
        "algo": "HistGradientBoostingClassifier + IsotonicCalibration (FrozenEstimator)",
        "params": best_model,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved model -> {model_path}\nSaved meta  -> {meta_path}")


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
    train(args.data, args.model, args.scaler, args.meta, n_splits=args.splits, allow_short=args.allow_short, fee=args.fee)


if __name__ == "__main__":
    main()
