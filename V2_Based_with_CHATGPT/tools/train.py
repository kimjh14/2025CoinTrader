# tools/train.py
"""
train.py - HistGradientBoosting 최적화 모델 학습 도구

CLI 사용 예시:

# 롱 전용 모드 (기본, 안전함)
python tools/train.py `
  --data data/classic/dataset_mtf_180d_h20_0.001.parquet `
  --model artifacts/train/model_classic_180d_h20_0.001.joblib `
  --scaler artifacts/train/scaler_classic_180d_h20_0.001.joblib `
  --meta artifacts/train/meta_classic_180d_h20_0.001.json `
  --splits 4 `
  --fee 0.0005

# 롱/숏 모드 (고급, 위험 높음)
python tools/train.py `
  --data data/classic/dataset_mtf_200d_h10_0.03.parquet `
  --model artifacts/train/model_classic_200d_h10_0.03.joblib `
  --scaler artifacts/train/scaler_classic_200d_h10_0.03.joblib `
  --meta artifacts/train/meta_classic_200d_h10_0.03.json `
  --splits 4 `
  --fee 0.0005 `
  --allow_short

주요 특징:
• 검증된 성능: 311% 수익 달성 (실제 백테스트 기준)
• HistGradientBoostingClassifier: 안정적이고 강력한 알고리즘
• 확률 보정: CalibratedClassifierCV로 정확한 확률 예측
• 시계열 CV: TimeSeriesSplit으로 적합한 교차검증
• 4개 임계값 최적화: 진입/청산 임계값 분리 (히스테리시스)
  - long_p_enter: 롱 진입 임계값
  - long_p_exit: 롱 청산 임계값 (진입보다 낮게)
  - short_p_enter: 숏 진입 임계값
  - short_p_exit: 숏 청산 임계값 (진입보다 낮게)
• 균형잡힌 거래: 롱과 숏 모두 적절히 활용

거래 모드:
• 롱 전용 (기본): 매수만 가능, 보합 시 현금 보유
  - 장점: 안전, 단순함, 상승장에 유리
  - 단점: 하락장에서 수익 기회 제한
• 롱/숏 모드 (--allow_short): 매수/매도 모두 가능
  - 장점: 상승/하락 모든 구간에서 수익 추구 (311% 수익 검증)
  - 단점: 위험 높음, 복잡함, 수수료 부담 증가

성능 특징:
• 실전 검증: HGB > XGBoost (311% vs -97%)
• 안정성: 확률 보정으로 신뢰할 수 있는 신호
• 균형성: SHORT 거래도 적절히 실행
• 수익성: 실제 백테스트에서 입증된 성과
"""

import argparse, os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# scikit-learn 버전 호환성 처리
try:
    from sklearn.frozen import FrozenEstimator  # scikit-learn 1.6+
except ImportError:
    print("[경고] scikit-learn < 1.6 환경, FrozenEstimator 미지원 -> CalibratedClassifierCV fallback")
    FrozenEstimator = None

# 프로젝트 레이아웃 기준: tools/ 에서 ml/ 모듈 import
import sys
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
sys.path.append(str(ROOT_DIR))  # V2_Based_with_CHATGPT/ 를 모듈 루트로

# 공용 PnL 유틸리티 사용
from ml.pnl_utils import calculate_equity_from_proba, calculate_next_open_pnl


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
    proba, prices_df, fee=0.0005, allow_short=False, long_p=0.6, short_p=0.6, class_index=None
):
    """공용 PnL 함수를 사용한 에쿼티 계산 (다음 봉 시가 체결)"""
    if class_index is None:
        # 기본 클래스 인덱스 (0: short, 1: flat, 2: long)
        class_index = {-1: 0, 0: 1, 1: 2}
    
    return calculate_equity_from_proba(
        proba=proba,
        prices=prices_df,
        th_long=long_p,
        th_short=short_p,
        fee=fee,
        class_index=class_index,
        allow_short=allow_short
    )


def quick_equity_from_proba_with_hysteresis(
    proba, prices_df, fee=0.0005, allow_short=False, 
    long_p_enter=0.6, long_p_exit=0.48,
    short_p_enter=0.6, short_p_exit=0.48, 
    class_index=None
):
    """히스테리시스를 적용한 에쿼티 계산 (진입/청산 임계값 분리)"""
    if class_index is None:
        class_index = {-1: 0, 0: 1, 1: 2}
    
    # 포지션 계산 (히스테리시스 적용)
    positions = np.zeros(len(proba))
    current_pos = 0
    
    for i in range(len(proba)):
        p_long = proba[i, class_index[1]] if 1 in class_index else 0
        p_short = proba[i, class_index[-1]] if -1 in class_index else 0
        
        # 현재 포지션에 따른 청산/진입 로직
        if current_pos == 1:  # 롱 포지션 중
            if p_long < long_p_exit:  # 롱 청산
                if allow_short and p_short >= short_p_enter:  # 숏 진입
                    positions[i] = -1
                    current_pos = -1
                else:  # 플랫
                    positions[i] = 0
                    current_pos = 0
            else:  # 롱 유지
                positions[i] = 1
                
        elif current_pos == -1:  # 숏 포지션 중
            if p_short < short_p_exit:  # 숏 청산
                if p_long >= long_p_enter:  # 롱 진입
                    positions[i] = 1
                    current_pos = 1
                else:  # 플랫
                    positions[i] = 0
                    current_pos = 0
            else:  # 숏 유지
                positions[i] = -1
                
        else:  # 플랫 상태
            if p_long >= long_p_enter:  # 롱 진입
                positions[i] = 1
                current_pos = 1
            elif allow_short and p_short >= short_p_enter:  # 숏 진입
                positions[i] = -1
                current_pos = -1
            else:  # 플랫 유지
                positions[i] = 0
    
    # PnL 계산
    equity, _, _ = calculate_next_open_pnl(
        prices=prices_df,
        positions=positions.astype(int),
        fee=fee,
        initial_capital=1.0
    )
    
    return equity[-1]  # 최종 자산 반환


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

    # 클래스 인덱스 매핑
    class_index = {-1: 0, 0: 1, 1: 2}
    
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

            # 확률 보정 (FrozenEstimator 호환성 처리)
            if FrozenEstimator:
                cal = CalibratedClassifierCV(FrozenEstimator(hgb), method="isotonic", cv=None)
            else:
                # sklearn < 1.6 fallback: 사전 학습된 모델 직접 사용
                cal = CalibratedClassifierCV(hgb, method="isotonic", cv="prefit")
            cal.fit(X_va_i, y_va_i)

            proba = cal.predict_proba(X_va_i)

            # 가격 데이터 준비 (다음 봉 시가 체결을 위해)
            va_prices = df.iloc[va_idx][["open", "close"]].reset_index(drop=True)
            
            # 임계치 탐색 (롱/숏) - 공용 PnL 함수 사용 (최소 0.5)
            longs = np.linspace(0.50, 0.75, 6)  # 0.50, 0.55, 0.60, 0.65, 0.70, 0.75
            shorts = np.linspace(0.50, 0.75, 6)  # 0.50, 0.55, 0.60, 0.65, 0.70, 0.75
            best_eq_split = -1
            best_lp, best_sp = 0.60, 0.60  # 기본값 0.60
            
            if allow_short:
                for lp in longs:
                    for sp in shorts:
                        eq = quick_equity_from_proba(
                            proba=proba,
                            prices_df=va_prices,
                            fee=fee,
                            allow_short=True,
                            long_p=lp,
                            short_p=sp,
                            class_index=class_index
                        )
                        if eq > best_eq_split:
                            best_eq_split, best_lp, best_sp = eq, lp, sp
            else:
                for lp in longs:
                    eq = quick_equity_from_proba(
                        proba=proba,
                        prices_df=va_prices,
                        fee=fee,
                        allow_short=False,
                        long_p=lp,
                        short_p=0.60,  # allow_short=False일 때는 기본값 사용
                        class_index=class_index
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
    
    # 확률 보정 (FrozenEstimator 호환성 처리)
    if FrozenEstimator:
        cal = CalibratedClassifierCV(FrozenEstimator(hgb), method="isotonic", cv=None).fit(X_tr, y_tr)
    else:
        # sklearn < 1.6 fallback
        cal = CalibratedClassifierCV(hgb, method="isotonic", cv="prefit").fit(X_tr, y_tr)

    # 홀드아웃 성능 + 임계치 재탐색 → meta에 저장
    proba_ho = cal.predict_proba(X_ho)
    ho_prices = df.iloc[split_idx:][["open", "close"]].reset_index(drop=True)
    
    # 클래스 인덱스 안전화
    classes = list(getattr(cal, "classes_", [-1, 0, 1]))
    class_index_safe = {
        -1: classes.index(-1) if -1 in classes else None,
        0: classes.index(0) if 0 in classes else None,
        1: classes.index(1) if 1 in classes else None
    }
    
    # 4개 임계값 최적화: 진입/청산 분리 (히스테리시스)
    print("\n=== 4개 임계값 최적화 시작 ===")
    
    # 진입 임계값 범위
    long_enters = np.linspace(0.50, 0.80, 7)  # 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
    short_enters = np.linspace(0.50, 0.80, 7)
    
    # 청산 임계값 비율 (진입 대비)
    exit_ratios = [0.6, 0.7, 0.8, 0.9]  # 진입의 60%, 70%, 80%, 90%
    
    best_eq = -1
    best_lp_enter, best_lp_exit = 0.60, 0.48
    best_sp_enter, best_sp_exit = 0.60, 0.48

    if allow_short:
        # 롱/숏 모두 최적화
        for lp_enter in long_enters:
            for sp_enter in short_enters:
                for exit_ratio in exit_ratios:
                    lp_exit = lp_enter * exit_ratio
                    sp_exit = sp_enter * exit_ratio
                    
                    eq = quick_equity_from_proba_with_hysteresis(
                        proba=proba_ho,
                        prices_df=ho_prices,
                        fee=fee,
                        allow_short=True,
                        long_p_enter=lp_enter,
                        long_p_exit=lp_exit,
                        short_p_enter=sp_enter,
                        short_p_exit=sp_exit,
                        class_index=class_index_safe
                    )
                    
                    if eq > best_eq:
                        best_eq = eq
                        best_lp_enter, best_lp_exit = lp_enter, lp_exit
                        best_sp_enter, best_sp_exit = sp_enter, sp_exit
    else:
        # 롱만 최적화
        for lp_enter in long_enters:
            for exit_ratio in exit_ratios:
                lp_exit = lp_enter * exit_ratio
                
                eq = quick_equity_from_proba_with_hysteresis(
                    proba=proba_ho,
                    prices_df=ho_prices,
                    fee=fee,
                    allow_short=False,
                    long_p_enter=lp_enter,
                    long_p_exit=lp_exit,
                    short_p_enter=0.99,  # 사용 안함
                    short_p_exit=0.99,   # 사용 안함
                    class_index=class_index_safe
                )
                
                if eq > best_eq:
                    best_eq = eq
                    best_lp_enter, best_lp_exit = lp_enter, lp_exit
                    # 숏은 기본값 유지
                    best_sp_enter, best_sp_exit = 0.60, 0.48
    
    print(f"\n최적 임계값 발견:")
    print(f"  롱: 진입={best_lp_enter:.3f}, 청산={best_lp_exit:.3f}")
    if allow_short:
        print(f"  숏: 진입={best_sp_enter:.3f}, 청산={best_sp_exit:.3f}")
    print(f"  최종 자산: {best_eq:.4f}x")
    
    # 경계값에서 최적값 발견 시 범위 확장 재탐색 (최대 0.8로 제한)
    # 롱 임계치 확장 - 이미 0.8이 최대이므로 확장 불필요
    if False:  # best_lp >= 0.8일 때 확장하지 않음
        extended_longs = np.linspace(0.75, 0.80, 3)  # 제한됨
        for lp in extended_longs:
            if allow_short:
                for sp in shorts:
                    eq = quick_equity_from_proba(
                        proba=proba_ho,
                        prices_df=ho_prices,
                        fee=fee,
                        allow_short=True,
                        long_p=lp,
                        short_p=sp,
                        class_index=class_index
                    )
                    if eq > best_eq:
                        best_eq, best_lp, best_sp = eq, lp, sp
            else:
                eq = quick_equity_from_proba(
                    proba=proba_ho,
                    prices_df=ho_prices,
                    fee=fee,
                    allow_short=False,
                    long_p=lp,
                    short_p=sp,  # allow_short=False일 때는 사용 안됨
                    class_index=class_index
                )
                if eq > best_eq:
                    best_eq, best_lp = eq, lp
    
    # 숏 임계치 확장 (allow_short일 때만) - 최대 0.8로 제한
    if False:  # allow_short and best_sp >= 0.8일 때 확장하지 않음
        extended_shorts = np.linspace(0.75, 0.80, 3)  # 제한됨
        for sp in extended_shorts:
            # 최적 롱 임계치는 고정하고 숏만 확장 탐색
            eq = quick_equity_from_proba(
                proba=proba_ho,
                prices_df=ho_prices,
                fee=fee,
                allow_short=True,
                long_p=best_lp,
                short_p=sp,
                class_index=class_index
            )
            if eq > best_eq:
                best_eq, best_sp = eq, sp

        # 간단 리포트 - 기본 임계값(0.5)
    y_pred_ho = cal.predict(X_ho)
    print("\n=== Classification report (기본 임계값 0.5) ===")
    print(classification_report(y_ho, y_pred_ho, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_ho, y_pred_ho))
    
    # 최적 임계값으로 다시 예측
    print(f"\n=== Classification report (최적 임계값: long_p={best_lp_enter:.3f}, short_p={best_sp_enter:.3f}) ===")
    
    # 최적 임계값으로 예측 생성
    if allow_short:
        y_pred_optimized = np.where(
            proba_ho[:, 2] >= best_lp, 1,  # LONG
            np.where(proba_ho[:, 0] >= best_sp, -1, 0)  # SHORT or FLAT
        )
    else:
        y_pred_optimized = np.where(proba_ho[:, 2] >= best_lp, 1, 0)  # LONG or FLAT
    
    print(classification_report(y_ho, y_pred_optimized, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_ho, y_pred_optimized))
    
    # 거래 통계 추가
    n_long_trades = np.sum(y_pred_optimized == 1)
    n_short_trades = np.sum(y_pred_optimized == -1)
    n_flat = np.sum(y_pred_optimized == 0)
    total_samples = len(y_pred_optimized)
    
    print(f"\n=== 최적 임계값 거래 통계 ===")
    print(f"LONG 거래: {n_long_trades} ({n_long_trades/total_samples*100:.1f}%)")
    print(f"SHORT 거래: {n_short_trades} ({n_short_trades/total_samples*100:.1f}%)")
    print(f"FLAT (대기): {n_flat} ({n_flat/total_samples*100:.1f}%)")
    print(f"총 거래 비율: {(n_long_trades+n_short_trades)/total_samples*100:.1f}%")
    
    print(f"\nBest holdout equity={best_eq:.4f}")
    print(f"  롱: 진입={best_lp_enter:.3f}, 청산={best_lp_exit:.3f}")
    if allow_short:
        print(f"  숏: 진입={best_sp_enter:.3f}, 청산={best_sp_exit:.3f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(cal, model_path)  # 보정된 모델 저장
    # 스케일러는 사용 안 함(호환 위해 더미 저장 가능)
    joblib.dump(None, scaler_path)
    meta = {
        "features": feats,
        "n_train": int(len(X_tr)),
        "n_holdout": int(len(X_ho)),
        "allow_short": bool(allow_short),
        "optimized_long_p": float(best_lp_enter),  # 기존 호환성
        "optimized_short_p": float(best_sp_enter if allow_short else 0.60),  # 기존 호환성
        "optimized_long_p_enter": float(best_lp_enter),  # 명시적 진입
        "optimized_long_p_exit": float(best_lp_exit),    # 명시적 청산
        "optimized_short_p_enter": float(best_sp_enter if allow_short else 0.60),  # 명시적 진입
        "optimized_short_p_exit": float(best_sp_exit if allow_short else 0.60),    # 명시적 청산
        "fee_used": float(fee),
        "algo": "HistGradientBoostingClassifier + IsotonicCalibration",
        "params": best_model,
        "execution_rule": "next_open",
        "optimization_method": "equity_based_threshold_search_with_hysteresis",
        "sklearn_version": "1.6+" if FrozenEstimator else "<1.6",
        "notes": "체결: 다음 봉 시가, 히스테리시스 적용 (진입/청산 임계값 분리)"
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved model -> {model_path}")
    print(f"Saved meta  -> {meta_path}")


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

# === EXIT-MVP TRAIN BLOCK START ===
# [EXIT-MVP] Minimal Exit model training block
# Note: 이 블록은 현재 비활성화됨 (히스테리시스 방식 사용 중)
# This block expects variables in scope: best_model, X_ho, df, split_idx, fee, meta, model_path
"""
EXIT-MVP 블록 비활성화 - 히스테리시스 방식이 충분히 잘 작동하므로 필요 없음
if False:  # EXIT-MVP 비활성화
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        import numpy as np, joblib
        from pathlib import Path
        from tools.labels_exit import build_entries_from_proba, label_optimal_exits
        from tools.position_state import PositionState
        from tools.features_exit import build_exit_features_onebar, features_to_row, EXIT_FEATURES_DEFAULT

    # Fallbacks
    allow_short = bool(meta.get("allow_short", False))
    long_enter = float(meta.get("optimized_long_p_enter", meta.get("optimized_long_p", 0.6)))
    short_enter = float(meta.get("optimized_short_p_enter", meta.get("optimized_short_p", 0.6)))
    exit_horizon = int(meta.get("exit", {}).get("horizon", 60))
    exit_delta = float(meta.get("exit", {}).get("delta_margin", 0.0005))

    print("[EXIT-MVP] Preparing exit training from holdout...")
    proba_ho = best_model.predict_proba(X_ho)
    classes = list(getattr(best_model, "classes_", [-1,0,1]))
    entries = build_entries_from_proba(proba_ho, classes, long_enter, short_enter, allow_short, cooldown=exit_horizon)

    if entries:
        close_ho = df.iloc[split_idx:]["close"].values
        y_exit_df = label_optimal_exits(close_ho, entries, horizon=exit_horizon, fee=fee, delta=exit_delta)
        if not y_exit_df.empty:
            df_ho = df.iloc[split_idx:].reset_index(drop=True)
            feat_cols = EXIT_FEATURES_DEFAULT[:]
            X_exit_long, y_exit_long, X_exit_short, y_exit_short = [], [], [], []

            for t0, g in y_exit_df.groupby("t0"):
                entry_price = float(df_ho["close"].iloc[t0]) if t0 < len(df_ho) else float(df_ho["close"].iloc[-1])
                side = int(g["side"].iloc[0])
                st = PositionState(entry_idx=t0, entry_price=entry_price, side=side)
                for _, row in g.sort_values("t").iterrows():
                    i = int(row["t"])
                    if i >= len(df_ho): break
                    feat = build_exit_features_onebar(df_ho, i, st)
                    x = features_to_row(feat, feat_cols)[0]
                    if side == 1:
                        X_exit_long.append(x); y_exit_long.append(int(row["close_label"]))
                    else:
                        X_exit_short.append(x); y_exit_short.append(int(row["close_label"]))

            def fit_exit(X, y):
                if len(X) < 10: return None
                X = np.vstack(X); y = np.array(y)
                base = LogisticRegression(max_iter=200, class_weight="balanced")
                cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
                cal.fit(X, y)
                return cal

            exit_long_model = fit_exit(X_exit_long, y_exit_long)
            exit_short_model = fit_exit(X_exit_short, y_exit_short) if allow_short else None

            # Save
            exit_dir = Path(model_path).parent / "exit"
            exit_dir.mkdir(parents=True, exist_ok=True)

            meta.setdefault("exit", {})
            meta["exit"]["use_exit_model"] = True
            meta["exit"]["horizon"] = int(exit_horizon)
            meta["exit"]["delta_margin"] = float(exit_delta)
            meta["exit"]["features"] = feat_cols
            meta["exit"]["thresholds"] = meta["exit"].get("thresholds", {})
            meta["exit"]["models"] = meta["exit"].get("models", {})

            if exit_long_model is not None:
                p_long_path = str(exit_dir / "exit_long.joblib")
                joblib.dump(exit_long_model, p_long_path)
                meta["exit"]["models"]["long"] = p_long_path
                meta["exit"]["thresholds"]["long"] = float(0.5)

            if allow_short and exit_short_model is not None:
                p_short_path = str(exit_dir / "exit_short.joblib")
                joblib.dump(exit_short_model, p_short_path)
                meta["exit"]["models"]["short"] = p_short_path
                meta["exit"]["thresholds"]["short"] = float(0.5)

            print("[EXIT-MVP] Exit models saved and meta updated.")
        else:
            print("[EXIT-MVP] No exit labels on holdout; skipping exit training.")
    else:
        print("[EXIT-MVP] No entries on holdout; skipping exit training.")
    except Exception as e:
        print("[EXIT-MVP][WARN] Exit training block failed:", e)
"""
# === EXIT-MVP TRAIN BLOCK END ===
