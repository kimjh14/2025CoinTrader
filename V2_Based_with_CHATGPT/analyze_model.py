"""
모델 분석 및 예측 과정 시각화
"""
import joblib
import numpy as np
import json
import pandas as pd

# 모델과 메타데이터 로드
model_path = 'artifacts/train/model_classic_200d_h3_0.002.joblib'
meta_path = 'artifacts/train/meta_classic_200d_h3_0.002.json'
scaler_path = 'artifacts/train/scaler_classic_200d_h3_0.002.joblib'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path) if scaler_path else None

with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

features = meta.get('features', [])
print(f"=== 모델 정보 ===")
print(f"피처 개수: {len(features)}개")
print(f"알고리즘: {meta.get('algo', 'Unknown')}")
print(f"학습 파라미터: {meta.get('params', {})}")
print(f"학습 데이터: {meta.get('n_train', 0):,}개")
print(f"검증 데이터: {meta.get('n_holdout', 0):,}개")

print(f"\n=== 피처 목록 (64개) ===")
for i, feat in enumerate(features, 1):
    print(f"{i:2d}. {feat}")

# 예시 데이터로 예측 과정 시뮬레이션
print(f"\n=== 예측 시뮬레이션 ===")

# 가상의 입력 데이터 생성 (현재 시장 상황과 유사)
sample_data = {
    'open': 163195000,
    'high': 163220000,
    'low': 163128000,
    'close': 163220000,
    'volume': 0.1074,
    'value': 17500000,
    'ret_1': 0.000564,  # 1분 수익률
    'logret_1': 0.000564,
    'ret_5': 0.000730,  # 5분 수익률
    'ret_15': 0.000564,  # 15분 수익률
    'vol_20': 0.000243,  # 20분 변동성
    'vol_60': 0.000261,  # 60분 변동성
    'ema_12': 163200000,
    'ema_26': 163210000,
    'macd': -13098.19,
    'macd_sig': -20013.70,
    'macd_hist': 6915.51,
    'rsi_14': 61.92,
    'bb_ma': 163150000,
    'bb_up': 163350000,
    'bb_dn': 162950000,
    'bb_width': 0.001216,
    'atr_14': 41220.41,
    'dist_bb_up': 0.000167,
    'dist_bb_dn': 0.001049,
    'ema_gap': -0.00006,
}

# 3분봉 지표 추가
for feat in features:
    if feat.startswith('m3_'):
        sample_data[feat] = np.random.randn() * 0.001
    elif feat.startswith('m5_'):
        sample_data[feat] = np.random.randn() * 0.001
    elif feat not in sample_data:
        sample_data[feat] = 0

# DataFrame으로 변환
X_sample = pd.DataFrame([sample_data])[features]

print("\n입력 피처 샘플 (일부):")
for feat in ['ret_1', 'rsi_14', 'macd', 'bb_width', 'vol_20']:
    if feat in sample_data:
        print(f"  {feat}: {sample_data[feat]:.6f}")

# 스케일링
if scaler:
    X_scaled = scaler.transform(X_sample)
    print("\n스케일링 후 값 (일부):")
    for i, feat in enumerate(['ret_1', 'rsi_14', 'macd', 'bb_width', 'vol_20']):
        if feat in features:
            idx = features.index(feat)
            print(f"  {feat}: {X_scaled[0][idx]:.4f}")
else:
    X_scaled = X_sample.values

# 예측
print("\n=== 예측 결과 ===")
try:
    # 확률 예측
    proba = model.predict_proba(X_scaled)[0]
    
    # 클래스 확인
    if hasattr(model, 'classes_'):
        classes = model.classes_
    else:
        classes = [-1, 0, 1]  # SHORT, FLAT, LONG
    
    print(f"클래스: {classes}")
    print(f"확률 배열: {proba}")
    
    # 각 클래스별 확률
    for i, cls in enumerate(classes):
        if cls == -1:
            print(f"P(SHORT): {proba[i]:.6f}")
        elif cls == 0:
            print(f"P(FLAT):  {proba[i]:.6f}")
        elif cls == 1:
            print(f"P(LONG):  {proba[i]:.6f}")
    
    # 결정
    long_threshold = meta.get('optimized_long_p', 0.55)
    short_threshold = meta.get('optimized_short_p', 0.55)
    
    p_long = proba[np.where(classes == 1)[0][0]] if 1 in classes else 0
    p_short = proba[np.where(classes == -1)[0][0]] if -1 in classes else 0
    
    if p_long >= long_threshold:
        decision = "LONG"
    elif p_short >= short_threshold:
        decision = "SHORT"
    else:
        decision = "FLAT"
    
    print(f"\n임계값: LONG={long_threshold:.3f}, SHORT={short_threshold:.3f}")
    print(f"최종 결정: {decision}")
    
except Exception as e:
    print(f"예측 중 오류: {e}")

print("\n=== 모델 내부 구조 (가능한 경우) ===")
# CalibratedClassifierCV 구조 분석
if hasattr(model, 'estimators_'):
    print(f"CalibratedClassifierCV 구조:")
    print(f"  - 교차검증 폴드 수: {len(model.estimators_)}")
    print(f"  - 보정 방법: {model.method}")
    
    # 첫 번째 estimator 분석
    if len(model.estimators_) > 0:
        first_est = model.estimators_[0]
        print(f"  - 첫 번째 폴드 estimator: {type(first_est)}")
        
        # 내부 모델 찾기
        if hasattr(first_est, 'estimator'):
            inner = first_est.estimator
            if hasattr(inner, 'estimator'):
                actual = inner.estimator
                print(f"  - 실제 모델: {type(actual).__name__}")
                
                # HistGradientBoostingClassifier 정보
                if hasattr(actual, 'max_depth'):
                    print(f"\n실제 모델 파라미터:")
                    print(f"  - max_depth: {actual.max_depth}")
                    print(f"  - learning_rate: {actual.learning_rate}")
                    print(f"  - min_samples_leaf: {actual.min_samples_leaf}")
                    if hasattr(actual, 'l2_regularization'):
                        print(f"  - l2_regularization: {actual.l2_regularization}")