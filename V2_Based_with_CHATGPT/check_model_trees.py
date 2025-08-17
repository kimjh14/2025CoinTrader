"""
모델 내부 트리 구조 확인 스크립트
"""
import joblib
import numpy as np
import json

# 모델 로드
model_path = 'artifacts/train/model_classic_200d_h3_0.002.joblib'
model = joblib.load(model_path)

print('=== 모델 구조 ===')
print(f'전체 모델 타입: {type(model).__name__}')
print(f'모델 속성들: {dir(model)[:10]}...')

# CalibratedClassifierCV 내부 접근
base_model = model
if hasattr(model, 'estimators_'):
    # sklearn의 CalibratedClassifierCV
    print(f'Estimators 개수: {len(model.estimators_)}')
    if len(model.estimators_) > 0:
        # FrozenEstimator를 거쳐서 실제 모델에 접근
        frozen = model.estimators_[0].estimator
        print(f'Frozen 모델 타입: {type(frozen).__name__}')
        
        # FrozenEstimator 내부의 실제 estimator 접근
        if hasattr(frozen, 'estimator'):
            base_model = frozen.estimator
            print(f'실제 기본 모델 타입: {type(base_model).__name__}')
        else:
            base_model = frozen

if base_model:
    print(f'\n기본 모델 속성들: {[attr for attr in dir(base_model) if not attr.startswith("_")][:15]}')

# HistGradientBoostingClassifier 정보
if hasattr(base_model, 'n_trees_per_iteration_'):
    print('\n=== HistGradientBoostingClassifier 파라미터 ===')
    n_trees = base_model.n_trees_per_iteration_
    # n_trees_per_iteration_가 int인지 배열인지 확인
    if isinstance(n_trees, int):
        n_trees_val = n_trees
    else:
        n_trees_val = n_trees[0] if len(n_trees) > 0 else n_trees
    
    n_iter = base_model.n_iter_ if hasattr(base_model, 'n_iter_') else 100
    print(f'클래스별 트리: {n_trees_val}개')
    print(f'반복 횟수: {n_iter}회')
    print(f'총 트리 개수: {n_trees_val * n_iter}개')
    print(f'최대 깊이: {base_model.max_depth}')
    print(f'학습률: {base_model.learning_rate}')
    print(f'최소 샘플 (리프): {base_model.min_samples_leaf}')
    print(f'L2 정규화: {base_model.l2_regularization}')
    
print('\n=== 피처 중요도 TOP 30 ===')
# 메타데이터에서 피처 이름 가져오기
with open('artifacts/train/meta_classic_200d_h3_0.002.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)
features = meta.get('features', [])

if hasattr(base_model, 'feature_importances_'):
    importances = base_model.feature_importances_
    indices = np.argsort(importances)[::-1][:30]
    
    for i, idx in enumerate(indices, 1):
        if idx < len(features):
            print(f'{i:2d}. {features[idx]:20s}: {importances[idx]:.4f} ({importances[idx]*100:.1f}%)')

# 트리 구조 상세 정보
print('\n=== 트리 구조 상세 ===')
if hasattr(base_model, '_predictors'):
    predictors = base_model._predictors
    print(f'트리 그룹 수: {len(predictors)}')
    
    # 첫 번째 트리 그룹 분석
    first_group = predictors[0]
    print(f'첫 번째 그룹의 트리 수: {len(first_group)}')
    
    # 첫 번째 트리 상세
    tree = first_group[0]
    print(f'\n첫 번째 트리 정보:')
    print(f'- 전체 노드 수: {len(tree.nodes)}')
    print(f'- 리프 노드 수: {tree.nodes["is_leaf"].sum()}')
    print(f'- 내부 노드 수: {(~tree.nodes["is_leaf"]).sum()}')
    
    # 분할 기준 피처 확인
    print(f'\n자주 사용되는 분할 피처 (첫 10개 트리):')
    feature_splits = {}
    for i in range(min(10, len(first_group))):
        tree = first_group[i]
        splits = tree.nodes[~tree.nodes["is_leaf"]]["feature_idx"]
        for feat_idx in splits:
            if feat_idx < len(features):
                feat_name = features[feat_idx]
                feature_splits[feat_name] = feature_splits.get(feat_name, 0) + 1
    
    # 정렬하여 출력
    sorted_splits = sorted(feature_splits.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat_name, count in sorted_splits:
        print(f'  {feat_name}: {count}회 사용')

print('\n=== 예측 임계값 ===')
print(f'LONG 임계값: {meta.get("optimized_long_p", 0.5):.3f}')
print(f'SHORT 임계값: {meta.get("optimized_short_p", 0.5):.3f}')
print(f'숏 허용: {meta.get("allow_short", False)}')