# Exit MVP 패키지 (진입/청산 분리 모델)

진입 후 **청산(close) 또는 포지션 유지(hold)** 결정을 위한 별도 모델 추가 기능

## 현재 상태 - 히스테리시스 방식 구현 완료

현재 코드베이스는 Exit MVP 대신 **히스테리시스 방식**으로 구현되어 있습니다:
- **4개 임계값 사용**: long_p_enter, long_p_exit, short_p_enter, short_p_exit
- **진입 임계값 > 청산 임계값**: 빈번한 거래 방지
- **현재 포지션 고려**: 포지션 상태에 따라 다른 임계값 적용

## Exit MVP 파일 (미통합 상태)
다음 파일들은 존재하지만 아직 메인 코드에 통합되지 않았습니다:

### tools/position_state.py
- **기능**: 포지션 상태 추적 (entry_idx, entry_price, side, bars_held, peak_price, trough_price)
- **사용**: 포지션 보유 중 MFE(Maximum Favorable Excursion), 드로다운 등 계산

### tools/features_exit.py  
- **기능**: 청산 결정을 위한 피처 생성
- **피처**: r_unreal(미실현 수익률), bars_held(보유 기간), mfe(최대 유리 이동), dd_from_peak(최고점 대비 하락), atr_rel, rsi_14, bb_width

### tools/labels_exit.py
- **기능**: 최적 청산 시점 레이블링
- **방법**: 진입 후 horizon(기본 60분) 내에서 최대 수익 시점 찾기

## 현재 구현 방식

### 학습 (tools/train.py)
```powershell
# 롱 전용 모드
python tools/train.py `
  --data data/classic/dataset_mtf_10d_h2_0.001.parquet `
  --model artifacts/train/model_classic_10d_h2_0.001.joblib `
  --scaler artifacts/train/scaler_classic_10d_h2_0.001.joblib `
  --meta artifacts/train/meta_classic_10d_h2_0.001.json `
  --splits 4 `
  --fee 0.0005

# 롱/숏 모드
python tools/train.py `
  --data data/classic/dataset_mtf_10d_h2_0.001.parquet `
  --model artifacts/train/model_classic_10d_h2_0.001.joblib `
  --scaler artifacts/train/scaler_classic_10d_h2_0.001.joblib `
  --meta artifacts/train/meta_classic_10d_h2_0.001.json `
  --splits 4 `
  --fee 0.0005 `
  --allow_short
```

**저장되는 임계값**:
- `optimized_long_p_enter`: 롱 진입 임계값
- `optimized_long_p_exit`: 롱 청산 임계값 (진입의 60-90%)
- `optimized_short_p_enter`: 숏 진입 임계값
- `optimized_short_p_exit`: 숏 청산 임계값 (진입의 60-90%)

### 백테스트 (tools/backtest.py)
```powershell
python tools/backtest.py `
  --data data/classic/dataset_mtf_5d_h2_0.001.parquet `
  --model artifacts/train/model_classic_10d_h2_0.001.joblib `
  --meta artifacts/train/meta_classic_10d_h2_0.001.json `
  --scaler artifacts/train/scaler_classic_10d_h2_0.001.joblib `
  --fee 0.0005 `
  --report artifacts/backtest/bt_classic_5d_h2_0.001.json `
  --log_csv artifacts/backtest/trades_classic_5d_h2_0.001.csv `
  --auto_md `
  --md_out artifacts/backtest/trades_classic_5d_h2_0.001.md
```

**히스테리시스 로직**:
```python
# 현재 포지션에 따른 임계값 적용
if current_pos == 1:  # 롱 포지션
    if p_long < long_p_exit:  # 청산
        pos = 0 or -1  # 플랫 또는 숏 전환
elif current_pos == -1:  # 숏 포지션
    if p_short < short_p_exit:  # 청산
        pos = 0 or 1  # 플랫 또는 롱 전환
else:  # 플랫
    if p_long >= long_p_enter:  # 롱 진입
        pos = 1
    elif p_short >= short_p_enter:  # 숏 진입
        pos = -1
```

### 라이브 페이퍼 (tools/live_paper_v2.py)
```powershell
python tools/live_paper_v2.py `
  --market KRW-BTC `
  --model artifacts/train/model_classic_10d_h2_0.001.joblib `
  --meta artifacts/train/meta_classic_10d_h2_0.001.json `
  --scaler artifacts/train/scaler_classic_10d_h2_0.001.joblib `
  --fee 0.0005 `
  --state artifacts/live/state_v2.json `
  --log_csv artifacts/live/trades_v2.csv `
  --once
```

## Exit MVP vs 히스테리시스 비교

### 현재 구현: 히스테리시스 방식 (✅ 사용 중)
- **단순함**: 하나의 모델로 진입/청산 모두 처리
- **4개 임계값**: 진입/청산 각각 다른 임계값 사용
- **빠른 백테스트**: 추가 모델 없이 빠른 실행
- **안정성**: 빈번한 거래 방지 효과

### Exit MVP 방식 (향후 가능)
- **정교함**: 진입 모델 + 별도 청산 모델
- **동적 피처**: 포지션 보유 중 상태 변화 추적 (MFE, 드로다운 등)
- **최적화**: 진입/청산 각각 독립적 최적화
- **복잡도**: 구현 및 유지보수 복잡

## 통합 방법 (필요 시)

Exit MVP를 통합하려면:

1. **train.py에 EXIT-MVP TRAIN BLOCK 추가**:
   - 홀드아웃 데이터로 진입 시점 생성
   - 각 진입에 대해 최적 청산 시점 레이블링
   - CalibratedClassifierCV로 청산 모델 학습
   - meta["exit"]에 청산 모델 저장

2. **backtest.py에 EXIT-MVP BACKTEST BLOCK 추가**:
   ```python
   use_exit = bool(meta.get("exit", {}).get("use_exit_model", False))
   if use_exit:
       # Exit 모델 사용
       from tools.features_exit import build_exit_features_onebar
       from tools.position_state import PositionState
       # ... 청산 모델 예측 로직
   else:
       # 현재 히스테리시스 방식
       pos, p_long, p_short, th = positions_from_proba(...)
   ```

3. **live_paper_v2.py에 EXIT-MVP LIVE BLOCK 추가**:
   - 실시간 포지션 상태 추적
   - 청산 모델 예측으로 청산 결정

## 결론

현재 히스테리시스 방식이 잘 작동하고 있으므로, Exit MVP는 성능 개선이 필요할 때 추가 구현하는 것을 권장합니다.
