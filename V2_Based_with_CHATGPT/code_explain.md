# V2_Based_with_CHATGPT 코드 설명

## 📂 프로젝트 구조

```
V2_Based_with_CHATGPT/
├── tools/                      # 핵심 도구 모음
│   ├── collect.py              # 업비트 API 데이터 수집
│   ├── build_dataset.py        # 기술지표 계산 및 데이터셋 생성
│   ├── train.py                # 모델 학습 (HistGradientBoosting)
│   ├── backtest.py             # 백테스팅 엔진
│   ├── backtest_period.py      # 기간별 백테스팅
│   ├── live_paper_v2.py        # 실시간 페이퍼 트레이딩
│   ├── config.py               # 전역 설정
│   ├── batch_data.py           # 일괄 데이터셋 생성
│   ├── batch_train.py          # 일괄 모델 학습
│   ├── batch_back.py           # 일괄 백테스팅
│   └── one_click_pipeline.py   # 원클릭 전체 파이프라인
├── ml/                         # 머신러닝 유틸리티
│   ├── indicators.py           # 기술지표 계산 함수
│   └── pnl_utils.py           # 수익률 계산 유틸리티
├── data/                       # 데이터 저장소
│   ├── raw/                    # 원본 OHLCV 데이터
│   ├── classic/                # CLASSIC 모드 데이터셋
│   └── seq/                    # SEQ 모드 데이터셋
├── artifacts/                  # 학습 결과물
│   ├── train/                  # 학습된 모델/스케일러
│   ├── backtest/               # 백테스트 결과
│   └── live/                   # 실시간 거래 상태
├── analyze_backtest_results.py # 백테스트 결과 종합 분석
├── analyze_best_trade.py       # 최고 수익 거래 상세 분석
└── analyze_model.py           # 모델 성능 분석

```

## 🔧 핵심 도구 설명

### 1. **collect.py** - 데이터 수집
```python
# 업비트에서 1분봉 데이터 수집
python tools/collect.py --market KRW-BTC --minutes 1 --days 10
```
- 업비트 API를 통해 실시간/과거 데이터 수집
- 1분봉 OHLCV 데이터를 parquet/csv로 저장
- 200개씩 나누어 API 제한 회피

### 2. **build_dataset.py** - 데이터셋 생성
```python
# CLASSIC 모드 (기술지표 기반)
python tools/build_dataset.py `
  --mode classic --in data/raw/krw_btc_1m_10d.parquet `
  --out data/classic/dataset.parquet `
  --horizon 2 --up 0.001 --dn -0.001 --tfs "3,5"

# SEQ 모드 (시퀀스 패턴)
python tools/build_dataset.py `
  --mode seq --in data/raw/krw_btc_1m_10d.parquet `
  --out data/seq/dataset.parquet `
  --n_steps 20 --horizon 2 --up 0.001 --dn -0.001
```

**기술지표 계산 (종가 기준)**:
- **이동평균**: EMA(12,26), SMA(20,50) - 종가 기준
- **볼린저밴드**: 20일 SMA ± 2σ - 종가 기준
- **MACD**: EMA(12) - EMA(26) - 종가 기준
- **RSI**: Wilder's RSI(14) - 종가 변화량 기준
- **ATR**: Wilder's ATR(14) - 고/저/종가 기준
- **수익률/변동성**: 1,5,15,30,60분 - 종가 기준

**멀티 타임프레임 (MTF)**:
- 3분봉, 5분봉 지표를 1분봉에 병합
- 각 타임프레임별 MACD, RSI, 볼린저밴드 등

**라벨링**:
- UP: horizon분 후 수익률 > threshold
- DOWN: horizon분 후 수익률 < -threshold  
- FLAT: 그 외

### 3. **train.py** - 모델 학습
```python
python tools/train.py `
  --data data/classic/dataset.parquet `
  --model artifacts/train/model.joblib `
  --splits 4 --fee 0.0005 --allow_short
```

**학습 프로세스**:
1. 데이터 로드 및 전처리
2. 시계열 분할 (TimeSeriesSplit)
3. HistGradientBoostingClassifier 학습
4. 최적 임계값 탐색 (Sharpe Ratio 최대화)
5. 모델/스케일러/메타 저장

**최적화**:
- Walk-forward 검증
- 수수료 고려 임계값 최적화
- 클래스 불균형 처리

### 4. **backtest.py** - 백테스팅
```python
python tools/backtest.py `
  --data data/classic/dataset.parquet `
  --model artifacts/train/model.joblib `
  --fee 0.0005 --report artifacts/backtest/report.json
```

**거래 로직**:
1. **신호 발생**: 현재 봉의 피처로 예측
2. **거래 실행**: 다음 봉 시가에서 체결
3. **포지션 관리**: LONG/SHORT/FLAT
4. **플립 기능**: 반대 포지션으로 즉시 전환

**성과 지표**:
- 누적 수익률, 샤프 비율
- 최대 낙폭(MDD), 승률
- 거래 횟수, 평균 보유시간

### 5. **live_paper_v2.py** - 실시간 페이퍼 트레이딩
```python
python tools/live_paper_v2.py `
  --market KRW-BTC `
  --model artifacts/train/model.joblib `
  --fee 0.0005 --state artifacts/live/state.json
```

**실시간 처리**:
1. 매분 최신 데이터 수집 (1200분)
2. 워밍업 적용 (420분)
3. 기술지표 계산 및 예측
4. 가상 거래 실행 및 상태 관리

**지표 표시 (개선됨)**:
- 실시간 OHLCV 및 거래량
- 볼린저밴드 가격 (상단/중심/하단) - 원화 표시
- 이동평균선 (EMA12/26, SMA20/50) - 원화 표시
- 멀티 타임프레임 지표 (3분/5분봉)
- 수익률/변동성 (퍼센트 표시)
- 예측 확률 및 거래 결정

### 6. **배치 도구 (batch_*.py)**

**batch_data.py** - 다중 파라미터 데이터셋 생성
```python
python tools/batch_data.py --mode classic
# DAYS: [10, 181]
# HORIZON: [1,2,3,5,7,10]
# THRESHOLD: [0.0001~0.002]
# 출력: n20_h2_t0.001 처리 중... [OK]
```

**batch_train.py** - 일괄 모델 학습
```python
python tools/batch_train.py --mode classic --splits 4
# 출력: d10_h2_t0.001 처리 중... [OK]
```

**batch_back.py** - 일괄 백테스팅
```python
python tools/batch_back.py --mode classic --fee 0.0005
# 출력: d10_h2_t0.001 처리 중... [OK]
```

### 7. **분석 도구**

**analyze_backtest_results.py**
- 모든 백테스트 결과 종합 분석
- TOP/BOTTOM 15 조합 표시
- Horizon/Threshold별 평균 수익률
- 날짜 필터링 기능 (DAYS_TO_ANALYZE 변수)

**analyze_best_trade.py** (개선됨)
- 최고 수익 거래 상세 분석
- 진입/청산 시점 기술지표 상세 표시
  - 볼린저밴드 가격 (원화)
  - 이동평균선 가격 (원화)
  - 수익률/변동성 (퍼센트)
  - 멀티 타임프레임 지표
- 신호 발생과 실행 시점 명확히 구분
- 보유 기간 중 확률 변화 추적

## 💡 주요 특징

### 거래 신호와 실행 분리
- **신호**: 현재 봉 종료 시점 (종가 기준 지표)
- **실행**: 다음 봉 시작 시점 (시가 체결)
- 현실적인 거래 시뮬레이션

### 멀티 타임프레임 분석
- 1분봉 + 3분봉 + 5분봉 지표 통합
- 다양한 시간대의 추세 포착
- 노이즈 감소 및 신호 신뢰도 향상

### 플립 거래 지원
- LONG → SHORT 즉시 전환
- SHORT → LONG 즉시 전환
- 시장 변화에 빠른 대응

### 최적화된 임계값
- 데이터별 최적 threshold 자동 탐색
- 수수료 고려한 실질 수익률 최대화
- Walk-forward 검증으로 과적합 방지

## 📊 기술지표 계산 기준

모든 지표는 **종가(close)** 기준으로 계산됩니다:

| 지표 | 계산 방식 | 기준 | 코드 |
|------|----------|------|------|
| EMA(12,26) | 지수이동평균 | 종가 | `ema(df["close"], 12)` |
| SMA(20,50) | 단순이동평균 | 종가 | `df["close"].rolling(20).mean()` |
| 볼린저밴드 | SMA(20) ± 2σ | 종가 | `bollinger(df["close"], 20, 2.0)` |
| MACD | EMA(12) - EMA(26) | 종가 | `ema_12 - ema_26` |
| RSI(14) | Wilder's RSI | 종가 변화량 | `rsi(df["close"], 14)` |
| ATR(14) | Wilder's ATR | 고/저/종가 | `atr(high, low, close, 14)` |
| 수익률 | (현재-이전)/이전 | 종가 | `df["close"].pct_change()` |
| 변동성 | 수익률 표준편차 | 종가 | `returns.rolling(20).std()` |

## 🚀 빠른 시작

### 1. 전체 파이프라인 실행
```bash
python tools/one_click_pipeline.py --days 10 --splits 4
```

### 2. 개별 단계 실행
```bash
# 데이터 수집
python tools/collect.py --market KRW-BTC --days 10

# 데이터셋 생성
python tools/build_dataset.py --mode classic --horizon 2

# 모델 학습
python tools/train.py --data data/classic/dataset.parquet

# 백테스팅
python tools/backtest.py --model artifacts/train/model.joblib

# 실시간 거래
python tools/live_paper_v2.py --model artifacts/train/model.joblib
```

### 3. 분석
```bash
# 백테스트 결과 분석
python analyze_backtest_results.py

# 최고 거래 분석
python analyze_best_trade.py --trades artifacts/backtest/trades.csv
```

## 📝 설정 파일

**config.py** - 전역 설정
- TARGET_COINS: 거래 대상 (KRW-BTC)
- 파일 경로 및 디렉토리 구조
- 기본 파라미터 값

**배치 도구 설정 변수**
- `batch_data.py`: DAYS_TO_PROCESS, HORIZON_CANDIDATES, THRESHOLD_CANDIDATES
- `batch_train.py`: 자동으로 batch_data 설정 따름
- `batch_back.py`: 자동으로 batch_data 설정 따름
- `analyze_backtest_results.py`: DAYS_TO_ANALYZE (날짜 필터)

## 🔄 업데이트 기록

### 2025.08.18 최신 업데이트
- **코드 문서 자동 업데이트 규칙 추가**: 코드 수정 시 관련 문서 즉시 업데이트
- **개선된 출력 형식**:
  - `live_paper_v2.py`: 볼린저밴드/이동평균 원화 표시, 퍼센트 변환
  - `analyze_best_trade.py`: 진입/청산 상세 지표, 중복 제거
  - `batch_*.py`: 파라미터 정보 포함한 진행 상황 표시
- **날짜 필터링**: `analyze_backtest_results.py`에 DAYS_TO_ANALYZE 변수 추가
- **플립 거래**: 모든 도구에서 포지션 즉시 전환 지원

### 2025.08.17
- 플립 거래 기능 구현
- 멀티 타임프레임 지표 통합
- 배치 처리 도구 완성

## ⚙️ 최적화 기법

### 1. 메모리 효율성
- Parquet 형식 사용 (압축률 높음)
- 청크 단위 처리
- 불필요한 컬럼 제거

### 2. 처리 속도
- 벡터화 연산 활용
- NumPy 배열 최적화
- 병렬 처리 지원

### 3. 안정성
- 오류 처리 및 재시도
- 로깅 및 모니터링
- 데이터 검증

## 🛠️ 문제 해결

### 일반적인 문제
1. **API 제한**: Rate limit 도달 시 자동 대기
2. **네트워크 오류**: 자동 재시도 메커니즘
3. **데이터 누락**: 결정론적 스테핑으로 방지

### 디버깅 명령어
```powershell
# 수집 로그
python tools/collect.py --verbose

# 학습 로그
python tools/train.py --debug

# 백테스팅 로그
python tools/backtest.py --verbose
```

---
*최종 업데이트: 2025.08.18*