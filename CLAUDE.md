# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소에서 작업할 때 참고할 가이드라인을 제공합니다.

## 🇰🇷 언어 사용 지침

**중요: 이 저장소에서 작업할 때는 반드시 한국어로 모든 응답과 설명을 제공해야 합니다.**

- 모든 대화와 설명은 한국어로 진행
- 코드 주석은 한국어로 작성
- 오류 메시지와 로그도 가능한 한 한국어로 제공
- 기술 문서와 README는 한국어로 작성
- 사용자와의 모든 상호작용은 한국어 우선

## 프로젝트 개요

이 저장소는 **암호화폐 자동매매 시스템**을 위한 두 가지 접근 방식을 포함합니다:

### V1_Based_HKM: 딥러닝 기반 시스템
- **다중 타임프레임 Bi-LSTM 모델** 사용
- 다양한 암호화폐 지원 (BTC, ETH, ADA, SOL 등)
- 복합 기술적 지표 분석 (40+ 지표)
- 백테스팅 및 실시간 거래 시스템

### V2_Based_with_CHATGPT: 머신러닝 기반 시스템
- **HistGradientBoostingClassifier** 모델 사용
- KRW-BTC 특화 거래 시스템
- Triple Barrier 라벨링 방식
- 페이퍼 트레이딩 및 실시간 모니터링

## 저장소 구조

```
2025CoinTrader/
├── V1_Based_HKM/                    # 딥러닝 기반 시스템
│   ├── tools/                      # 실행 도구들
│   ├── ml/                         # ML 라이브러리
│   ├── config/                     # 설정 파일들
│   ├── artifacts/                  # 학습된 모델들
│   ├── old/                        # 이전 버전들
│   └── docs/                       # 문서 및 분석자료
│
├── V2_Based_with_CHATGPT/          # 머신러닝 기반 시스템
│   ├── tools/                      # 실행 도구들
│   ├── ml/                         # ML 라이브러리
│   ├── data/                       # 데이터 파일들
│   └── artifacts/                  # 결과물들
│
├── CLAUDE.md                       # 이 파일
└── README.md                       # 프로젝트 개요
```

# V1_Based_HKM 시스템

## 핵심 특징
- **AI 모델**: 다중 타임프레임 Bi-LSTM (양방향 LSTM 2층)
- **지원 코인**: BTC, ETH, SOL, XRP, ADA, DOGE 등
- **타임프레임**: 1분~일봉 다중 분석
- **기술적 지표**: 40+ 지표 (MACD, RSI, 볼린저밴드 등)

## 주요 명령어

### 환경 설정
```bash
cd V1_Based_HKM
pip install -r requirements.txt
python tools/test_setup.py
```

### 데이터 수집 및 모델 훈련
```bash
# 1. 데이터 수집
python tools/run_data_collection.py

# 2. 모델 훈련
python tools/train_model.py

# 3. 학습 모니터링
python tools/monitor_training.py
```

### 백테스팅 실행
```bash
# 개별 코인 백테스팅
python tools/backtest_btc.py
python tools/backtest_ada.py

# 통합 백테스팅
python tools/integrated_backtester.py
```

### 실시간 거래
```bash
# 실시간 거래 시스템
python tools/live_trading.py

# 실시간 예측
python tools/live_predictor.py
```

## V1 아키텍처
- **모델**: Bi-LSTM + Attention (Multi-head, 4 heads)
- **출력**: 3-클래스 분류 (상승/하락/보합)
- **성능**: 정확도 65-75%, F1 Score 0.6-0.7

# V2_Based_with_CHATGPT 시스템

## 핵심 특징
- **AI 모델**: HistGradientBoostingClassifier
- **타겟**: KRW-BTC 특화
- **라벨링**: Triple Barrier 방식
- **실시간**: 페이퍼 트레이딩 지원

## 주요 명령어

### 환경 설정
```bash
cd V2_Based_with_CHATGPT
pip install -r requirements.txt
```

### 완전한 파이프라인 실행
```bash
# 1. 데이터 수집 (30일치 1분봉)
python tools/collect.py --market KRW-BTC --minutes 1 --days 30 --out data/krw_btc_1m.parquet

# 2. 피처 & 라벨 생성
python tools/build_dataset.py --in data/krw_btc_1m.parquet --out data/dataset.parquet

# 3. 모델 학습
python tools/train.py --data data/dataset.parquet --model artifacts/model.joblib --scaler artifacts/scaler.joblib --meta artifacts/meta.json

# 4. 백테스팅
python tools/backtest.py --data data/dataset.parquet --model artifacts/model.joblib --scaler artifacts/scaler.joblib --report artifacts/backtest_report.json

# 5. 실시간 페이퍼 트레이딩
python tools/live_paper.py --market KRW-BTC --model artifacts/model.joblib --scaler artifacts/scaler.joblib
```

## V2 아키텍처
- **모델**: HistGradientBoostingClassifier + CalibratedClassifierCV
- **피처**: EMA, MACD, RSI, Bollinger Bands, ATR, VWAP, ADX
- **라벨링**: Triple Barrier (상한/하한/시간) 기반 3-class 분류

# 공통 설정 및 주의사항

## API 설정
두 시스템 모두 업비트 API 키가 필요합니다:
```python
# config/ 폴더의 설정 파일에서
UPBIT_ACCESS_KEY = "your_access_key"
UPBIT_SECRET_KEY = "your_secret_key"
```

## 하드웨어 요구사항
- **V1**: GPU 권장 (NVIDIA CUDA 11.8+), RAM 16GB 권장
- **V2**: CPU만으로도 가능, RAM 8GB 이상

## 투자 위험 경고
1. **높은 위험**: 암호화폐 투자는 높은 위험을 수반하며 손실이 발생할 수 있습니다
2. **백테스팅 한계**: 과거 성과가 미래 수익을 보장하지 않습니다
3. **실전 거래**: 슬리피지 및 수수료를 고려해야 합니다
4. **API 제한**: 업비트 API 호출 제한을 준수해야 합니다

## 개발 가이드라인

### 코드 스타일
- 모든 주석과 문서는 한국어로 작성
- 함수명과 변수명은 영어, 설명은 한국어
- 로깅과 에러 메시지는 한국어 우선

### 테스트
```bash
# V1 시스템 테스트
cd V1_Based_HKM && python tools/test_setup.py

# V2 시스템 테스트 (설정 확인)
cd V2_Based_with_CHATGPT && python -c "from tools.config import Config; print(Config.TARGET_COINS)"
```

### 성능 모니터링
- V1: `tools/monitor_training.py`로 학습 과정 모니터링
- V2: `tools/backtest.py`로 백테스팅 성능 확인

## 버전 선택 가이드

### V1_Based_HKM을 선택하는 경우:
- 다양한 암호화폐를 거래하고 싶은 경우
- 딥러닝 모델의 복잡한 패턴 학습을 원하는 경우
- GPU 리소스가 충분한 경우
- 다중 타임프레임 분석이 필요한 경우

### V2_Based_with_CHATGPT를 선택하는 경우:
- KRW-BTC만 집중적으로 거래하고 싶은 경우
- 빠른 학습과 실행 속도가 필요한 경우
- CPU 환경에서 실행해야 하는 경우
- 페이퍼 트레이딩부터 시작하고 싶은 경우

두 시스템 모두 완전한 백테스팅과 실시간 거래 기능을 제공하므로, 요구사항에 맞는 시스템을 선택하여 사용하시기 바랍니다.