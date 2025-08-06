# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

이 저장소는 딥러닝 기반 암호화폐 자동매매 시스템입니다. 다중 타임프레임 Bi-LSTM 모델을 사용하여 암호화폐 가격을 예측하고 백테스팅을 통해 전략을 검증하는 완전한 파이프라인을 제공합니다.

## 개발 환경 설정

### 필수 패키지 설치
```bash
# 자동 설치 스크립트 실행
python install_requirements.py

# 또는 수동 설치 (requirements.txt가 있는 경우)
pip install pandas>=1.5.0 numpy>=1.21.0 matplotlib>=3.5.0 seaborn>=0.11.0
pip install torch torchvision torchaudio scikit-learn>=1.1.0 plotly>=5.0.0
```

### GPU 환경 설정 (권장)
```bash
# PyTorch CUDA 버전 설치 (GPU 가속용)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 설치 검증
```bash
python test_installation.py
```

## 핵심 시스템 구조

### 1. Data_maker/ - 데이터 수집 및 전처리
- **data_collector.py**: 업비트 API를 통한 다중 타임프레임 데이터 수집
- **data_analyzer.py**: 40+ 기술적 지표 계산 및 데이터 품질 분석
- **run_data_collection.py**: 인터랙티브 데이터 수집 실행 스크립트
- **config.py**: API 키 및 거래 설정 관리

### 2. Model_maker/ - AI 모델 개발
- **Training_v04.py**: 다중 타임프레임 Bi-LSTM 모델 훈련
- **monitor_v04_training.py**: 학습 과정 실시간 모니터링
- **analyze_trading_signals.py**: 모델 예측 결과 분석

### 3. Upbit_Trader/ - 백테스팅 및 전략 검증
- **backtest_ada_optimized.py**: ADA 코인 최적화 백테스팅
- **backtest_btc_optimized.py**: BTC 코인 백테스팅
- **analyze_backtest_results.py**: 백테스팅 결과 분석

## 주요 개발 워크플로우

### 1. 데이터 수집
```bash
# 인터랙티브 모드
python Data_maker/run_data_collection.py

# 명령줄 인자 모드  
python Data_maker/run_data_collection.py -c BTC,ETH -s 2025-01-01 -e 2025-01-31
```

### 2. 모델 훈련
```bash
# 메인 모델 훈련
python Model_maker/Training_v04.py

# 학습 진행 모니터링
python Model_maker/monitor_v04_training.py
```

### 3. 백테스팅 실행
```bash
# 개별 코인 백테스팅
python Upbit_Trader/backtest_ada_optimized.py
python Upbit_Trader/backtest_btc_optimized.py

# 통합 백테스팅
python integrated_crypto_backtester.py
```

### 4. 결과 분석
```bash
# 모델 예측 분석
python Model_maker/analyze_trading_signals.py

# 백테스팅 결과 분석
python Upbit_Trader/analyze_backtest_results.py
```

## 핵심 아키텍처

### AI 모델 구조
- **다중 타임프레임**: 1분봉 + 5분봉 MACD 데이터 통합
- **Bi-LSTM**: 양방향 LSTM 2층 (각 128 hidden units)
- **Attention**: Multi-head attention (4 heads)
- **출력**: 3-클래스 분류 (상승/하락/보합)

### 데이터 파이프라인
1. 업비트 API → 다중 타임프레임 OHLCV 데이터
2. 기술적 지표 계산 (MACD, RSI, 볼린저밴드 등)
3. 시간 동기화 및 정규화
4. LSTM 훈련용 시퀀스 데이터 생성

### 백테스팅 시스템
- 과거 데이터 기반 전략 성능 검증
- 리스크 관리 (손절, 익절, 포지션 사이징)
- 성과 지표 계산 (샤프 비율, 최대 낙폭, 승률)

## 설정 및 환경변수

### API 키 설정
`Data_maker/config.py` 파일에서 업비트 API 키 설정:
```python
UPBIT_ACCESS_KEY = "your_access_key"
UPBIT_SECRET_KEY = "your_secret_key"
```

### GPU 사용 확인
```bash
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
```

## 생성되는 주요 파일들

### 모델 파일
- `multitimeframe_bilstm_crypto_model_v04.pth`: 최종 훈련 모델
- `multitimeframe_preprocessor_v04.pkl`: 데이터 전처리기
- `model_info_v04.pkl`: 모델 메타데이터

### 데이터 파일
- `{코인명}_training_dataset.csv`: 훈련용 데이터셋
- `data_quality_analysis.json`: 데이터 품질 분석 결과

### 분석 결과
- `{코인명}_price_chart.html`: 인터랙티브 가격 차트
- `backtest_results.png`: 백테스팅 결과 시각화
- `multitimeframe_confusion_matrix_v04.png`: 모델 성능 평가

## 성능 및 최적화

### 하드웨어 요구사항
- RAM: 최소 8GB, 권장 16GB
- GPU: NVIDIA GPU (CUDA 11.8+ 지원) 권장
- 저장공간: 최소 2GB

### 모델 성능 지표
- 정확도: 65-75% (검증 데이터 기준)
- F1 Score: 0.6-0.7 (Macro 평균)
- 예측 지연시간: < 100ms

### 백테스팅 성과 (ADA 기준)
- 연간 수익률: 15-25%
- 샤프 비율: 1.2-1.8
- 최대 낙폭: -15% 이하

## 주의사항

1. **투자 위험**: 암호화폐 투자는 높은 위험을 수반하며 손실이 발생할 수 있습니다
2. **API 제한**: 업비트 API 호출 제한을 준수해야 합니다
3. **백테스팅**: 과거 성과가 미래 수익을 보장하지 않습니다
4. **실전 거래**: 슬리피지 및 수수료를 고려해야 합니다

## 테스트 및 검증

### 시스템 테스트
```bash
# 전체 시스템 테스트
python test_installation.py

# API 연결 테스트
python test_upbit_api.py

# 배치 결과 분석
python analyze_batch_results.py
```

### 데이터 품질 확인
모든 데이터 수집 후 `data_quality_analysis.json` 파일을 확인하여 데이터 품질을 검증하세요.

## 지원 코인
- 주요 코인: BTC, ETH, SOL, XRP, ADA, DOGE
- 훈련 코인: APT, BTC, DOGE, HBAR, SOL, XRP
- 검증 코인: SUI (Out-of-sample 테스트)

## 타임프레임
- 1분봉, 3분봉, 5분봉, 15분봉, 1시간봉, 4시간봉, 일봉 지원
- 다중 타임프레임 통합 분석 가능