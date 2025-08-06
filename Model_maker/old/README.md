# 암호화폐 가격 예측 모델 (Multi-Timeframe Bi-LSTM V04)

## 개요
이 프로젝트는 다중 시간대 Bi-LSTM (Bidirectional Long Short-Term Memory) 딥러닝 모델을 사용하여 암호화폐 가격 변동을 예측하는 시스템입니다. 1분봉과 5분봉 데이터의 MACD 지표를 통합하여 단기 가격 움직임을 예측합니다.

## 주요 특징
- **다중 시간대 분석**: 1분봉 + 5분봉 MACD 지표 통합
- **Bi-LSTM 아키텍처**: 양방향 LSTM과 Attention 메커니즘 적용
- **3-클래스 분류**: 하락(< -0.05%), 보류(-0.05% ~ 0.05%), 상승(> 0.05%)
- **GPU 최적화**: CUDA 지원 및 멀티 GPU 병렬 처리
- **실시간 예측**: 1분 후 가격 변동 예측

## 프로젝트 구조
```
Model_maker/
├── Training_v04.py           # 메인 학습 코드
├── monitor_v04_training.py   # 학습 진행 모니터링 도구
├── analyze_trading_signals.py # 거래 신호 분석 도구
├── data/raw/                 # 암호화폐 OHLCV 데이터
│   ├── {COIN}_1m.csv        # 1분봉 데이터
│   ├── {COIN}_5m.csv        # 5분봉 데이터
│   └── ...                  # 기타 시간대 데이터
├── 설명파일들/              # 분석 결과 및 문서
│   ├── *.png                # 시각화 결과
│   └── *.md                 # 전략 및 개선사항 문서
└── 이전모델생성기/          # 이전 버전 모델 코드
```

## 기술 스택
- **Python 3.8+**
- **PyTorch**: 딥러닝 프레임워크
- **pandas & numpy**: 데이터 처리
- **scikit-learn**: 전처리 및 평가
- **matplotlib & seaborn**: 시각화

## 모델 아키텍처

### MultiTimeframePreprocessor
- 1분봉과 5분봉 데이터를 시간 정렬 (interpolation)
- MACD 지표 계산 및 정규화
- 다이버전스 특징 추출
- 시퀀스 데이터 생성 (길이: 10)

### MultiTimeframeBiLSTM
- **입력층**: 10개 특징 (1분/5분 MACD, 다이버전스, 가격 변화율)
- **Bi-LSTM층**: 2층, 128 hidden units
- **Attention층**: Self-attention + Multi-head attention
- **출력층**: 3-클래스 분류 (Softmax)

## 사용법

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 모델 학습
```bash
python Training_v04.py
```

### 3. 학습 모니터링
```bash
python monitor_v04_training.py
```

### 4. 결과 분석
```bash
python analyze_trading_signals.py
```

## 학습 데이터
- **학습 코인**: APT, BTC, DOGE, HBAR, SOL, XRP
- **검증 코인**: SUI
- **기간**: 2025-05-03 ~ 2025-07-30
- **총 데이터**: 6개 코인 × 여러 시간대

## 성능 지표
- **정확도 (Accuracy)**: 검증 데이터 기준
- **F1 Score**: Macro 평균
- **거래 신호 비율**: 실제 거래 가능한 신호의 비율
- **혼동 행렬**: 클래스별 예측 성능

## 출력 파일
- `multitimeframe_bilstm_crypto_model_v04.pth`: 최종 학습된 모델
- `best_multitimeframe_model_v04.pth`: 최고 성능 모델 (F1 기준)
- `multitimeframe_preprocessor_v04.pkl`: 전처리기 객체
- `model_info_v04.pkl`: 모델 메타데이터
- `multitimeframe_training_history_v04.png`: 학습 히스토리 그래프
- `multitimeframe_confusion_matrix_v04.png`: 혼동 행렬 시각화

## 주의사항
- GPU 메모리가 충분한지 확인 (최소 4GB 권장)
- 데이터 파일이 `data/raw/` 디렉토리에 있어야 함
- 학습 시간은 GPU 성능에 따라 30분~2시간 소요

## 개선 계획
- 더 많은 기술 지표 추가
- 장기 시간대 (15분, 1시간) 통합
- 강화학습 기반 거래 전략 구현
- 실시간 거래 시스템 연동

## 라이선스
이 프로젝트는 연구 및 교육 목적으로만 사용됩니다.