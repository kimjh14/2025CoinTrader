# 🚀 2025CoinTrader - 통합 암호화폐 자동매매 시스템

## 📋 프로젝트 개요

**2025CoinTrader**는 두 가지 접근 방식을 제공하는 종합적인 암호화폐 트레이딩 시스템입니다:

### 🧠 V1_Based_HKM: 딥러닝 시스템
- **AI 모델**: 다중 타임프레임 Bi-LSTM + Attention
- **지원 코인**: BTC, ETH, ADA, SOL, XRP, DOGE 등 다양한 암호화폐
- **특징**: 40+ 기술적 지표, 복합 시간대 분석, GPU 가속

### ⚡ V2_Based_with_CHATGPT: 머신러닝 시스템  
- **AI 모델**: HistGradientBoostingClassifier
- **타겟**: KRW-BTC 특화 거래
- **특징**: Triple Barrier 라벨링, 페이퍼 트레이딩, 빠른 실행

## 🏗️ 시스템 구조

```
2025CoinTrader/
├── V1_Based_HKM/                    # 딥러닝 기반 시스템
│   ├── tools/                      # 실행 도구들
│   │   ├── collect_data.py         # 데이터 수집
│   │   ├── train_model.py          # 모델 훈련
│   │   ├── backtest_btc.py         # 백테스팅
│   │   └── live_trading.py         # 실시간 거래
│   ├── ml/                         # ML 모듈
│   │   ├── indicators.py           # 기술적 지표
│   │   ├── models.py               # LSTM 모델
│   │   └── preprocessing.py        # 전처리
│   ├── config/                     # 설정 파일
│   ├── artifacts/                  # 학습된 모델
│   └── requirements.txt
│
├── V2_Based_with_CHATGPT/          # 머신러닝 기반 시스템
│   ├── tools/                      # 실행 도구들
│   │   ├── collect.py              # 데이터 수집
│   │   ├── train.py                # 모델 훈련
│   │   ├── backtest.py             # 백테스팅
│   │   └── live_paper.py           # 페이퍼 트레이딩
│   ├── ml/                         # ML 모듈
│   │   ├── indicators.py           # 기술적 지표
│   │   ├── labeling.py             # 라벨링
│   │   └── utils.py                # 유틸리티
│   ├── data/                       # 데이터 파일
│   ├── artifacts/                  # 결과물
│   └── requirements.txt
│
├── requirements.txt                # 통합 의존성
├── CLAUDE.md                       # 개발 가이드
└── README.md                       # 이 파일
```

## 🚀 빠른 시작

### 공통 설정
```bash
# 저장소 클론
git clone <repository_url>
cd 2025CoinTrader

# 통합 의존성 설치
pip install -r requirements.txt
```

### V1 딥러닝 시스템 사용
```bash
cd V1_Based_HKM

# 1. 설치 확인
python tools/test_setup.py

# 2. 데이터 수집
python tools/run_data_collection.py

# 3. 모델 훈련
python tools/train_model.py

# 4. 백테스팅
python tools/backtest_btc.py

# 5. 실시간 거래
python tools/live_trading.py
```

### V2 머신러닝 시스템 사용
```bash
cd V2_Based_with_CHATGPT

# 1. 데이터 수집 (30일치)
python tools/collect.py --market KRW-BTC --minutes 1 --days 30 --out data/krw_btc_1m.parquet

# 2. 피처 생성
python tools/build_dataset.py --in data/krw_btc_1m.parquet --out data/dataset.parquet

# 3. 모델 훈련
python tools/train.py --data data/dataset.parquet --model artifacts/model.joblib --scaler artifacts/scaler.joblib --meta artifacts/meta.json

# 4. 백테스팅
python tools/backtest.py --data data/dataset.parquet --model artifacts/model.joblib --scaler artifacts/scaler.joblib --report artifacts/backtest_report.json

# 5. 페이퍼 트레이딩
python tools/live_paper.py --market KRW-BTC --model artifacts/model.joblib --scaler artifacts/scaler.joblib
```

## 🎯 시스템 비교

| 특징 | V1_Based_HKM | V2_Based_with_CHATGPT |
|------|-------------|----------------------|
| **모델** | Bi-LSTM + Attention | HistGradientBoostingClassifier |
| **지원 코인** | 다중 암호화폐 | KRW-BTC 특화 |
| **타임프레임** | 1분~일봉 다중 | 1분봉 중심 |
| **GPU 요구** | NVIDIA GPU 권장 | CPU만으로 충분 |
| **학습 시간** | 길음 (시간 단위) | 빠름 (분 단위) |
| **정확도** | 65-75% | 비슷한 수준 |
| **특화 기능** | 다중 시간대 분석 | 페이퍼 트레이딩 |

## ⚙️ 시스템 요구사항

### 하드웨어
- **V1**: RAM 16GB, NVIDIA GPU (CUDA 11.8+) 권장
- **V2**: RAM 8GB, CPU만으로 가능

### 소프트웨어
- **Python**: 3.8 이상
- **운영체제**: Windows 10/11, Linux, macOS

## 🔧 API 설정

두 시스템 모두 업비트 API 키가 필요합니다:

```python
# V1: config/data_config.py
# V2: tools/config.py
UPBIT_ACCESS_KEY = "your_access_key"
UPBIT_SECRET_KEY = "your_secret_key"
```

## 📊 성능 지표

### V1 딥러닝 시스템
- **정확도**: 65-75% (검증 데이터)
- **F1 Score**: 0.6-0.7 (Macro)
- **연간 수익률**: 15-25% (백테스팅)
- **샤프 비율**: 1.2-1.8
- **최대 낙폭**: -15% 이하

### V2 머신러닝 시스템
- **모델**: Calibrated Classifier로 확률 보정
- **라벨링**: Triple Barrier 방식
- **실행 속도**: < 100ms
- **페이퍼 트레이딩**: 실시간 의사결정

## 🛡️ 리스크 관리

### 공통 안전장치
- **백테스팅**: 실거래 전 반드시 과거 데이터로 검증
- **페이퍼 트레이딩**: V2는 실제 주문 없는 시뮬레이션 제공
- **손절 설정**: ATR 기반 동적 손절
- **포지션 사이징**: 리스크 기반 자동 계산

### 투자 주의사항
1. **높은 변동성**: 암호화폐는 극심한 가격 변동을 보입니다
2. **과최적화 위험**: 백테스팅 결과가 미래를 보장하지 않습니다
3. **API 제한**: 거래소 API 호출 제한을 준수해야 합니다
4. **기술적 위험**: 시스템 오류나 네트워크 문제가 발생할 수 있습니다

## 🔮 시스템 선택 가이드

### V1을 선택하는 경우:
- ✅ 다양한 암호화폐를 거래하고 싶음
- ✅ 딥러닝의 복잡한 패턴 학습을 원함
- ✅ GPU 리소스가 충분함
- ✅ 다중 타임프레임 분석이 필요함

### V2를 선택하는 경우:
- ✅ KRW-BTC만 집중 거래하고 싶음
- ✅ 빠른 학습과 실행이 필요함
- ✅ CPU 환경에서 사용해야 함
- ✅ 페이퍼 트레이딩부터 시작하고 싶음

## 📚 문서 및 지원

- **개발 가이드**: [CLAUDE.md](./CLAUDE.md) - 상세한 개발 지침
- **API 문서**: 각 시스템의 tools/ 폴더 내 스크립트 참조
- **이슈 리포팅**: GitHub Issues 활용

## ⚠️ 법적 고지

본 시스템은 **교육 및 연구 목적**으로만 제공됩니다:

1. **투자 책임**: 모든 투자 결정과 손실에 대한 책임은 사용자에게 있습니다
2. **금융 조언 아님**: 본 소프트웨어는 금융 조언을 제공하지 않습니다
3. **상업적 사용 금지**: 상업적 목적의 사용은 금지됩니다
4. **라이선스**: MIT 라이선스 하에 배포됩니다

## 🏃‍♂️ 다음 단계

1. **시작하기**: 요구사항에 맞는 시스템 선택
2. **백테스팅**: 반드시 과거 데이터로 전략 검증
3. **페이퍼 트레이딩**: V2 시스템으로 실전 없는 거래 연습
4. **점진적 접근**: 소액으로 실거래 시작 (본인 책임)

---

**⚡ 2025CoinTrader - 두 가지 힘으로 더 스마트한 암호화폐 거래**