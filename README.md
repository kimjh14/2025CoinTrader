# 2025CoinTrader - 암호화폐 자동매매 시스템

## 프로젝트 개요

두 가지 AI 접근 방식을 제공하는 암호화폐 트레이딩 시스템입니다.

### V1_Based_HKM: 딥러닝 시스템
- **모델**: 다중 타임프레임 Bi-LSTM + Attention
- **지원**: 다중 암호화폐 (BTC, ETH, ADA, SOL 등)
- **특징**: 40+ 기술적 지표, GPU 가속

### V2_Based_with_CHATGPT: 머신러닝 시스템  
- **모델**: HistGradientBoostingClassifier + 확률 보정
- **지원**: KRW-BTC 특화
- **특징**: 멀티 타임프레임, 페이퍼 트레이딩

## 시스템 구조

```
2025CoinTrader/
├── V1_Based_HKM/                    # 딥러닝 시스템
│   ├── tools/                      # 실행 스크립트들
│   ├── ml/                         # 딥러닝 모듈
│   ├── config/                     # 설정 파일
│   └── artifacts/                  # 모델 저장소
│
├── V2_Based_with_CHATGPT/          # 머신러닝 시스템
│   ├── tools/                      # 실행 스크립트들
│   ├── ml/                         # ML 모듈
│   ├── data/                       # 데이터 저장소
│   └── artifacts/                  # 결과 저장소
│
├── requirements.txt                # 통합 의존성
├── CLAUDE.md                       # 개발 가이드
└── README.md                       # 이 파일
```

## 빠른 시작

### 공통 설정
```bash
git clone <repository_url>
cd 2025CoinTrader
pip install -r requirements.txt
```

### V1 시스템 (딥러닝)
```bash
cd V1_Based_HKM
python tools/test_setup.py
python tools/run_data_collection.py
python tools/train_model.py
python tools/backtest_btc.py
```

### V2 시스템 (머신러닝)
```bash
cd V2_Based_with_CHATGPT
python tools/collect.py --market KRW-BTC --days 180
python tools/build_dataset.py --mode classic
python tools/train.py --allow_short
python tools/backtest.py
```

## 시스템 비교

| 특징 | V1 | V2 |
|------|----|----|
| **모델** | Bi-LSTM | HistGradientBoosting |
| **코인 지원** | 다중 | KRW-BTC |
| **하드웨어** | GPU 권장 | CPU 충분 |
| **학습 시간** | 시간 단위 | 분 단위 |
| **특화 기능** | 다중 타임프레임 | 페이퍼 트레이딩 |

## API 설정

업비트 API 키 설정:
```python
# V1: config/data_config.py
# V2: tools/config.py
UPBIT_ACCESS_KEY = "your_access_key"
UPBIT_SECRET_KEY = "your_secret_key"
```

## 시스템 요구사항

- **Python**: 3.8 이상
- **V1**: RAM 16GB, NVIDIA GPU 권장
- **V2**: RAM 8GB, CPU만으로 가능

## 투자 위험 경고

1. 암호화폐 투자는 높은 위험을 수반합니다
2. 과거 성과가 미래 수익을 보장하지 않습니다
3. 반드시 백테스팅 후 소액으로 시작하세요
4. 본 시스템은 교육 목적으로만 제공됩니다

## 선택 가이드

**V1을 선택하는 경우:**
- 다양한 암호화폐 거래
- GPU 리소스 보유
- 복합적 패턴 학습 원함

**V2를 선택하는 경우:**
- KRW-BTC 집중 거래
- 빠른 실행 속도 필요
- CPU 환경 사용
- 페이퍼 트레이딩 선호

## 라이선스

MIT 라이선스 하에 배포됩니다. 교육 및 연구 목적으로만 사용하세요.