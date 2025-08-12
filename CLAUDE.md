# CLAUDE.md - 개발 가이드라인

## 언어 사용 지침

**모든 작업은 한국어로 수행**
- 모든 대화와 설명은 한국어
- 코드 주석은 한국어
- 오류 메시지와 로그도 한국어
- 사용자와의 모든 상호작용은 한국어 우선

# Claude Code 동작 규칙

## 윈도우 환경 규칙 (필수)
- **모든 경로는 윈도우 형식 사용**: `D:\python\2025CoinTrader` (슬래시 아님)
- **PowerShell 줄바꿈은 백틱(`) 사용**: `\` 아님
- **파일 경로 구분자는 백슬래시**: `tools\backtest.py`
- **CLI 예시는 PowerShell 기준으로 작성**

### 올바른 PowerShell 명령어 예시
```powershell
python tools/backtest.py `
  --data data/classic/dataset.parquet `
  --model artifacts/model.joblib `
  --fee 0.0005
```

### 잘못된 예시 (사용 금지)
```bash
python tools/backtest.py \
  --data data/classic/dataset.parquet \
  --model artifacts/model.joblib \
  --fee 0.0005
```

## 코드 실행 규칙 (최우선)
- **중요**: Claude Code는 코드 분석, 수정, 최적화만 수행
- **모든 실행은 사용자가 직접 수행**: 특별히 해달라고 요청하지 않는 경우
- Claude는 명령어 제안만 하고 직접 실행하지 않음
- 코드 변경 후 테스트도 사용자가 직접 수행
- 데이터셋 생성, 모델 학습, 백테스팅 등 모든 실행 작업은 사용자 책임

## 행동 전 계획 확인 규칙 (필수)
- **필수**: 어떤 행동을 하기 전에 반드시 계획을 먼저 제시
- **사용자 승인**: 계획을 설명하고 진행해도 될지 사용자에게 확인 요청
- **승인 후 진행**: 사용자가 승인한 후에만 실제 작업 수행
- **예외 없음**: 파일 읽기, 코드 수정, 분석 등 모든 작업에 적용
- 긴급하거나 간단한 작업도 예외 없이 계획 수립 후 승인 요청

## 코드 스타일 및 출력 규칙
- 모든 주석과 문서는 한국어로 작성
- 함수명과 변수명은 영어, 설명은 한국어
- 로깅과 에러 메시지는 한국어 우선
- **이모지를 출력에 넣지 말것**
- **최대한 필요한 내용만 프린트할 것**
- 불필요한 설명이나 부연설명 최소화

## 개발 가이드라인

### 테스트 (사용자 직접 실행)
```bash
# V1 시스템 테스트
cd V1_Based_HKM && python tools/test_setup.py

# V2 시스템 테스트 (설정 확인)
cd V2_Based_with_CHATGPT && python -c "from tools.config import Config; print(Config.TARGET_COINS)"
```

### 성능 모니터링 (사용자 직접 실행)
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