"""
config.py - 시스템 설정 관리 모듈

CLI에서 직접 실행하지 않는 설정 모듈입니다.
다른 도구에서 다음과 같이 import하여 사용:

from config import Config
config = Config()
api_key = config.UPBIT_ACCESS_KEY
target_coins = config.TARGET_COINS

⚠️ 보안 주의사항:
• API 키를 환경변수(.env)로 관리하는 것을 권장
• 실제 거래 시 데모 키가 아닌 실제 키로 교체 필요
• Git 커밋 시 API 키 노출 주의
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    # 업비트 API 설정
    UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', 'TNlLWAApD8YPABJhPSMyExnVWRfcS0Y1HJXQOqx5')
    UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', 'vdce6rZ6niWxa7UxmCoVfxyiSomghHBweD4mD0TO')
    
    # 텔레그램 설정
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # Discord 설정
    DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
    
    # 데이터베이스 설정
    DATABASE_PATH = 'trading_data.db'
    
    # 거래 설정
    MAX_POSITIONS = 3
    MAX_INVESTMENT_RATIO = 0.1  # 10%
    TRAILING_STOP_RATIO = 0.03  # 3%
    
    # 타임프레임 가중치
    TIMEFRAME_WEIGHTS = {
        '5m': 0.4,
        '3m': 0.35,
        '1m': 0.25
    }
    
    # 분석 대상 코인 (거래량 상위 10개)
    TARGET_COINS = [
        'KRW-OMNI', 'KRW-ETH', 'KRW-BTC', 'KRW-HYPER', 'KRW-CKB',
        'KRW-USDT', 'KRW-SOL', 'KRW-ENA', 'KRW-SUI', 'KRW-KNC'
    ]
    
    # 기술적 지표 설정
    MACD_FAST = 5
    MACD_SLOW = 35
    MACD_SIGNAL = 5
    RSI_PERIOD = 14
    EMA_PERIODS = [5, 10, 20, 50]
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    STOCHASTIC_K = 14
    STOCHASTIC_D = 3
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    
    # 매매 신호 임계값
    SIGNAL_THRESHOLD = 7.0  # 7/10 이상
    ADX_TREND_THRESHOLD = 25
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    STOCHASTIC_OVERBOUGHT = 80
    
    # 백테스팅 설정
    BACKTEST_DAYS = 30
    
    # 웹 인터페이스 설정
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5002  # main.py와 다른 포트 사용
    FLASK_DEBUG = True 