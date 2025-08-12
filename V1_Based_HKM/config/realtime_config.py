"""
실시간 테스터용 간단한 설정 파일
"""

class Config:
    """설정 클래스"""
    
    def __init__(self):
        # 업비트 API 키 (실시간 수집은 읽기 전용 API만 사용)
        self.UPBIT_ACCESS_KEY = ""  # 읽기 전용이므로 키 없어도 됨
        self.UPBIT_SECRET_KEY = ""  # 읽기 전용이므로 키 없어도 됨
        
        # 기본 설정
        self.SERVER_URL = "https://api.upbit.com"
        
        # 타임아웃 설정
        self.REQUEST_TIMEOUT = 10
        
        # 재시도 설정
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 1