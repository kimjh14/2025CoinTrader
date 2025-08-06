"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📈 2025CoinTrader - 암호화폐 데이터 수집 및 기술적 지표 계산 모듈 (data_collector.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • 업비트 API를 통한 실시간 암호화폐 OHLCV 데이터 수집
    • 다중 타임프레임 데이터 동시 수집 (1분봉~일봉)
    • 40+ 기술적 지표 자동 계산 및 추가
    • LSTM 모델 훈련용 레이블 생성 (상승/하락/횡보)
    • 데이터 품질 검증 및 전처리

🔧 핵심 기능:
    1. 데이터 수집: 업비트 REST API를 통한 캔들 데이터 수집
    2. 기술적 지표: MACD, RSI, 볼린저밴드, 이동평균, ATR, 스토캐스틱 등
    3. 멀티프레임: 1분, 3분, 5분, 15분, 1시간, 4시간, 일봉 동시 처리
    4. 레이블링: 5분 후 가격 변동을 기준으로 3-클래스 분류 레이블 생성
    5. 데이터 융합: 다중 타임프레임 데이터를 시간 동기화하여 통합

📊 생성되는 데이터:
    • {코인명}_training_dataset.csv: LSTM 훈련용 통합 데이터셋
    • 기술적 지표: MACD, Signal, Histogram, RSI, BB상하단, MA5/10/20/60
    • 변동성 지표: ATR, 가격 변동성, 거래량 비율
    • 예측 레이블: 0(상승), 1(하락), 2(횡보)

📋 사용 방법:
    collector = CryptoDataCollector()
    data = collector.collect_data_for_coin('KRW-BTC', '2025-01-01', '2025-01-31')
    collector.save_training_data(data, 'BTC')

⚠️ 주의사항:
    • API 호출 제한을 준수하여 과도한 요청 방지
    • 네트워크 오류 발생 시 자동 재시도 로직 포함
    • 데이터 수집 중 진행상황을 로그로 확인 가능

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json

# config.py에서 API 키 가져오
from config import Config

class CryptoDataCollector:
    """가상화폐 데이터 수집 및 기술적 지표 계산 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = Config()
        self.setup_logging()
        
        # 분석 대상 코인들
        self.target_coins = ['KRW-BTC', 'KRW-DOGE', 'KRW-SOL', 'KRW-XRP', 'KRW-SUI', 'KRW-HBAR', 'KRW-APT']
        
        # 타임프레임 설정
        self.timeframes = {
            '1m': 1,
            '3m': 3, 
            '5m': 5,
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 'D'
        }
        
        # 데이터 저장 경로 설정 (환경 변수 확인)
        result_folder = os.environ.get('RESULT_FOLDER_PATH')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if result_folder and os.path.exists(result_folder):
            # 결과 폴더가 지정된 경우 해당 폴더 내에 DATA 폴더 생성
            self.data_dir = os.path.join(result_folder, 'DATA')
        else:
            # 기본 경로 사용
            self.data_dir = os.path.join(current_dir, 'DATA')
            
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        
        # 디렉토리 생성
        self.create_directories()
        
    def setup_logging(self):
        """로깅 설정"""
        # 환경 변수에서 결과 폴더 경로 확인
        result_folder = os.environ.get('RESULT_FOLDER_PATH')
        
        if result_folder and os.path.exists(result_folder):
            # 결과 폴더가 지정된 경우 해당 폴더에 로그 생성
            log_file_path = os.path.join(result_folder, 'data_collection.log')
        else:
            # 기본 경로에 로그 생성
            log_file_path = 'data_collection.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True  # 기존 로깅 설정 덮어쓰기
        )
        self.logger = logging.getLogger(__name__)
        
    def create_directories(self):
        """필요한 디렉토리 생성"""
        for directory in [self.data_dir, self.raw_dir, self.processed_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.info(f"디렉토리 생성: {directory}")
                
    def collect_candle_data(self, market: str, timeframe: str, count: int = 200, to_date: str = None) -> pd.DataFrame:
        """캔들 데이터 수집"""
        try:
            # 업비트 API 직접 호출 (to 파라미터 지원)
            if timeframe == "1" or timeframe == "3" or timeframe == "5" or timeframe == "10" or timeframe == "15" or timeframe == "30" or timeframe == "60" or timeframe == "240":
                target_url = f"https://api.upbit.com/v1/candles/minutes/{timeframe}"
            elif timeframe == "D":
                target_url = "https://api.upbit.com/v1/candles/days"
            elif timeframe == "W":
                target_url = "https://api.upbit.com/v1/candles/weeks"
            elif timeframe == "M":
                target_url = "https://api.upbit.com/v1/candles/months"
            else:
                raise Exception(f"잘못된 틱 종류: {timeframe}")
            
            # 쿼리 파라미터 설정
            params = {"market": market, "count": count}
            if to_date:
                # 한국시간을 UTC로 변환 (업비트 API는 UTC 기준)
                kst_datetime = datetime.strptime(to_date, '%Y-%m-%dT%H:%M:%S')
                utc_datetime = kst_datetime - timedelta(hours=9)
                params["to"] = utc_datetime.strftime('%Y-%m-%dT%H:%M:%S')
                # 함수에 to_date를 받을땐 한국시간으로 받고, upbit api는 UTC시간으로 받기 때문에 내부에서 9시간 빼줌
            
            # API 호출
            response = requests.get(target_url, params=params)
            response.raise_for_status()
            candle_data = response.json()
            
            if not candle_data:
                self.logger.warning(f"{market} {timeframe} 데이터가 없습니다.")
                return pd.DataFrame()
            
            # DataFrame으로 변환
            df = pd.DataFrame(candle_data)
            
            # 컬럼명 중복 확인 및 처리
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])
            
            # 컬럼명 정리
            df = df.rename(columns={
                'candle_date_time_kst': 'timestamp',
                'opening_price': 'open',
                'high_price': 'high', 
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume',
                'candle_acc_trade_price': 'trade_price_volume'
            })
            
            # 시간순 정렬 (최신이 위로)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # timestamp를 datetime으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"{market} {timeframe} 데이터 수집 실패: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        if df.empty:
            return df
            
        try:
            # MACD 계산
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI 계산
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드 계산
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # 이동평균선
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            
            # ATR (Average True Range) 계산
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr'] = true_range.rolling(window=14).mean()
            
            # 거래량 관련 지표
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 가격 변화율
            df['price_change'] = df['close'].pct_change()
            df['price_change_5'] = df['close'].pct_change(periods=5)
            
            # 변동성 지표
            df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            
            self.logger.info("기술적 지표 계산 완료")
            return df
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 실패: {str(e)}")
            return df
    
    def extract_micro_features(self, df: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
        """미세한 특징 추출 (초봉 데이터를 통계적 특징으로 변환)"""
        if df.empty or len(df) < window_size:
            return df
            
        try:
            # 윈도우별 통계적 특징 계산
            features = []
            
            for i in range(window_size, len(df)):
                window_data = df.iloc[i-window_size:i]
                
                feature_dict = {
                    'timestamp': df.iloc[i]['timestamp'],
                    'volatility': window_data['close'].std(),
                    'trend_strength': (window_data['close'].iloc[-1] - window_data['close'].iloc[0]) / window_data['close'].iloc[0],
                    'high_low_range': window_data['high'].max() - window_data['low'].min(),
                    'uptick_ratio': (window_data['close'].diff() > 0).sum() / window_size,
                    'volume_concentration': window_data['volume'].max() / window_data['volume'].mean(),
                    'price_momentum': window_data['close'].pct_change().mean(),
                    'volume_momentum': window_data['volume'].pct_change().mean()
                }
                features.append(feature_dict)
            
            micro_df = pd.DataFrame(features)
            self.logger.info(f"미세 특징 추출 완료: {len(micro_df)}개")
            return micro_df
            
        except Exception as e:
            self.logger.error(f"미세 특징 추출 실패: {str(e)}")
            return pd.DataFrame()
    
    def create_labels(self, df: pd.DataFrame, future_periods: int = 5, threshold: float = 0.002) -> pd.DataFrame:
        """예측 목표 레이블 생성 (3-class 분류)"""
        if df.empty:
            return df
            
        try:
            # 미래 수익률 계산
            df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1
            
            # 3-class 레이블 생성
            def create_label(return_val):
                if pd.isna(return_val):
                    return np.nan
                elif return_val > threshold:
                    return 0  # 상승 (Buy)
                elif return_val < -threshold:
                    return 1  # 하락 (Sell)
                else:
                    return 2  # 횡보 (Hold)
            
            df['label'] = df['future_return'].apply(create_label)
            
            # 레이블 분포 확인
            label_counts = df['label'].value_counts()
            self.logger.info(f"레이블 분포: {label_counts.to_dict()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"레이블 생성 실패: {str(e)}")
            return df
    
    def collect_all_data(self, days_back: int = 90, start_date=None, end_date=None):
        """모든 코인의 데이터 수집
        
        Args:
            days_back: 수집할 일수 (start_date가 없을 때 사용)
            start_date: 시작 날짜 (datetime 객체)
            end_date: 종료 날짜 (datetime 객체)
        """
        if start_date and end_date:
            days_back = (end_date - start_date).days + 1
            self.logger.info(f"전체 데이터 수집 시작... (수집 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}, {days_back}일)")
        else:
            self.logger.info(f"전체 데이터 수집 시작... (수집 기간: {days_back}일 = {days_back/30:.1f}개월)")
        
        for coin in self.target_coins:
            self.logger.info(f"=== {coin} 데이터 수집 ===")
            
            for timeframe_name, timeframe_value in self.timeframes.items():
                try:
                    # 데이터 수집 (3개월치 데이터 - API 제한 고려)
                    if timeframe_name == '1m':
                        # 1분봉: 3개월 = 90일 * 24시간 * 60분 = 129,600개
                        # API 제한으로 200개씩 여러 번 수집
                        count = 200
                    elif timeframe_name == '3m':
                        # 3분봉: 3개월 = 90일 * 24시간 * 20분 = 43,200개
                        count = 200
                    elif timeframe_name == '5m':
                        # 5분봉: 3개월 = 90일 * 24시간 * 12분 = 25,920개
                        count = 200
                    elif timeframe_name == '15m':
                        # 15분봉: 3개월 = 90일 * 24시간 * 4분 = 8,640개
                        count = 200
                    elif timeframe_name == '1h':
                        # 1시간봉: 3개월 = 90일 * 24시간 = 2,160개
                        count = 200
                    elif timeframe_name == '4h':
                        # 4시간봉: 3개월 = 90일 * 6시간 = 540개
                        count = 200
                    else:  # 1d
                        # 일봉: 3개월 = 90일
                        count = 90   # 90일 (3개월)
                    
                    # 3개월치 데이터를 200개씩 순차적으로 수집 (과거부터 현재까지)
                    all_data = []
                    current_date = None
                    total_collected = 0
                    
                    # 각 타임프레임별로 필요한 총 데이터 수 계산
                    if timeframe_name == '1m':
                        total_needed = days_back * 24 * 60  # 일수 * 24시간 * 60분
                    elif timeframe_name == '3m':
                        total_needed = days_back * 24 * 20  # 일수 * 24시간 * 20분
                    elif timeframe_name == '5m':
                        total_needed = days_back * 24 * 12  # 일수 * 24시간 * 12분
                    elif timeframe_name == '15m':
                        total_needed = days_back * 24 * 4   # 일수 * 24시간 * 4분
                    elif timeframe_name == '1h':
                        total_needed = days_back * 24       # 일수 * 24시간
                    elif timeframe_name == '4h':
                        total_needed = days_back * 6        # 일수 * 6시간
                    else:  # 1d
                        total_needed = days_back            # 일봉: 일수만큼
                    
                    self.logger.info(f"{coin} {timeframe_name} {days_back}일치 데이터 수집 시작 (목표: {total_needed}개)")
                    
                    # 날짜 설정
                    from datetime import datetime, timedelta
                    if start_date and end_date:
                        # 전달받은 날짜 사용
                        actual_start_date = start_date
                        actual_end_date = end_date
                    else:
                        # days_back 기준으로 계산
                        actual_end_date = datetime.now()
                        actual_start_date = actual_end_date - timedelta(days=days_back)
                    
                    # 과거부터 현재까지 순차적으로 수집
                    current_date = actual_start_date
                    last_progress_pct = 0
                    
                    while current_date <= actual_end_date and total_collected < total_needed:
                        # 현재 날짜를 ISO 형식으로 변환
                        to_date = current_date.strftime('%Y-%m-%dT%H:%M:%S')
                        
                        # 200개씩 수집
                        df_chunk = self.collect_candle_data(coin, str(timeframe_value), 200, to_date)
                        
                        if not df_chunk.empty:
                            all_data.append(df_chunk)
                            total_collected += len(df_chunk)
                            
                            # 다음 날짜로 이동 (타임프레임에 따라)
                            if timeframe_name == '1m':
                                current_date += timedelta(minutes=200)
                            elif timeframe_name == '3m':
                                current_date += timedelta(minutes=600)
                            elif timeframe_name == '5m':
                                current_date += timedelta(minutes=1000)
                            elif timeframe_name == '15m':
                                current_date += timedelta(minutes=3000)
                            elif timeframe_name == '1h':
                                current_date += timedelta(hours=200)
                            elif timeframe_name == '4h':
                                current_date += timedelta(hours=800)
                            else:  # 1d
                                current_date += timedelta(days=200)
                            
                            # 진행률 로그를 1% 단위로만 출력 (로그 양 줄이기)
                            progress_pct = int(total_collected / total_needed * 100)
                            if progress_pct > last_progress_pct and progress_pct % 1 == 0:
                                self.logger.info(f"{coin} {timeframe_name} 진행률: {progress_pct}% ({total_collected}/{total_needed})")
                                last_progress_pct = progress_pct
                            
                            # API 호출 제한을 위한 대기
                            time.sleep(0.1)
                        else:
                            self.logger.warning(f"{coin} {timeframe_name} {to_date} 데이터가 없습니다.")
                            # 다음 날짜로 이동
                            if timeframe_name == '1d':
                                current_date += timedelta(days=1)
                            else:
                                current_date += timedelta(hours=1)
                    
                    # 모든 데이터를 하나로 합치기
                    if all_data:
                        df = pd.concat(all_data, ignore_index=True)
                        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                        self.logger.info(f"{coin} {timeframe_name} 총 {len(df)}개 데이터 수집 완료 (목표 대비 {len(df)/total_needed*100:.1f}%)")
                    else:
                        df = pd.DataFrame()
                        self.logger.error(f"{coin} {timeframe_name} 데이터 수집 실패")
                    
                    if not df.empty:
                        # 기술적 지표 계산
                        df = self.calculate_technical_indicators(df)
                        
                        # 미세 특징 추출 (1분봉의 경우)
                        if timeframe_name == '1m':
                            micro_features = self.extract_micro_features(df)
                            if not micro_features.empty:
                                # 미세 특징을 메인 데이터프레임에 병합
                                df = df.merge(micro_features, on='timestamp', how='left', suffixes=('', '_micro'))
                        
                        # 레이블 생성
                        df = self.create_labels(df)
                        
                        # 데이터 저장
                        filename = f"{coin.replace('KRW-', '')}_{timeframe_name}.csv"
                        filepath = f"{self.raw_dir}/{filename}"
                        df.to_csv(filepath, index=False)
                        
                        self.logger.info(f"{filename} 저장 완료")
                        
                        # API 호출 제한을 위한 대기
                        time.sleep(0.1)
                        
                except Exception as e:
                    self.logger.error(f"{coin} {timeframe_name} 처리 실패: {str(e)}")
                    continue
    
    def analyze_data_quality(self):
        """수집된 데이터 품질 분석"""
        self.logger.info("데이터 품질 분석 시작...")
        
        analysis_results = {}
        
        for coin in self.target_coins:
            coin_name = coin.replace('KRW-', '')
            analysis_results[coin_name] = {}
            
            for timeframe_name in self.timeframes.keys():
                filename = f"{coin_name}_{timeframe_name}.csv"
                filepath = f"{self.raw_dir}/{filename}"
                
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    
                    analysis = {
                        'total_rows': len(df),
                        'missing_values': df.isnull().sum().to_dict(),
                        'date_range': {
                            'start': df['timestamp'].min() if 'timestamp' in df.columns else 'N/A',
                            'end': df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'
                        },
                        'price_stats': {
                            'min': df['close'].min() if 'close' in df.columns else 'N/A',
                            'max': df['close'].max() if 'close' in df.columns else 'N/A',
                            'mean': df['close'].mean() if 'close' in df.columns else 'N/A'
                        }
                    }
                    
                    analysis_results[coin_name][timeframe_name] = analysis
        
        # 분석 결과 저장
        with open(f"{self.processed_dir}/data_quality_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info("데이터 품질 분석 완료")
        return analysis_results
    
    def create_training_dataset(self):
        """LSTM 훈련을 위한 데이터셋 생성"""
        self.logger.info("훈련 데이터셋 생성 시작...")
        
        # 1분봉 데이터를 기준으로 멀티 타임프레임 데이터셋 생성
        for coin in self.target_coins:
            coin_name = coin.replace('KRW-', '')
            
            # 1분봉 데이터 로드
            df_1m = pd.read_csv(f"{self.raw_dir}/{coin_name}_1m.csv")
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
            
            # 다른 타임프레임 데이터 로드 및 병합
            for timeframe_name in ['5m', '15m', '1h']:
                try:
                    df_other = pd.read_csv(f"{self.raw_dir}/{coin_name}_{timeframe_name}.csv")
                    df_other['timestamp'] = pd.to_datetime(df_other['timestamp'])
                    
                    # 컬럼명에 타임프레임 접미사 추가
                    suffix_cols = ['close', 'volume', 'rsi', 'macd', 'bb_middle', 'atr']
                    rename_dict = {col: f"{col}_{timeframe_name}" for col in suffix_cols if col in df_other.columns}
                    df_other = df_other.rename(columns=rename_dict)
                    
                    # 1분봉 데이터에 병합
                    df_1m = df_1m.merge(
                        df_other[['timestamp'] + list(rename_dict.values())], 
                        on='timestamp', 
                        how='left'
                    )
                    
                except Exception as e:
                    self.logger.warning(f"{coin_name} {timeframe_name} 병합 실패: {str(e)}")
                    continue
            
            # 최종 훈련 데이터셋 저장
            training_filepath = f"{self.processed_dir}/{coin_name}_training_dataset.csv"
            df_1m.to_csv(training_filepath, index=False)
            
            self.logger.info(f"{coin_name} 훈련 데이터셋 생성 완료: {len(df_1m)}행")
        
        self.logger.info("훈련 데이터셋 생성 완료")

def main():
    """메인 실행 함수"""
    collector = CryptoDataCollector()
    
    # 1. 데이터 수집
    collector.collect_all_data(days_back=30)
    
    # 2. 데이터 품질 분석
    analysis_results = collector.analyze_data_quality()
    
    # 3. 훈련 데이터셋 생성
    collector.create_training_dataset()
    
    print("데이터 수집 및 분석 완료!")

if __name__ == "__main__":
    main()