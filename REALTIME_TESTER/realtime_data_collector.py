"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📈 2025CoinTrader - 실시간 데이터 수집 모듈 (realtime_data_collector.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • 업비트 API를 통한 실시간 암호화폐 데이터 수집
    • 매 분마다 신규 1분봉 데이터 수집 및 다중 타임프레임 업데이트
    • 코드 실행 시점 기준 과거 2일 초기 데이터 수집
    • 실시간 백테스팅을 위한 연속적 데이터 공급

🔧 핵심 기능:
    1. 초기화: 실행 시점 기준 과거 2일 데이터 수집
    2. 실시간 수집: 매 분마다 새로운 1분봉 데이터 수집
    3. 다중 타임프레임: 1M, 3M, 5M, 15M 자동 생성 및 업데이트
    4. 기술적 지표: MACD, RSI, 볼린저밴드 등 실시간 계산

📊 생성되는 데이터:
    • {코인명}_1m_realtime.csv: 실시간 1분봉 데이터
    • {코인명}_3m_realtime.csv: 실시간 3분봉 데이터  
    • {코인명}_5m_realtime.csv: 실시간 5분봉 데이터
    • {코인명}_15m_realtime.csv: 실시간 15분봉 데이터

🎮 사용 방법:
    collector = RealtimeDataCollector()
    collector.start_realtime_collection('KRW-BTC')  # BTC 실시간 수집 시작

⚠️ 주의사항:
    • 매분 00초~01초 사이에 데이터 수집 실행
    • 프로그램 중단 시까지 무한 루프로 실행
    • 네트워크 오류 시 자동 재시도

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
from typing import Dict, Tuple
import threading
import json

# 로컬 config 모듈 임포트
from config import Config

class RealtimeDataCollector:
    """실시간 가상화폐 데이터 수집 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = Config()
        
        # 실시간 데이터 저장 경로
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_dir, 'REALTIME_DATA')
        self.create_directories()
        
        # 실시간 데이터 버퍼 (메모리에 보관)
        self.data_buffers = {}
        
        # 수집 상태 제어
        self.is_collecting = False
        self.collection_thread = None
        
    def create_directories(self):
        """필요한 디렉토리 생성"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"📁 디렉토리 생성: {self.data_dir}")
    
    def collect_candle_data(self, market: str, timeframe: str, count: int = 200, to_date: str = None) -> pd.DataFrame:
        """캔들 데이터 수집 (기존 collector와 동일)"""
        try:
            if timeframe in ["1", "3", "5", "10", "15", "30", "60", "240"]:
                target_url = f"https://api.upbit.com/v1/candles/minutes/{timeframe}"
            elif timeframe == "D":
                target_url = "https://api.upbit.com/v1/candles/days"
            else:
                raise Exception(f"잘못된 틱 종류: {timeframe}")
            
            params = {"market": market, "count": count}
            if to_date:
                params["to"] = to_date
            
            # API 호출 (429 에러 대응을 위한 재시도 로직 포함)
            max_retries = 3
            retry_delay = 1  # 기본 1초 대기
            
            for attempt in range(max_retries):
                try:
                    response = requests.get(target_url, params=params)
                    response.raise_for_status()
                    candle_data = response.json()
                    break  # 성공하면 루프 종료
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Too Many Requests
                        if attempt < max_retries - 1:  # 마지막 시도가 아니면 재시도
                            wait_time = retry_delay * (2 ** attempt)  # 지수적 백오프
                            print(f"⏳ API 제한 감지 (429). {wait_time}초 대기 후 재시도... (시도 {attempt + 1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise e  # 마지막 시도에서 실패하면 예외 발생
                    else:
                        raise e  # 다른 HTTP 에러는 즉시 발생
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"⏳ 네트워크 오류 감지. {wait_time}초 대기 후 재시도... (시도 {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
            
            if not candle_data:
                print(f"⚠️  {market} {timeframe}분봉 데이터가 없습니다.")
                return pd.DataFrame()
            
            df = pd.DataFrame(candle_data)
            
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])
            
            df = df.rename(columns={
                'candle_date_time_kst': 'timestamp',
                'opening_price': 'open',
                'high_price': 'high', 
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume',
                'candle_acc_trade_price': 'trade_price_volume'
            })
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 데이터 검증
            if df.empty:
                print(f"⚠️  {market} {timeframe}분봉 데이터가 비어있습니다.")
                return df
            
            # 필수 컬럼 확인
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"⚠️  {market} {timeframe}분봉 데이터에 필수 컬럼이 누락되었습니다: {missing_columns}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            print(f"❌ {market} {timeframe}분봉 데이터 수집 실패: {str(e)}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산 (기존 data_collector.py와 동일)"""
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
            
            return df
            
        except Exception as e:
            print(f"❌ 기술적 지표 계산 실패: {str(e)}")
            return df
    
    def create_labels(self, df: pd.DataFrame, future_periods: int = 5, threshold: float = 0.002) -> pd.DataFrame:
        """예측 목표 레이블 생성 (3-class 분류) - 기존 data_collector.py와 동일"""
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
            # print(f"📊 레이블 분포: {label_counts.to_dict()}")  # 실시간에서는 너무 자주 출력되므로 주석 처리
            
            return df
            
        except Exception as e:
            print(f"❌ 레이블 생성 실패: {str(e)}")
            return df
    
    def resample_to_timeframe(self, df_1m: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
        """1분봉 데이터를 다른 타임프레임으로 리샘플링"""
        if df_1m.empty:
            return pd.DataFrame()
            
        try:
            df = df_1m.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # 리샘플링 규칙
            rule = f'{timeframe_minutes}min'
            
            # OHLCV 데이터 리샘플링
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # 인덱스를 컬럼으로 복구
            resampled = resampled.reset_index()
            
            # 기술적 지표 계산 및 레이블 생성
            resampled = self.calculate_technical_indicators(resampled)
            resampled = self.create_labels(resampled)
            
            return resampled
            
        except Exception as e:
            print(f"❌ {timeframe_minutes}분봉 리샘플링 실패: {str(e)}")
            return pd.DataFrame()
    
    def initialize_historical_data(self, market: str) -> bool:
        """초기 과거 2일 데이터 수집"""
        start_time = datetime.now()
        print(f"\n🚀 {market} 초기 과거 2일 데이터 수집 시작... [{start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")
        
        try:
            # 현재 시각 기준 과거 2일 데이터 수집
            current_time = start_time
            
            # 1분봉: 완성된 마지막 분까지 수집 - 2일 = 2880개 (1일 = 1440분)
            last_completed_minute = current_time.replace(second=0, microsecond=0) - timedelta(minutes=1)
            start_minute_1m = last_completed_minute - timedelta(minutes=2879)  # 2879개 + 마지막 1개 = 2880개
            
            # 5분봉: 완성된 마지막 5분봉까지 수집
            current_minute = current_time.minute
            # 현재 진행 중인 5분봉의 시작 시간을 구하고, 그 이전 5분봉이 마지막 완성봉
            current_5m_start = current_minute - (current_minute % 5)
            last_completed_5m = current_5m_start - 5
            if last_completed_5m < 0:
                # 시간이 넘어가는 경우 (예: 02분 실행 시 전 시간 55분봉)
                last_completed_5m_time = (current_time - timedelta(hours=1)).replace(minute=55, second=0, microsecond=0)
            else:
                last_completed_5m_time = current_time.replace(minute=last_completed_5m, second=0, microsecond=0)
            
            # 5분봉 수집 범위 계산 - 2일 = 576개 (1일 = 288개)
            start_5m_time = last_completed_5m_time - timedelta(minutes=2875)  # 575개 * 5분 = 2875분
            
            print(f"📅 1분봉 수집 범위: {start_minute_1m.strftime('%Y-%m-%d %H:%M')} ~ {last_completed_minute.strftime('%Y-%m-%d %H:%M')}")
            print(f"📅 5분봉 수집 범위: {start_5m_time.strftime('%Y-%m-%d %H:%M')} ~ {last_completed_5m_time.strftime('%Y-%m-%d %H:%M')}")
            
            # 1분봉 데이터 수집 - 2880개 (2일치)
            print(f"🔄 1분봉 데이터 수집 시작... (목표: 2880개)")
            
            df_1m_list = []
            total_1m_collected = 0
            max_1m_batches = 15  # 최대 15번 배치 (2880개 목표)
            current_1m_oldest_time = None
            
            for batch in range(max_1m_batches):
                if batch == 0:
                    # 첫 번째 배치: 최신 200개
                    print(f"📥 1분봉 배치 {batch + 1}: 최신 200개 수집")
                    batch_1m_df = self.collect_candle_data(market, "1", 200)
                else:
                    # 두 번째 배치부터: 이전 배치의 가장 오래된 시간 기준
                    to_time_str = current_1m_oldest_time.strftime('%Y-%m-%dT%H:%M:%S')
                    print(f"📥 1분봉 배치 {batch + 1}: {to_time_str} 기준으로 200개 수집")
                    batch_1m_df = self.collect_candle_data(market, "1", 200, to_time_str)
                
                if batch_1m_df.empty:
                    print(f"⚠️ 1분봉 배치 {batch + 1}: 데이터 없음, 수집 종료")
                    break
                
                # timestamp 처리
                batch_1m_df['timestamp'] = pd.to_datetime(batch_1m_df['timestamp'])
                batch_1m_df = batch_1m_df.sort_values('timestamp').reset_index(drop=True)
                
                df_1m_list.append(batch_1m_df)
                total_1m_collected += len(batch_1m_df)
                
                # 가장 오래된 시간 업데이트
                current_1m_oldest_time = batch_1m_df['timestamp'].min()
                
                print(f"✅ 1분봉 배치 {batch + 1}: {len(batch_1m_df)}개 수집 (누적: {total_1m_collected}개)")
                print(f"   시간 범위: {batch_1m_df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} ~ {batch_1m_df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
                print(f"   다음 기준 시간: {current_1m_oldest_time.strftime('%Y-%m-%d %H:%M')}")
                
                # 목표치 도달 확인
                if total_1m_collected >= 2880:
                    print(f"🎯 1분봉 목표치 도달: {total_1m_collected}개 수집 완료")
                    break
                
                # 2일치 시간 범위 도달 확인
                if current_1m_oldest_time <= start_minute_1m:
                    print(f"🎯 1분봉 2일 범위 도달: {current_1m_oldest_time.strftime('%Y-%m-%d %H:%M')} <= {start_minute_1m.strftime('%Y-%m-%d %H:%M')}")
                    break
                
                # API 호출 간 대기
                time.sleep(0.5)
            
            # 1분봉 데이터 합치기
            if df_1m_list:
                df_1m = pd.concat(df_1m_list, ignore_index=True)
                df_1m = df_1m.drop_duplicates(subset=['timestamp'], keep='first')
                df_1m = df_1m.sort_values('timestamp').reset_index(drop=True)
                
                # 완성된 분봉만 필터링
                df_1m = df_1m[df_1m['timestamp'] <= last_completed_minute]
                
                print(f"📋 최종 1분봉 데이터: {len(df_1m)}개")
                print(f"📅 1분봉 실제 수집 범위: {df_1m['timestamp'].min().strftime('%Y-%m-%d %H:%M')} ~ {df_1m['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
                
                if not df_1m.empty:
                    df_1m = self.calculate_technical_indicators(df_1m)
                    df_1m = self.create_labels(df_1m)
                else:
                    print(f"❌ {market} 필터링 후 1분봉 데이터 없음")
                    return False
            else:
                print(f"❌ {market} 1분봉 데이터 수집 실패")
                return False
            
            # 5분봉 데이터 수집 - 576개 (2일치)
            print(f"🔄 5분봉 데이터 수집 시작... (목표: 576개)")
            
            df_5m_list = []
            total_5m_collected = 0
            max_5m_batches = 4  # 최대 4번 배치 (576개 목표)
            current_5m_oldest_time = None
            
            for batch in range(max_5m_batches):
                if batch == 0:
                    # 첫 번째 배치: 최신 200개
                    print(f"📥 5분봉 배치 {batch + 1}: 최신 200개 수집")
                    batch_5m_df = self.collect_candle_data(market, "5", 200)
                else:
                    # 두 번째 배치부터: 이전 배치의 가장 오래된 시간 기준
                    to_time_str = current_5m_oldest_time.strftime('%Y-%m-%dT%H:%M:%S')
                    print(f"📥 5분봉 배치 {batch + 1}: {to_time_str} 기준으로 200개 수집")
                    batch_5m_df = self.collect_candle_data(market, "5", 200, to_time_str)
                
                if batch_5m_df.empty:
                    print(f"⚠️ 5분봉 배치 {batch + 1}: 데이터 없음, 수집 종료")
                    break
                
                # timestamp 처리
                batch_5m_df['timestamp'] = pd.to_datetime(batch_5m_df['timestamp'])
                batch_5m_df = batch_5m_df.sort_values('timestamp').reset_index(drop=True)
                
                # 중복 방지
                if current_5m_oldest_time is not None:
                    batch_5m_df = batch_5m_df[batch_5m_df['timestamp'] < current_5m_oldest_time]
                
                if batch_5m_df.empty:
                    print(f"⚠️ 5분봉 배치 {batch + 1}: 중복 제거 후 데이터 없음, 수집 종료")
                    break
                
                df_5m_list.append(batch_5m_df)
                total_5m_collected += len(batch_5m_df)
                
                # 가장 오래된 시간 업데이트
                current_5m_oldest_time = batch_5m_df['timestamp'].min()
                
                print(f"✅ 5분봉 배치 {batch + 1}: {len(batch_5m_df)}개 수집 (누적: {total_5m_collected}개)")
                print(f"   시간 범위: {batch_5m_df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} ~ {batch_5m_df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
                print(f"   다음 기준 시간: {current_5m_oldest_time.strftime('%Y-%m-%d %H:%M')}")
                
                # 목표치 도달 확인
                if total_5m_collected >= 576:
                    print(f"🎯 5분봉 목표치 도달: {total_5m_collected}개 수집 완료")
                    break
                
                # 2일치 시간 범위 도달 확인
                if current_5m_oldest_time <= start_5m_time:
                    print(f"🎯 5분봉 2일 범위 도달: {current_5m_oldest_time.strftime('%Y-%m-%d %H:%M')} <= {start_5m_time.strftime('%Y-%m-%d %H:%M')}")
                    break
                
                # API 호출 간 대기
                time.sleep(0.5)
            
            # 5분봉 데이터 합치기
            if df_5m_list:
                df_5m = pd.concat(df_5m_list, ignore_index=True)
                df_5m = df_5m.drop_duplicates(subset=['timestamp'], keep='first')
                df_5m = df_5m.sort_values('timestamp').reset_index(drop=True)
                
                # 완성된 5분봉만 필터링
                df_5m = df_5m[df_5m['timestamp'] <= last_completed_5m_time]
                
                print(f"📋 최종 5분봉 데이터: {len(df_5m)}개")
                print(f"📅 5분봉 실제 수집 범위: {df_5m['timestamp'].min().strftime('%Y-%m-%d %H:%M')} ~ {df_5m['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
                
                if not df_5m.empty:
                    df_5m = self.calculate_technical_indicators(df_5m)
                    df_5m = self.create_labels(df_5m)
                else:
                    df_5m = pd.DataFrame()
            else:
                df_5m = pd.DataFrame()
            
            # 데이터 버퍼에 저장 (1분봉, 5분봉만)
            coin_name = market.replace('KRW-', '')
            self.data_buffers[coin_name] = {
                '1m': df_1m,
                '5m': df_5m
            }
            
            # 완료 시간 및 소요 시간 계산
            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            print(f"✅ {market} 초기 데이터 수집 완료: [{end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] (소요시간: {elapsed_time:.3f}초)")
            print()
            
            # 초기 데이터 출력 (마지막 값만)
            if not df_1m.empty:
                last_1m = df_1m.iloc[-1]
                timestamp = last_1m['timestamp'].strftime('%Y-%m-%d %H:%M')
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📊 1분봉 [{timestamp}] 시가: {last_1m['open']:,.0f} | 고가: {last_1m['high']:,.0f} | 저가: {last_1m['low']:,.0f} | 종가: {last_1m['close']:,.0f}")
                print()
            
            if not df_5m.empty:
                last_5m = df_5m.iloc[-1]
                timestamp = last_5m['timestamp'].strftime('%Y-%m-%d %H:%M')
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔥 5분봉 [{timestamp}] 시가: {last_5m['open']:,.0f} | 고가: {last_5m['high']:,.0f} | 저가: {last_5m['low']:,.0f} | 종가: {last_5m['close']:,.0f}")
                print()
            
            # 파일로 저장
            self.save_data_to_files(coin_name)
            
            return True
            
        except Exception as e:
            print(f"❌ {market} 초기 데이터 수집 실패: {str(e)}")
            return False
    
    def collect_new_minute_data(self, market: str) -> bool:
        """새로운 분봉 데이터 1개 수집 (완성된 분봉만)"""
        try:
            current_time = datetime.now()
            # 완성된 마지막 분봉 시간 (현재 시간의 이전 분)
            last_completed_minute = current_time.replace(second=0, microsecond=0) - timedelta(minutes=1)
            to_date = last_completed_minute.strftime('%Y-%m-%dT%H:%M:%S')
            
            # 최신 1분봉 데이터 2개 수집 후 필터링 (여유있게)
            df_new = self.collect_candle_data(market, "1", 2)
            
            if not df_new.empty:
                # 완성된 분봉만 필터링
                df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
                df_new = df_new[df_new['timestamp'] <= last_completed_minute].tail(1).reset_index(drop=True)
            
            if df_new.empty:
                return False
            
            coin_name = market.replace('KRW-', '')
            
            # 기존 1분봉 데이터에 추가
            if coin_name in self.data_buffers and '1m' in self.data_buffers[coin_name]:
                # 기존 데이터와 합치기
                existing_df = self.data_buffers[coin_name]['1m']
                
                # 중복 제거를 위해 timestamp로 비교
                new_timestamp = df_new.iloc[-1]['timestamp']
                if not existing_df[existing_df['timestamp'] == new_timestamp].empty:
                    return False
                
                # 새 데이터 추가하고 가장 과거 데이터 제거 (2880개 유지)
                updated_df = pd.concat([existing_df, df_new], ignore_index=True)
                
                # 정확히 2880개만 유지 (새로 추가되면 가장 과거 1개 제거)
                if len(updated_df) > 2880:
                    updated_df = updated_df.tail(2880).reset_index(drop=True)
                
                # 기술적 지표 재계산 및 레이블 생성
                updated_df = self.calculate_technical_indicators(updated_df)
                updated_df = self.create_labels(updated_df)
                
                self.data_buffers[coin_name]['1m'] = updated_df
                
                # 신규 분봉 데이터 출력
                new_row = df_new.iloc[-1]
                candle_time = new_row['timestamp'].strftime('%Y-%m-%d %H:%M')
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🆕 1분봉 [{candle_time}] 시가: {new_row['open']:,.0f} | 고가: {new_row['high']:,.0f} | 저가: {new_row['low']:,.0f} | 종가: {new_row['close']:,.0f}")
                print()
                
            else:
                # 처음 수집하는 경우
                df_new = self.calculate_technical_indicators(df_new)
                df_new = self.create_labels(df_new)
                self.data_buffers[coin_name] = {'1m': df_new}
            
            # 다중 타임프레임 업데이트
            self.update_multi_timeframes(coin_name)
            
            # 파일 저장
            self.save_data_to_files(coin_name)
            
            return True
            
        except Exception as e:
            print(f"❌ {market} 신규 분봉 수집 실패: {str(e)}")
            return False
    
    def update_multi_timeframes(self, coin_name: str):
        """5분봉 데이터 업데이트 (글로벌 시간 기준 5분봉 완성 시점에만)"""
        if coin_name not in self.data_buffers:
            return
            
        try:
            current_time = datetime.now()
            market = f"KRW-{coin_name}"
            
            # 5분봉 업데이트 조건: 
            # 현재 분이 5의 배수이고(00, 05, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
            # 초가 1초 이내일 때 (5분봉이 방금 완성된 시점)
            if current_time.minute % 5 == 0 and current_time.second <= 1:
                df_5m_new = self.collect_candle_data(market, "5", 1)
                if not df_5m_new.empty and '5m' in self.data_buffers[coin_name]:
                    # 기존 5분봉 데이터에 추가
                    existing_5m = self.data_buffers[coin_name]['5m']
                    new_timestamp = df_5m_new.iloc[-1]['timestamp']
                    
                    # 중복 방지: 동일한 timestamp의 5분봉이 없을 때만 추가
                    if existing_5m.empty or existing_5m[existing_5m['timestamp'] == new_timestamp].empty:
                        updated_5m = pd.concat([existing_5m, df_5m_new], ignore_index=True)
                        
                        # 정확히 576개만 유지 (새로 추가되면 가장 과거 1개 제거)
                        if len(updated_5m) > 576:  # 2일분 = 576개
                            updated_5m = updated_5m.tail(576).reset_index(drop=True)
                        
                        updated_5m = self.calculate_technical_indicators(updated_5m)
                        updated_5m = self.create_labels(updated_5m)
                        self.data_buffers[coin_name]['5m'] = updated_5m
                        
                        # 5분봉 업데이트 알림
                        new_row = df_5m_new.iloc[-1]
                        candle_time = new_row['timestamp'].strftime('%Y-%m-%d %H:%M')
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔥 5분봉 [{candle_time}] 시가: {new_row['open']:,.0f} | 고가: {new_row['high']:,.0f} | 저가: {new_row['low']:,.0f} | 종가: {new_row['close']:,.0f}")
                        print()
            
            # self.logger.debug(f"{coin_name} 5분봉 업데이트 확인 완료")
            
        except Exception as e:
            print(f"❌ {coin_name} 5분봉 업데이트 실패: {str(e)}")
    
    def save_data_to_files(self, coin_name: str):
        """데이터를 CSV 파일로 저장"""
        if coin_name not in self.data_buffers:
            return
            
        try:
            for timeframe, df in self.data_buffers[coin_name].items():
                if not df.empty:
                    filename = f"{coin_name}_{timeframe}_realtime.csv"
                    filepath = os.path.join(self.data_dir, filename)
                    df.to_csv(filepath, index=False)
            
            # print(f"💾 {coin_name} 데이터 파일 저장 완료")
            
        except Exception as e:
            print(f"❌ {coin_name} 데이터 저장 실패: {str(e)}")
    
    def wait_for_next_minute(self):
        """다음 분의 시작까지 대기 (1초 후에 실행)"""
        current_time = datetime.now()
        
        # 다음 분의 1초로 설정
        next_minute = current_time.replace(second=1, microsecond=0) + timedelta(minutes=1)
        
        wait_seconds = (next_minute - current_time).total_seconds()
        
        if wait_seconds > 0:
            time.sleep(wait_seconds)
    
    def realtime_collection_loop(self, market: str):
        """실시간 수집 메인 루프"""
        # print(f"🔄 {market} 실시간 데이터 수집 루프 시작...")
        
        while self.is_collecting:
            try:
                # 새로운 분봉 데이터 수집
                success = self.collect_new_minute_data(market)
                
                # 다음 분까지 대기
                self.wait_for_next_minute()
                
            except KeyboardInterrupt:
                print("⏹️ 사용자에 의한 수집 중단...")
                break
            except Exception as e:
                print(f"❌ 실시간 수집 루프 오류: {str(e)}")
                time.sleep(10)  # 오류 발생 시 10초 대기 후 재시도
    
    def start_realtime_collection(self, market: str):
        """실시간 데이터 수집 시작"""
        if self.is_collecting:
            print("⚠️ 이미 실시간 수집이 진행 중입니다.")
            return
        
        # print(f"🚀 {market} 실시간 데이터 수집 시작...")
        
        # 초기 과거 데이터 수집
        if not self.initialize_historical_data(market):
            print("❌ 초기 데이터 수집 실패. 실시간 수집을 시작할 수 없습니다.")
            return
        
        # 실시간 수집 시작
        self.is_collecting = True
        self.collection_thread = threading.Thread(
            target=self.realtime_collection_loop,
            args=(market,),
            daemon=True
        )
        self.collection_thread.start()

        # print(f"✅ {market} 실시간 수집이 시작되었습니다. 중단하려면 Ctrl+C를 누르세요.\n")

        # 메인 스레드에서 대기
        try:
            while self.is_collecting:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_realtime_collection()
    
    def stop_realtime_collection(self):
        """실시간 데이터 수집 중단"""
        print("⏹️ 실시간 데이터 수집 중단 요청...")
        self.is_collecting = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        print("✅ 실시간 데이터 수집이 중단되었습니다.")
    
    def get_latest_data(self, coin_name: str, timeframe: str) -> pd.DataFrame:
        """최신 데이터 반환 (실시간 백테스팅용)"""
        if coin_name in self.data_buffers and timeframe in self.data_buffers[coin_name]:
            return self.data_buffers[coin_name][timeframe].copy()
        else:
            return pd.DataFrame()
    
    def get_data_status(self) -> Dict:
        """현재 수집된 데이터 상태 반환"""
        status = {}
        for coin_name, timeframes in self.data_buffers.items():
            status[coin_name] = {}
            for tf, df in timeframes.items():
                if not df.empty:
                    status[coin_name][tf] = {
                        'rows': len(df),
                        'latest_time': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S'),
                        'latest_price': df['close'].iloc[-1]
                    }
        return status

def main():
    """메인 실행 함수 - BTC 실시간 수집"""
    # print("🚀 BTC 실시간 데이터 수집기 시작...")
    print("중단하려면 Ctrl+C를 누르세요.")
    
    collector = RealtimeDataCollector()
    
    try:
        collector.start_realtime_collection('KRW-BTC')
    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다...")
    finally:
        collector.stop_realtime_collection()

if __name__ == "__main__":
    main()