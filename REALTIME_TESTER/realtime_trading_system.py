"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📈 2025CoinTrader - 실시간 거래 시스템 (realtime_trading_system.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • 실시간 데이터 수집기 + AI 예측기 통합 시스템
    • 매 분마다 데이터 수집 → AI 분석 → 매매신호 생성 자동화
    • 12분 01초에 11분봉 수집 → 12분봉 상승/하락/횡보 예측 → 매매 결정

🔧 핵심 기능:
    1. 실시간 데이터 수집: BTC 1분봉/5분봉 자동 수집
    2. AI 자동 분석: 수집 즉시 다음 분봉 예측
    3. 매매신호 생성: 예측 결과를 매수/매도/홀드 신호로 변환
    4. 통합 출력: 데이터 수집과 예측 결과를 한 번에 표시

📊 시스템 흐름:
    데이터 수집 → AI 예측 → 매매신호 → 출력
    
    예시:
    [16:12:01] 🆕 1분봉 [16:11] 시가: 159,888,000 | 고가: 159,888,000 | 저가: 159,805,000 | 종가: 159,888,000
    [16:12:01] 🟢 매수 신호 - 상승 예측 (신뢰도: 67.8%)

🎮 사용 방법:
    python realtime_trading_system.py
    
    중단: Ctrl+C

⚠️ 주의사항:
    • AI 모델 파일 필요: multitimeframe_bilstm_crypto_model_v04.pth
    • 예측 결과는 참고용이며 실제 투자 결과를 보장하지 않음
    • 충분한 데이터 축적 후 신뢰도 높은 신호 생성

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta

# 로컬 모듈 임포트
from realtime_data_collector import RealtimeDataCollector
from realtime_predictor import RealtimePredictor

class RealtimeTradingSystem:
    """실시간 거래 시스템 - 데이터 수집 + AI 예측 통합"""
    
    def __init__(self):
        """초기화"""
        print("🚀 실시간 AI 거래 시스템 초기화 중...")
        
        # 데이터 수집기 초기화
        self.data_collector = RealtimeDataCollector()
        
        # AI 예측기 초기화
        self.predictor = RealtimePredictor()
        
        # 시스템 상태
        self.is_running = False
        
        print("✅ 실시간 AI 거래 시스템 준비 완료!")
        print("💡 시스템 동작: 데이터 수집 → AI 분석 → 매매신호")
        
    def on_new_minute_data(self, coin_name: str):
        """새로운 1분봉 데이터 수집 시 호출되는 콜백"""
        try:
            # 최신 데이터 가져오기
            df_1m = self.data_collector.get_latest_data(coin_name, '1m')
            df_5m = self.data_collector.get_latest_data(coin_name, '5m')
            
            if df_1m.empty:
                return
                
            # AI 예측 수행
            prediction_output = self.predictor.analyze_realtime_data(df_1m, df_5m)
            
            # 예측 결과 출력
            print(prediction_output)
            print()  # 빈 줄 추가
            
        except Exception as e:
            print(f"❌ AI 예측 중 오류: {str(e)}")
    
    def on_new_5min_data(self, coin_name: str):
        """새로운 5분봉 데이터 수집 시 호출되는 콜백"""
        try:
            # 5분봉 업데이트 시에도 추가 분석 가능
            # 현재는 1분봉 기준으로만 예측하므로 별도 처리 없음
            pass
            
        except Exception as e:
            print(f"❌ 5분봉 처리 중 오류: {str(e)}")
    
    def start_realtime_system(self, market: str = 'KRW-BTC'):
        """실시간 시스템 시작"""
        if self.is_running:
            print("⚠️ 실시간 시스템이 이미 실행 중입니다.")
            return
            
        print(f"\n🎯 {market} 실시간 AI 거래 시스템 시작...")
        print("📊 데이터 수집과 AI 분석이 자동으로 진행됩니다.")
        print("⏹️ 중단하려면 Ctrl+C를 누르세요.\n")
        
        try:
            # 초기 과거 데이터 수집
            if not self.data_collector.initialize_historical_data(market):
                print("❌ 초기 데이터 수집 실패. 시스템을 시작할 수 없습니다.")
                return
            
            # 실시간 시스템 시작
            self.is_running = True
            coin_name = market.replace('KRW-', '')
            
            # 실시간 수집 스레드 시작
            self.data_collector.is_collecting = True
            self.collection_thread = threading.Thread(
                target=self.realtime_loop,
                args=(market, coin_name),
                daemon=True
            )
            self.collection_thread.start()
            
            # 메인 스레드에서 대기
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.stop_system()
        except Exception as e:
            print(f"❌ 시스템 오류: {str(e)}")
            self.stop_system()
    
    def realtime_loop(self, market: str, coin_name: str):
        """실시간 수집 및 분석 루프"""
        while self.is_running:
            try:
                # 새로운 분봉 데이터 수집
                success = self.data_collector.collect_new_minute_data(market)
                
                if success:
                    # 데이터 수집 성공 시 AI 예측 수행
                    self.on_new_minute_data(coin_name)
                
                # 5분봉 업데이트 확인 및 처리
                self.data_collector.update_multi_timeframes(coin_name)
                
                # 다음 분까지 대기
                self.data_collector.wait_for_next_minute()
                
            except Exception as e:
                print(f"❌ 실시간 루프 오류: {str(e)}")
                time.sleep(10)  # 오류 발생 시 10초 대기 후 재시도
    
    def stop_system(self):
        """실시간 시스템 중단"""
        print("\n⏹️ 실시간 AI 거래 시스템 중단 요청...")
        self.is_running = False
        self.data_collector.is_collecting = False
        
        if hasattr(self, 'collection_thread') and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        print("✅ 실시간 AI 거래 시스템이 중단되었습니다.")
    
    def get_system_status(self) -> dict:
        """시스템 상태 반환"""
        data_status = self.data_collector.get_data_status()
        
        return {
            'is_running': self.is_running,
            'data_collector_active': self.data_collector.is_collecting,
            'ai_model_loaded': self.predictor.model is not None,
            'data_status': data_status
        }

def main():
    """메인 실행 함수"""
    print("🤖 2025CoinTrader 실시간 AI 거래 시스템")
    print("=" * 60)
    print("💡 이 시스템은 실시간으로 BTC 데이터를 수집하고")
    print("💡 AI 모델로 다음 분봉을 예측하여 매매신호를 생성합니다.")
    print("=" * 60)
    print()
    
    # 시스템 생성 및 시작
    trading_system = RealtimeTradingSystem()
    
    try:
        trading_system.start_realtime_system('KRW-BTC')
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다...")
    finally:
        trading_system.stop_system()

if __name__ == "__main__":
    main()