#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
🚀 2025CoinTrader - 통합 암호화폐 데이터 수집 및 백테스팅 시스템 (integrated_crypto_backtester.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • 랜덤한 기간에 대한 암호화폐 데이터 자동 수집
    • 수집된 데이터를 이용한 자동 백테스팅 수행
    • 2018년 1월 1일 ~ 현재 사이의 랜덤한 시작점 선택
    • 사용자 지정 기간(일수)에 대한 데이터 수집 및 분석
    • 완전 자동화된 워크플로우 제공

🔧 핵심 기능:
    1. 사용자 인터페이스: 코인명과 분석 기간(일수) 입력
    2. 랜덤 기간 생성: 2018-01-01 ~ 현재 사이의 랜덤한 시작점
    3. 자동 데이터 수집: run_data_collection.py 모듈 실행
    4. 자동 백테스팅: backtest_btc_optimized.py 모듈 실행
    5. 통합 로깅: 전체 과정에 대한 상세 로그 기록

💰 사용 시나리오:
    • 다양한 시장 상황에서의 전략 성능 검증
    • 무작위 기간 테스트를 통한 전략 견고성 확인
    • 과최적화(overfitting) 방지를 위한 out-of-sample 테스트
    • 시장 변동성이 다른 구간에서의 성능 비교

📋 사용 방법:
    python integrated_crypto_backtester.py
    
    입력 예시:
    - 코인: BTC
    - 기간: 30일
    
    결과: 2018-2024 사이 랜덤한 30일 구간 데이터로 백테스트 수행

📊 실행 순서:
    1. 사용자 입력 (코인명, 기간)
    2. 랜덤 시작 날짜 생성
    3. 데이터 수집 모듈 실행
    4. 백테스팅 모듈 실행
    5. 결과 파일 정리 및 폴더 이동
    6. 실행 요약 리포트 생성

📁 결과 파일 구조:
    RESULT/YYYYMMDD_HHMMSS/
    ├── integrated_backtester_*.log     # 전체 실행 로그
    ├── BATCH1/
    │   ├── data_collection.log         # 데이터 수집 로그
    │   ├── backtest_btc_optimized_*.log # 백테스팅 상세 로그
    │   ├── backtest_btc_optimized_results_*.csv # 거래 내역 데이터
    │   └── DATA/                       # 수집된 데이터
    └── BATCH2/
        ├── data_collection.log         # 데이터 수집 로그
        ├── backtest_btc_optimized_*.log # 백테스팅 상세 로그
        ├── backtest_btc_optimized_results_*.csv # 거래 내역 데이터
        └── DATA/                       # 수집된 데이터

⚠️ 주의사항:
    • 업비트 API 키가 올바르게 설정되어 있어야 함
    • 훈련된 LSTM 모델 파일이 필요
    • 랜덤 기간이므로 데이터 가용성에 따라 실제 수집 기간이 달라질 수 있음
    • 시장 휴일이나 거래 중단 기간은 자동으로 제외됨

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import subprocess
import random
import re
from datetime import datetime, timedelta
import logging

# 현재 파일의 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'Data_maker'))
sys.path.append(os.path.join(current_dir, 'Upbit_Trader'))

# 로거 초기화 (설정은 _setup_logging에서 수행)
logger = logging.getLogger(__name__)

class IntegratedCryptoBacktester:
    """통합 암호화폐 데이터 수집 및 백테스팅 시스템"""
    
    def __init__(self):
        self.current_dir = current_dir
        self.data_collection_script = os.path.join(current_dir, 'Data_maker', 'run_data_collection.py')
        self.backtest_script = os.path.join(current_dir, 'Upbit_Trader', 'backtest_btc_optimized.py')
        
        # 실행 시작 시간 저장 (결과 폴더명으로 사용)
        self.start_time = datetime.now()
        self.start_timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # 결과 폴더 경로 설정
        self.result_base_dir = os.path.join(current_dir, 'RESULT')
        self.result_dir = os.path.join(self.result_base_dir, self.start_timestamp)
        
        # 랜덤 시드 설정 (재현 가능한 결과를 위해)
        random.seed()
        
        # 로깅 설정 (결과 폴더에 저장)
        self._setup_logging()
        
        # 시작 로그
        logger.info(f"시작: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} | 결과폴더: RESULT/{self.start_timestamp}")
    
    def _setup_logging(self):
        """로깅 설정"""
        # 결과 폴더 생성
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        # 로그 파일 경로 설정 (결과 폴더에만 저장)
        log_file = os.path.join(self.result_dir, f'integrated_backtester_{self.start_timestamp}.log')
        
        # 모든 기존 핸들러 제거
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
        
        # 루트 로거의 핸들러도 제거
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 로거 설정
        global logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # 로거의 기존 핸들러 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 파일 핸들러 추가 (결과 폴더에만)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # 콘솔 핸들러 추가
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        # 로거가 상위 로거로 전파되지 않도록 설정
        logger.propagate = False
        
        # 루트 로거도 동일하게 설정 (다른 모듈에서 사용할 수 있도록)
        logging.root.setLevel(logging.INFO)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 루트 로거에도 파일 핸들러 추가
        root_file_handler = logging.FileHandler(log_file, encoding='utf-8')
        root_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.root.addHandler(root_file_handler)
        
        # 루트 로거에도 콘솔 핸들러 추가
        root_console_handler = logging.StreamHandler()
        root_console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.root.addHandler(root_console_handler)
    
    def get_user_input(self):
        """사용자 입력 받기"""
        # 기본적으로 BTC 사용
        coin_input = "BTC"
        
        # 기간 입력
        while True:
            try:
                days_input = input("분석 기간을 일 단위로 입력하세요 (예: 30): ").strip()
                days = int(days_input)
                if 1 <= days <= 365:
                    break
                else:
                    print("1일 이상 365일 이하로 입력해주세요.")
            except ValueError:
                print("숫자로 입력해주세요.")
        
        # 배치수 입력
        while True:
            try:
                batch_input = input("실행할 배치 수를 입력하세요 (예: 5): ").strip()
                batch_count = int(batch_input)
                if 1 <= batch_count <= 100:
                    break
                else:
                    print("1개 이상 100개 이하로 입력해주세요.")
            except ValueError:
                print("숫자로 입력해주세요.")
        
        return coin_input, days, batch_count
    
    def generate_random_period(self, days):
        """랜덤한 시작 날짜와 종료 날짜 생성"""
        # 기준 날짜 설정
        start_boundary = datetime(2018, 1, 1)
        end_boundary = datetime.now()
        
        # 분석 기간을 고려하여 실제 선택 가능한 범위 계산
        latest_possible_start = end_boundary - timedelta(days=days)
        
        if latest_possible_start <= start_boundary:
            logger.warning("요청한 기간이 너무 깁니다. 최대 가능한 기간으로 조정합니다.")
            start_date = start_boundary
            end_date = end_boundary
        else:
            # 랜덤한 시작 날짜 선택
            time_range = (latest_possible_start - start_boundary).days
            random_days = random.randint(0, time_range)
            start_date = start_boundary + timedelta(days=random_days)
            end_date = start_date + timedelta(days=days-1)
        
        return start_date, end_date
    
    def run_data_collection(self, coin, start_date, end_date, batch_dir):
        """데이터 수집 실행"""
        try:
            # 데이터 수집 스크립트 실행
            cmd = [
                sys.executable, 
                self.data_collection_script,
                '-c', coin,
                '-s', start_date.strftime('%Y-%m-%d'),
                '-e', end_date.strftime('%Y-%m-%d')
            ]
            
            # 배치 폴더에서 실행
            result = subprocess.run(cmd, cwd=batch_dir, capture_output=False, text=True)
            
            return result.returncode == 0
                
        except Exception as e:
            logger.error(f"데이터 수집 실패: {str(e)}")
            return False
    
    def run_backtest(self, coin, batch_dir):
        """백테스팅 실행"""
        try:
            # 코인에 따라 적절한 백테스팅 스크립트 선택
            if coin == 'BTC':
                backtest_script = os.path.join(self.current_dir, 'Upbit_Trader', 'backtest_btc_optimized.py')
            else:
                # 기본적으로 BTC 스크립트 사용 (다른 코인용으로 수정 필요시)
                backtest_script = os.path.join(self.current_dir, 'Upbit_Trader', 'backtest_btc_optimized.py')
            
            cmd = [sys.executable, backtest_script]
            
            # 배치 폴더에서 실행
            result = subprocess.run(cmd, cwd=batch_dir, capture_output=False, text=True)
            
            return result.returncode == 0
                
        except Exception as e:
            logger.error(f"백테스팅 실패: {str(e)}")
            return False
    
    def create_batch_directory(self, batch_num):
        """배치별 폴더 생성"""
        batch_dir = os.path.join(self.result_dir, f'BATCH{batch_num}')
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        return batch_dir
    
    def extract_backtest_results(self, batch_dir, batch_num):
        """백테스트 결과 추출"""
        try:
            # 백테스트 로그 파일 경로
            log_file = os.path.join(batch_dir, f'backtest_btc_optimized_{self.start_timestamp}.log')
            
            if not os.path.exists(log_file):
                return {
                    'initial_capital': 10000000,
                    'final_value': 10000000,
                    'total_return': 0.00,
                    'total_trades': 0,
                    'filtered_signals': 0
                }
            
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # 파일의 끝부분에서 "BTC OPTIMIZED BACKTEST RESULTS" 섹션 찾기
                results_section_start = -1
                for i in range(len(lines) - 1, -1, -1):
                    if "BTC OPTIMIZED BACKTEST RESULTS" in lines[i]:
                        results_section_start = i
                        break
                
                if results_section_start == -1:
                    # 결과 섹션을 찾지 못한 경우 기본값 반환
                    return {
                        'initial_capital': 10000000,
                        'final_value': 10000000,
                        'total_return': 0.00,
                        'total_trades': 0,
                        'filtered_signals': 0
                    }
                
                # 결과 섹션에서 데이터 추출
                initial_capital = 10000000
                final_value = 10000000
                total_return = 0.00
                total_trades = 0
                filtered_signals = 0
                
                for i in range(results_section_start, len(lines)):
                    line = lines[i].strip()
                    
                    if "Initial Capital:" in line:
                        match = re.search(r'Initial Capital: ([\d,]+) KRW', line)
                        if match:
                            initial_capital = int(match.group(1).replace(',', ''))
                    
                    elif "Final Value:" in line:
                        match = re.search(r'Final Value: ([\d,]+) KRW', line)
                        if match:
                            final_value = int(match.group(1).replace(',', ''))
                    
                    elif "Total Return:" in line:
                        match = re.search(r'Total Return: ([-\d.]+)%', line)
                        if match:
                            total_return = float(match.group(1))
                    
                    elif "Total Trades:" in line:
                        match = re.search(r'Total Trades: (\d+)', line)
                        if match:
                            total_trades = int(match.group(1))
                    
                    elif "Filtered Signals:" in line:
                        match = re.search(r'Filtered Signals: (\d+)', line)
                        if match:
                            filtered_signals = int(match.group(1))
                
                return {
                    'initial_capital': initial_capital,
                    'final_value': final_value,
                    'total_return': total_return,
                    'total_trades': total_trades,
                    'filtered_signals': filtered_signals
                }
                    
        except Exception as e:
            logger.warning(f"배치 {batch_num} 결과 추출 실패: {str(e)}")
            return {
                'initial_capital': 10000000,
                'final_value': 10000000,
                'total_return': 0.00,
                'total_trades': 0,
                'filtered_signals': 0
            }
    
    def create_batch_summary(self, coin, batch_results):
        """배치별 결과 요약 생성"""
        try:
            logger.info("="*80)
            logger.info("📊 배치 백테스팅 결과 종합")
            logger.info("="*80)
            
            total_trades = 0
            total_signals = 0
            total_return_sum = 0
            successful_batches = 0
            profitable_batches = 0
            loss_batches = 0
            
            # 성공한 배치들의 결과 출력
            for result in batch_results:
                if result['status'] == 'success':
                    batch_num = result['batch_num']
                    start_date = result['start_date']
                    end_date = result['end_date']
                    days = result['days']
                    
                    # 백테스트 결과 추출
                    backtest_results = self.extract_backtest_results(result['batch_dir'], batch_num)
                    
                    logger.info(f"[BATCH{batch_num}] {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({days}일)")
                    logger.info(f"  초기자본: {backtest_results['initial_capital']:,} KRW")
                    logger.info(f"  최종자산: {backtest_results['final_value']:,} KRW")
                    logger.info(f"  수익률: {backtest_results['total_return']:.2f}%")
                    logger.info(f"  거래횟수: {backtest_results['total_trades']}회")
                    logger.info(f"  필터된신호: {backtest_results['filtered_signals']}개")
                    logger.info("")
                    
                    total_trades += backtest_results['total_trades']
                    total_signals += backtest_results['filtered_signals']
                    total_return_sum += backtest_results['total_return']
                    successful_batches += 1
                    
                    # 수익/손실 배치 구분
                    if backtest_results['total_return'] > 0:
                        profitable_batches += 1
                    elif backtest_results['total_return'] < 0:
                        loss_batches += 1
                    # 수익률이 0%인 경우는 손실 배치로 분류하지 않음
                else:
                    logger.info(f"[BATCH{result['batch_num']}] 실패 - {result.get('error', '알 수 없는 오류')}")
                    logger.info("")
            
            # 전체 통계
            logger.info("="*60)
            logger.info("📈 전체 배치 통계")
            logger.info("="*60)
            logger.info(f"총 배치 수: {len(batch_results)}개")
            
            if successful_batches > 0:
                avg_return = total_return_sum / successful_batches
                avg_trades = total_trades / successful_batches
                # logger.info(f"평균 수익률: {avg_return:.2f}%")
                # logger.info(f"총 거래횟수: {total_trades}회")
                # logger.info(f"배치당 평균 거래: {avg_trades:.1f}회")
            else:
                logger.info(f"평균 수익률: 0.00%")
                logger.info(f"총 거래횟수: 0회")
                logger.info(f"배치당 평균 거래: 0.0회")
            
            failed_batches = len(batch_results) - successful_batches
            logger.info(f"수익 배치: {profitable_batches}개 ({(profitable_batches/len(batch_results))*100:.1f}%)")
            logger.info(f"손실 배치: {loss_batches}개 ({(loss_batches/len(batch_results))*100:.1f}%)")
            logger.info("="*80)
            
        except Exception as e:
            logger.warning(f"배치 요약 생성 실패: {str(e)}")
    
    def cleanup_old_files(self):
        """이전 실행 결과 파일 정리 (선택적)"""
        try:
            # 오래된 RESULT 폴더 정리 (30일 이상)
            if os.path.exists(self.result_base_dir):
                current_time = datetime.now()
                for folder_name in os.listdir(self.result_base_dir):
                    folder_path = os.path.join(self.result_base_dir, folder_name)
                    if os.path.isdir(folder_path):
                        try:
                            # 폴더명에서 날짜 추출 (YYYYMMDD_HHMMSS 형식)
                            folder_date = datetime.strptime(folder_name, "%Y%m%d_%H%M%S")
                            if (current_time - folder_date).days > 30:
                                # shutil.rmtree(folder_path) # 실제 삭제는 주의
                                logger.info(f"오래된 결과 폴더 삭제: {folder_name}")
                        except ValueError:
                            # 날짜 형식이 맞지 않는 폴더는 건너뛰기
                            continue
        except Exception as e:
            logger.warning(f"오래된 파일 정리 중 오류: {str(e)}")
    
    def run(self):
        """메인 실행 함수"""
        try:
            # 1. 사용자 입력
            coin, days, batch_count = self.get_user_input()
            
            batch_results = []
            
            for batch_num in range(1, batch_count + 1):
                try:
                    # 배치별 폴더 생성
                    batch_dir = self.create_batch_directory(batch_num)
                    
                    # 환경변수 설정 (각 모듈이 결과 폴더를 인식할 수 있도록)
                    os.environ['RESULT_FOLDER_PATH'] = batch_dir
                    os.environ['RESULT_TIMESTAMP'] = self.start_timestamp
                    
                    # logger.info(f"배치 {batch_num} 환경변수 설정: RESULT_FOLDER_PATH={batch_dir}")
                    
                    # 2. 랜덤 기간 생성 (각 배치마다 새로운 기간)
                    start_date, end_date = self.generate_random_period(days)
                    
                    # 3. 데이터 수집 (배치 폴더에서 실행)
                    if not self.run_data_collection(coin, start_date, end_date, batch_dir):
                        batch_results.append({
                            'batch_num': batch_num,
                            'status': 'failed',
                            'error': '데이터 수집 실패',
                            'batch_dir': batch_dir
                        })
                        continue
                    
                    # 4. 백테스팅 (배치 폴더에서 실행)
                    if not self.run_backtest(coin, batch_dir):
                        batch_results.append({
                            'batch_num': batch_num,
                            'status': 'failed',
                            'error': '백테스팅 실패',
                            'batch_dir': batch_dir
                        })
                        continue
                    
                    # 배치 결과 저장
                    batch_results.append({
                        'batch_num': batch_num,
                        'start_date': start_date,
                        'end_date': end_date,
                        'days': (end_date - start_date).days + 1,
                        'status': 'success',
                        'batch_dir': batch_dir
                    })
                    
                except Exception as e:
                    batch_results.append({
                        'batch_num': batch_num,
                        'status': 'failed',
                        'error': str(e),
                        'batch_dir': batch_dir if 'batch_dir' in locals() else None
                    })
                    continue
            
            # 5. 배치별 결과 요약
            self.create_batch_summary(coin, batch_results)
            
            successful_batches = sum(1 for r in batch_results if r['status'] == 'success')
            failed_batches = len(batch_results) - successful_batches
            
            return 0 if failed_batches == 0 else 1
            
        except KeyboardInterrupt:
            logger.warning("사용자가 프로그램을 중단했습니다.")
            return 1
        except Exception as e:
            logger.error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
            return 1
        finally:
            # 정리 작업
            self.cleanup_old_files()

def main():
    """메인 함수"""
    backtester = IntegratedCryptoBacktester()
    exit_code = backtester.run()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()