#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
🚀 2025CoinTrader - 데이터 수집 메인 실행 스크립트 (run_data_collection.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • Data_maker 모듈의 메인 실행 파일 (전체 데이터 수집 파이프라인 조율)
    • 사용자 인터페이스 제공 (인터랙티브 모드 및 명령줄 인자 지원)
    • data_collector.py와 data_analyzer.py를 순차적으로 실행
    • 수집 진행상황 모니터링 및 로그 관리
    • 에러 처리 및 복구 메커니즘 제공

🔧 핵심 기능:
    1. 유연한 실행 옵션: 인터랙티브 모드 또는 명령줄 인자 모드
    2. 데이터 수집: CryptoDataCollector를 통한 다중 코인/타임프레임 수집
    3. 데이터 분석: CryptoDataAnalyzer를 통한 품질 분석 및 시각화
    4. 진행 관리: 수집 진행상황 실시간 출력 및 로깅
    5. 에러 복구: 네트워크 오류 시 자동 재시도 로직

📋 사용 방법:
    # 인터랙티브 모드 (프로그램이 질문하며 안내)
    python run_data_collection.py
    
    # 명령줄 인자 모드 (모든 옵션 직접 지정)
    python run_data_collection.py -c BTC,ETH,SOL -s 2025-01-01 -e 2025-01-31
    
    # 부분 지정 (일부만 지정하고 나머지는 인터랙티브)
    python run_data_collection.py -c BTC,ETH

📊 실행 순서:
    1. 설정 로드 및 검증 (config.py, API 키 확인)
    2. 대상 코인 및 기간 설정
    3. 데이터 수집 실행 (data_collector.py)
    4. 데이터 분석 실행 (data_analyzer.py)
    5. 결과 파일 생성 및 완료 보고

⚠️ 주의사항:
    • 업비트 API 키가 config.py에 올바르게 설정되어 있어야 함
    • 대량 데이터 수집 시 API 호출 제한 고려
    • 수집 중 중단 시 부분 데이터 복구 기능 제공

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
from datetime import datetime, timedelta
import argparse

def main():
    """메인 실행 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='가상화폐 데이터 수집 및 분석')
    parser.add_argument('-c', '--coins', type=str, 
                        help='수집할 코인 목록 (쉼표로 구분, 예: BTC,ETH,SOL)')
    parser.add_argument('-s', '--start', type=str,
                        help='시작 날짜 (YYYY-MM-DD 형식)')
    parser.add_argument('-e', '--end', type=str,
                        help='종료 날짜 (YYYY-MM-DD 형식, 미입력시 오늘)')
    
    args = parser.parse_args()
    
    # 인터랙티브 모드: 인자가 없으면 사용자에게 입력 받기
    if not args.coins:
        print("수집할 코인을 입력하세요 (쉼표로 구분, 예: BTC,ETH,SOL):")
        coins_input = input().strip()
        if not coins_input:
            print("코인을 입력하지 않았습니다. 기본값 사용: BTC,ETH,SOL")
            coins_input = "BTC,ETH,SOL"
    else:
        coins_input = args.coins
    
    if not args.start:
        print("시작 날짜를 입력하세요 (YYYY-MM-DD 형식):")
        start_date_str = input().strip()
        if not start_date_str:
            # 기본값: 30일 전
            start_date = datetime.now() - timedelta(days=30)
            print(f"시작 날짜를 입력하지 않았습니다. 기본값 사용: {start_date.strftime('%Y-%m-%d')}")
        else:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            except ValueError:
                print("잘못된 날짜 형식입니다. 30일 전을 기본값으로 사용합니다.")
                start_date = datetime.now() - timedelta(days=30)
    else:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    
    if not args.end:
        if not args.start:  # 인터랙티브 모드일 때만
            print("종료 날짜를 입력하세요 (YYYY-MM-DD 형식, 엔터시 오늘):")
            end_date_str = input().strip()
            if not end_date_str:
                end_date = datetime.now()
                print(f"종료 날짜를 입력하지 않았습니다. 오늘 날짜 사용: {end_date.strftime('%Y-%m-%d')}")
            else:
                try:
                    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                except ValueError:
                    print("잘못된 날짜 형식입니다. 오늘 날짜를 사용합니다.")
                    end_date = datetime.now()
        else:
            end_date = datetime.now()
    else:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    # 날짜 검증
    if start_date >= end_date:
        print("시작 날짜가 종료 날짜보다 늦습니다. 날짜를 교환합니다.")
        start_date, end_date = end_date, start_date
    
    # 수집 기간 계산
    days_back = (end_date - start_date).days + 1
    
    # 코인 목록 파싱
    coin_list = [coin.strip().upper() for coin in coins_input.split(',')]
    
    print("=" * 60)
    print("가상화폐 LSTM 모델 데이터 수집 및 분석 시스템")
    print("=" * 60)
    print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"수집 대상 코인: {', '.join(coin_list)}")
    print(f"수집 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({days_back}일)")
    print()
    
    try:
        # 1. 데이터 수집
        print("1단계: 데이터 수집 시작...")
        from data_collector import CryptoDataCollector
        
        collector = CryptoDataCollector()
        # 사용자가 지정한 코인 목록 설정
        collector.target_coins = [f'KRW-{coin}' for coin in coin_list]
        
        # 환경 변수에서 결과 폴더 확인 및 설정
        result_folder = os.environ.get('RESULT_FOLDER_PATH')
        if result_folder and os.path.exists(result_folder):
            # 결과 폴더가 지정된 경우 해당 폴더 내에 DATA 폴더 생성
            collector.data_dir = os.path.join(result_folder, 'DATA')
            collector.raw_dir = os.path.join(collector.data_dir, 'raw')
            collector.processed_dir = os.path.join(collector.data_dir, 'processed')
            # 디렉토리 재생성
            collector.create_directories()
            print(f"데이터 저장 경로: {collector.data_dir}")
        
        collector.collect_all_data(days_back=days_back, start_date=start_date, end_date=end_date)
        print("[OK] 데이터 수집 완료")
        print()
        
        # 2. 데이터 품질 분석
        print("2단계: 데이터 품질 분석...")
        analysis_results = collector.analyze_data_quality()
        print("[OK] 데이터 품질 분석 완료")
        print()
        
        # 3. 훈련 데이터셋 생성
        print("3단계: 훈련 데이터셋 생성...")
        collector.create_training_dataset()
        print("[OK] 훈련 데이터셋 생성 완료")
        print()
        
        # 4. 데이터 분석 및 시각화
        print("4단계: 데이터 분석 및 시각화...")
        from data_analyzer import CryptoDataAnalyzer
        
        analyzer = CryptoDataAnalyzer()
        # 사용자가 지정한 코인 목록 설정
        analyzer.target_coins = coin_list
        
        # 환경 변수에서 결과 폴더 확인 및 설정
        if result_folder and os.path.exists(result_folder):
            analyzer.data_dir = os.path.join(result_folder, 'DATA')
            analyzer.raw_dir = os.path.join(analyzer.data_dir, 'raw')
            analyzer.processed_dir = os.path.join(analyzer.data_dir, 'processed')
        
        report = analyzer.generate_analysis_report()
        print("[OK] 데이터 분석 및 시각화 완료")
        print()
        
        # 5. 결과 요약
        print("=" * 60)
        print("작업 완료 요약")
        print("=" * 60)
        
        # 수집된 데이터 요약
        print("[DATA] 수집된 데이터:")
        
        # 환경 변수에서 결과 폴더 확인
        result_folder = os.environ.get('RESULT_FOLDER_PATH')
        if result_folder and os.path.exists(result_folder):
            # 결과 폴더가 지정된 경우
            data_dir = os.path.join(result_folder, 'DATA')
            processed_dir = os.path.join(data_dir, 'processed')
        else:
            # 기본 경로 사용
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, 'DATA')
            processed_dir = os.path.join(data_dir, 'processed')
        
        for coin in coin_list:
            filepath = os.path.join(processed_dir, f"{coin}_training_dataset.csv")
            if os.path.exists(filepath):
                import pandas as pd
                df = pd.read_csv(filepath)
                print(f"  - {coin}: {len(df):,}개 데이터 포인트")
        
        # 생성된 파일들
        print("\n[FILES] 생성된 파일들:")
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(('.csv', '.json', '.png', '.html')):
                        rel_path = os.path.relpath(os.path.join(root, file), data_dir)
                        print(f"  - {rel_path}")
        
        print("\n[NEXT] 다음 단계:")
        print("  1. analysis_report.json 파일을 확인하여 데이터 품질 검토")
        print("  2. *_price_chart.html 파일을 브라우저에서 열어 시각적 분석")
        print("  3. LSTM 모델 개발 및 훈련 진행")
        
        print("\n[COMPLETE] 모든 작업이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"\n[ERROR] 오류 발생: {str(e)}")
        print("상세한 오류 정보는 로그 파일을 확인하세요.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 