"""
업비트 API 테스트 - 실제 데이터 가용성 확인
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def test_upbit_data_availability():
    """업비트 API 데이터 가용성 테스트"""
    
    # 테스트 설정
    market = "KRW-BTC"
    start_date = datetime(2023, 3, 8)
    end_date = datetime(2023, 3, 9)  # 1일만 테스트
    
    print(f"=== 업비트 API 데이터 가용성 테스트 ===")
    print(f"코인: {market}")
    print(f"기간: {start_date} ~ {end_date}")
    print()
    
    # 현재부터 과거로 1시간씩 확인
    current_time = end_date
    total_expected = 24 * 60  # 1일 = 1440분
    total_collected = 0
    missing_periods = []
    
    while current_time >= start_date:
        try:
            # 해당 시간 이후 60개 캔들 요청
            to_date = current_time.strftime('%Y-%m-%dT%H:%M:%S')
            
            response = requests.get(
                "https://api.upbit.com/v1/candles/minutes/1",
                params={"market": market, "count": 60, "to": to_date}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    collected_count = len(df)
                    total_collected += collected_count
                    
                    # 해당 시간대의 첫 번째와 마지막 캔들 확인
                    first_candle = data[-1]['candle_date_time_kst']  # 가장 과거
                    last_candle = data[0]['candle_date_time_kst']    # 가장 최근
                    
                    print(f"{to_date}: {collected_count}개 수집 ({first_candle} ~ {last_candle})")
                    
                    if collected_count < 60:
                        missing_periods.append(f"{to_date}: {60-collected_count}개 누락")
                        
                else:
                    print(f"{to_date}: 데이터 없음")
                    missing_periods.append(f"{to_date}: 완전 누락")
            else:
                print(f"{to_date}: API 오류 ({response.status_code})")
                
        except Exception as e:
            print(f"{to_date}: 예외 발생 - {e}")
            
        # 1시간 이전으로 이동
        current_time -= timedelta(hours=1)
        time.sleep(0.2)  # API 제한 고려
    
    print("\n=== 결과 요약 ===")
    print(f"예상 데이터: {total_expected}개")
    print(f"실제 수집: {total_collected}개")
    print(f"수집률: {total_collected/total_expected*100:.1f}%")
    print(f"누락 구간: {len(missing_periods)}개")
    
    if missing_periods:
        print("\n=== 누락 구간 상세 ===")
        for period in missing_periods[:10]:  # 상위 10개만 출력
            print(f"  - {period}")

if __name__ == "__main__":
    test_upbit_data_availability()