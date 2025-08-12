"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📊 2025CoinTrader - 모델 훈련 진행상황 모니터링 도구 (monitor_v04_training.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • Training_v04.py 실행 중 훈련 진행상황 실시간 모니터링
    • 생성되는 모델 파일 및 결과 파일 상태 추적
    • 훈련 완료 여부 및 성과 지표 확인
    • 훈련 과정에서 발생하는 오류 및 이슈 감지
    • 생성된 시각화 파일 및 모델 메타데이터 검증

🔧 핵심 기능:
    1. 파일 모니터링: 모델 파일, 시각화 파일, 로그 파일 생성 상태 추적
    2. 진행률 표시: 훈련 진행상황을 실시간으로 표시
    3. 성능 확인: 생성된 confusion matrix, training history 그래프 확인
    4. 상태 리포트: 훈련 완료 시 최종 결과 요약 보고
    5. 에러 감지: 훈련 중 발생할 수 있는 문제점 조기 발견

📁 모니터링 대상 파일:
    • best_multitimeframe_model_v04.pth: 최고 성능 모델 가중치
    • multitimeframe_bilstm_crypto_model_v04.pth: 최종 모델 가중치
    • multitimeframe_preprocessor_v04.pkl: 데이터 전처리기
    • model_info_v04.pkl: 모델 메타데이터 및 훈련 정보
    • multitimeframe_training_history_v04.png: 훈련 히스토리 그래프
    • multitimeframe_confusion_matrix_v04.png: 혼동 행렬 시각화

📋 사용 방법:
    # 별도 터미널에서 실행 (Training_v04.py와 동시 실행)
    python monitor_v04_training.py
    
    # 또는 Training_v04.py 실행 후 결과 확인용으로 사용
    python monitor_v04_training.py

⚠️ 주의사항:
    • Training_v04.py와 같은 디렉토리에서 실행해야 함
    • 대용량 모델 파일 생성 시 디스크 공간 확인
    • GPU 메모리 부족 시 훈련이 중단될 수 있음

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import glob
from datetime import datetime

def monitor_v04_training():
    """V04 다중 시간대 모델 훈련 모니터링"""
    print("=== 다중 시간대 모델 V04 훈련 모니터링 ===")
    print("특징: 1분봉 + 5분봉 MACD 통합")
    print("목표: 더 나은 예측 정확도\n")
    
    expected_files = [
        'best_multitimeframe_model_v04.pth',
        'multitimeframe_training_history_v04.png',
        'multitimeframe_confusion_matrix_v04.png',
        'multitimeframe_preprocessor_v04.pkl',
        'multitimeframe_bilstm_crypto_model_v04.pth',
        'model_info_v04.pkl'
    ]
    
    start_time = datetime.now()
    last_check_time = start_time
    
    while True:
        current_time = datetime.now()
        elapsed = current_time - start_time
        since_last_check = (current_time - last_check_time).seconds
        
        # 15초마다 확인
        if since_last_check >= 15:
            print(f"\n[{current_time.strftime('%H:%M:%S')}] 경과 시간: {elapsed.seconds//60}분 {elapsed.seconds%60}초")
            print("파일 상태 확인:")
            
            completed_files = []
            for file in expected_files:
                if os.path.exists(file):
                    file_size = os.path.getsize(file) / 1024  # KB
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file))
                    print(f"  [O] {file} ({file_size:.1f} KB) - {mod_time.strftime('%H:%M:%S')}")
                    completed_files.append(file)
                else:
                    print(f"  [X] {file} - 대기 중...")
            
            # 모든 파일이 생성되었는지 확인
            if len(completed_files) >= len(expected_files) - 1:  # 마지막 파일 제외
                print("\n[COMPLETE] 훈련이 완료되었습니다!")
                break
            
            # 최신 pth 파일 확인
            pth_files = glob.glob("*multitimeframe*v04*.pth")
            if pth_files:
                latest_pth = max(pth_files, key=os.path.getctime)
                print(f"\n최신 체크포인트: {latest_pth}")
                
                # 모델 크기 확인
                size_mb = os.path.getsize(latest_pth) / 1024 / 1024
                print(f"모델 크기: {size_mb:.2f} MB")
            
            last_check_time = current_time
        
        time.sleep(1)  # CPU 사용률 감소
    
    print("\n=== 훈련 완료 ===")
    
    # 결과 확인
    if os.path.exists('multitimeframe_training_history_v04.png'):
        size = os.path.getsize('multitimeframe_training_history_v04.png') / 1024
        print(f"\n훈련 기록 이미지 생성: multitimeframe_training_history_v04.png ({size:.1f} KB)")
    
    if os.path.exists('multitimeframe_confusion_matrix_v04.png'):
        size = os.path.getsize('multitimeframe_confusion_matrix_v04.png') / 1024
        print(f"혼동 행렬 이미지 생성: multitimeframe_confusion_matrix_v04.png ({size:.1f} KB)")
    
    if os.path.exists('model_info_v04.pkl'):
        import pickle
        with open('model_info_v04.pkl', 'rb') as f:
            model_info = pickle.load(f)
        print("\n모델 정보:")
        print(f"- 특징 수: {model_info['num_features']}")
        print(f"- 임계값: ±{model_info['threshold']}%")
        print(f"- 최고 F1 Score: {model_info.get('best_f1_score', 'N/A'):.4f}")
    
    print("\n예상 개선사항:")
    print("- 5분봉 MACD로 더 긴 추세 포착")
    print("- 1분봉과 5분봉 다이버전스 활용")
    print("- 노이즈 감소로 더 안정적인 신호")

if __name__ == "__main__":
    monitor_v04_training()