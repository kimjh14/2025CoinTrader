#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2025CoinTrader 설치 확인 테스트 스크립트 (한글 인코딩 수정)
"""

def test_imports():
    """모든 필수 모듈 import 테스트"""
    print("모듈 import 테스트 시작...\n")
    
    test_modules = [
        # 핵심 데이터 처리
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('sklearn.preprocessing', 'StandardScaler'),
        ('sklearn.metrics', 'accuracy_score'),
        
        # 딥러닝
        ('torch', None),
        ('torch.nn', 'nn'),
        ('torch.nn.functional', 'F'),
        ('torch.optim', 'optim'),
        ('torch.utils.data', 'DataLoader'),
        
        # 시각화
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('plotly.graph_objects', 'go'),
        ('plotly.express', 'px'),
        
        # 웹 및 유틸리티
        ('requests', None),
        ('tqdm', None),
        
        # 표준 라이브러리
        ('datetime', 'datetime'),
        ('collections', 'deque'),
        ('logging', None),
        ('json', None),
        ('os', None),
        ('sys', None),
        ('time', None),
        ('pickle', None),
        ('typing', 'Dict'),
        ('multiprocessing', None),
    ]
    
    success_count = 0
    fail_count = 0
    
    for module_info in test_modules:
        if len(module_info) == 2:
            module_name, import_as = module_info
            try:
                if import_as:
                    if '.' in import_as:
                        # 특정 클래스/함수 import
                        exec(f"from {module_name} import {import_as.split('.')[-1]}")
                    else:
                        # 별칭 import
                        exec(f"import {module_name} as {import_as}")
                else:
                    exec(f"import {module_name}")
                print(f"OK {module_name}")
                success_count += 1
            except ImportError as e:
                print(f"FAIL {module_name}: {e}")
                fail_count += 1
    
    print(f"\n테스트 결과:")
    print(f"성공: {success_count}개")
    print(f"실패: {fail_count}개")
    
    if fail_count == 0:
        print("\n모든 모듈이 정상적으로 설치되었습니다!")
        print("2025CoinTrader 프로젝트를 실행할 수 있습니다.")
    else:
        print(f"\n{fail_count}개 모듈 설치가 필요합니다.")
        print("pip install 명령어로 누락된 모듈을 설치해주세요.")
    
    return fail_count == 0

def test_torch_gpu():
    """PyTorch GPU 지원 확인"""
    print("\nPyTorch GPU 지원 확인...")
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA를 사용할 수 없습니다. CPU에서 실행됩니다.")
            print("GPU 가속을 위해서는:")
            print("1. NVIDIA GPU 드라이버 최신 버전 설치")
            print("2. CUDA Toolkit 11.8 설치")
    except ImportError:
        print("PyTorch가 설치되지 않았습니다.")

def test_config_loading():
    """config.py 로딩 테스트"""
    print("\nConfig 파일 테스트...")
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'Data_maker'))
        from config import Config
        config = Config()
        print("config.py 로딩 성공")
        print(f"대상 코인 수: {len(config.TARGET_COINS)}")
        print(f"타임프레임 가중치 설정: {bool(config.TIMEFRAME_WEIGHTS)}")
    except Exception as e:
        print(f"config.py 로딩 실패: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("2025CoinTrader 설치 확인 테스트")
    print("=" * 60)
    
    # 모듈 import 테스트
    success = test_imports()
    
    # PyTorch GPU 테스트
    test_torch_gpu()
    
    # Config 테스트
    test_config_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("설치 완료! 이제 다음 명령어로 프로젝트를 시작할 수 있습니다:")
        print("   python Data_maker/run_data_collection.py")
        print("   python Model_maker/Training_v04.py")
        print("   python Upbit_Trader/backtest_ada_optimized.py")
    else:
        print("일부 모듈이 누락되었습니다. 위의 pip install 명령어를 실행해주세요.")
    print("=" * 60)