#!/usr/bin/env python
"""
2025CoinTrader 프로젝트 필수 모듈 설치 스크립트
모든 필요한 패키지를 자동으로 설치합니다.
"""

import subprocess
import sys
import pkg_resources
from packaging import version
import importlib.util

class ModuleInstaller:
    def __init__(self):
        self.required_packages = {
            # 핵심 데이터 처리
            'pandas': '1.5.0',
            'numpy': '1.21.0', 
            'scikit-learn': '1.1.0',
            
            # 딥러닝 및 AI
            'torch': '1.12.0',
            'torchvision': '0.13.0',
            'torchaudio': '0.12.0',
            
            # 시각화
            'matplotlib': '3.5.0',
            'seaborn': '0.11.0',
            'plotly': '5.0.0',
            
            # 웹 요청 및 설정
            'requests': '2.28.0',
            'python-dotenv': '0.19.0',
            
            # 추가 도구
            'tqdm': '4.62.0',
            'imbalanced-learn': '0.8.0',
            'typing-extensions': '4.0.0',
            'argparse': '1.1',  # 기본 모듈이지만 확인용
        }
        
        self.installed = []
        self.failed = []
        
    def check_package(self, package_name, min_version):
        """패키지 설치 여부 및 버전 확인"""
        try:
            installed_version = pkg_resources.get_distribution(package_name).version
            if version.parse(installed_version) >= version.parse(min_version):
                print(f"✅ {package_name} {installed_version} (요구사항: >={min_version})")
                return True
            else:
                print(f"⚠️  {package_name} {installed_version} < {min_version} (업그레이드 필요)")
                return False
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package_name} 미설치")
            return False
    
    def install_package(self, package_name, min_version=None):
        """개별 패키지 설치"""
        try:
            if min_version:
                install_name = f"{package_name}>={min_version}"
            else:
                install_name = package_name
                
            print(f"🔄 {package_name} 설치 중...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
            print(f"✅ {package_name} 설치 완료")
            self.installed.append(package_name)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {package_name} 설치 실패: {e}")
            self.failed.append(package_name)
            return False
    
    def install_pytorch_with_cuda(self):
        """PyTorch CUDA 버전 설치 (GPU 가속용)"""
        print("\n🔥 PyTorch CUDA 버전 설치 시도...")
        try:
            # CUDA 11.8 버전용 PyTorch 설치
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            print("✅ PyTorch CUDA 버전 설치 완료")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  CUDA 버전 설치 실패, CPU 버전으로 폴백...")
            return self.install_package('torch') and \
                   self.install_package('torchvision') and \
                   self.install_package('torchaudio')
    
    def check_special_imports(self):
        """특수 import 확인"""
        special_checks = {
            'collections.deque': 'collections',
            'datetime.datetime': 'datetime', 
            'typing.Dict': 'typing',
            'sklearn.preprocessing': 'scikit-learn',
            'sklearn.metrics': 'scikit-learn',
        }
        
        print("\n🔍 특수 모듈 import 확인...")
        for import_path, package in special_checks.items():
            try:
                module_name = import_path.split('.')[0]
                importlib.import_module(module_name)
                print(f"✅ {import_path} 사용 가능")
            except ImportError:
                print(f"❌ {import_path} 사용 불가 - {package} 설치 필요")
    
    def run_installation(self):
        """전체 설치 프로세스 실행"""
        print("🚀 2025CoinTrader 필수 모듈 설치 시작\n")
        
        # 1. 현재 설치된 패키지 확인
        print("📋 현재 설치된 패키지 확인...")
        needs_install = []
        
        for package, min_ver in self.required_packages.items():
            if package in ['torch', 'torchvision', 'torchaudio']:
                continue  # PyTorch는 별도 처리
            if not self.check_package(package, min_ver):
                needs_install.append((package, min_ver))
        
        # 2. PyTorch 별도 설치 (CUDA 지원)
        print("\n🔥 PyTorch 설치...")
        if not self.install_pytorch_with_cuda():
            print("❌ PyTorch 설치 실패")
        
        # 3. 필요한 패키지 설치
        if needs_install:
            print(f"\n📦 {len(needs_install)}개 패키지 설치 필요...")
            for package, min_ver in needs_install:
                self.install_package(package, min_ver)
        else:
            print("\n✅ 모든 필수 패키지가 이미 설치되어 있습니다!")
        
        # 4. 특수 import 확인
        self.check_special_imports()
        
        # 5. 결과 요약
        self.print_summary()
    
    def print_summary(self):
        """설치 결과 요약"""
        print("\n" + "="*50)
        print("📊 설치 결과 요약")
        print("="*50)
        
        if self.installed:
            print(f"✅ 설치 성공: {len(self.installed)}개")
            for pkg in self.installed:
                print(f"   - {pkg}")
        
        if self.failed:
            print(f"\n❌ 설치 실패: {len(self.failed)}개")
            for pkg in self.failed:
                print(f"   - {pkg}")
            print("\n⚠️  실패한 패키지는 수동으로 설치해주세요:")
            for pkg in self.failed:
                print(f"   pip install {pkg}")
        
        if not self.failed:
            print("\n🎉 모든 패키지 설치 완료!")
            print("이제 2025CoinTrader의 모든 Python 프로그램을 실행할 수 있습니다.")
        
        print("\n🔧 추가 설정:")
        print("1. CUDA GPU 사용을 위해서는 NVIDIA GPU 드라이버 및 CUDA Toolkit 설치 필요")
        print("2. 업비트 API 키를 config.py에 설정")
        print("3. 텔레그램 봇 토큰 설정 (선택사항)")

if __name__ == "__main__":
    installer = ModuleInstaller()
    installer.run_installation()