#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📊 배치별 백테스팅 결과 분석 도구 (analyze_batch_results.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • 완료된 배치들의 백테스트 결과를 자동으로 수집
    • 각 배치별 수익률, 거래횟수, 필터된 신호 등 통계 분석
    • 전체 배치에 대한 종합 통계 생성
    • 수익/손실 배치 분류 및 성과 분석

🔧 핵심 기능:
    1. 배치 폴더 자동 스캔
    2. 백테스트 로그 파일에서 결과 추출
    3. 배치별 상세 결과 출력
    4. 전체 통계 및 성과 분석
    5. 결과를 파일로 저장

📋 사용 방법:
    python analyze_batch_results.py [결과폴더경로]
    
    예시:
    python analyze_batch_results.py RESULT/20250806_014751

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import re
import glob
from datetime import datetime
import pandas as pd

class BatchResultAnalyzer:
    """배치별 백테스팅 결과 분석기"""
    
    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.timestamp = os.path.basename(result_dir)
        self.batch_results = []
        
    def extract_backtest_results(self, batch_dir, batch_num):
        """백테스트 결과 추출"""
        try:
            # 백테스트 로그 파일 찾기
            log_pattern = os.path.join(batch_dir, f'backtest_btc_optimized_{self.timestamp}.log')
            log_files = glob.glob(log_pattern)
            
            if not log_files:
                return None
            
            log_file = log_files[0]
            
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                # 파일의 끝부분에서 "BTC OPTIMIZED BACKTEST RESULTS" 섹션 찾기
                results_section_start = -1
                for i in range(len(lines) - 1, -1, -1):
                    if "BTC OPTIMIZED BACKTEST RESULTS" in lines[i]:
                        results_section_start = i
                        break
                
                if results_section_start == -1:
                    return None
                
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
                    'batch_num': batch_num,
                    'initial_capital': initial_capital,
                    'final_value': final_value,
                    'total_return': total_return,
                    'total_trades': total_trades,
                    'filtered_signals': filtered_signals
                }
                    
        except Exception as e:
            print(f"배치 {batch_num} 결과 추출 실패: {str(e)}")
            return None
    
    def scan_batch_directories(self):
        """배치 디렉토리 스캔"""
        batch_dirs = []
        
        # BATCH1, BATCH2, ... 형태의 디렉토리 찾기
        for item in os.listdir(self.result_dir):
            item_path = os.path.join(self.result_dir, item)
            if os.path.isdir(item_path) and item.startswith('BATCH'):
                try:
                    batch_num = int(item.replace('BATCH', ''))
                    batch_dirs.append((batch_num, item_path))
                except ValueError:
                    continue
        
        # 배치 번호 순으로 정렬
        batch_dirs.sort(key=lambda x: x[0])
        
        return batch_dirs
    
    def analyze_results(self):
        """결과 분석"""
        print("="*80)
        print("📊 배치별 백테스팅 결과 분석")
        print("="*80)
        print(f"분석 대상: {self.result_dir}")
        print(f"타임스탬프: {self.timestamp}")
        print()
        
        # 배치 디렉토리 스캔
        batch_dirs = self.scan_batch_directories()
        
        if not batch_dirs:
            print("❌ 배치 디렉토리를 찾을 수 없습니다.")
            return
        
        print(f"발견된 배치 수: {len(batch_dirs)}개")
        print()
        
        # 각 배치 결과 추출
        total_trades = 0
        total_signals = 0
        total_return_sum = 0
        profitable_batches = 0
        loss_batches = 0
        successful_batches = 0
        
        for batch_num, batch_dir in batch_dirs:
            result = self.extract_backtest_results(batch_dir, batch_num)
            
            if result:
                self.batch_results.append(result)
                successful_batches += 1
                
                print(f"[BATCH{batch_num:2d}] 수익률: {result['total_return']:6.2f}% | "
                      f"거래: {result['total_trades']:3d}회 | "
                      f"신호: {result['filtered_signals']:4d}개 | "
                      f"자산: {result['final_value']:,} KRW")
                
                total_trades += result['total_trades']
                total_signals += result['filtered_signals']
                total_return_sum += result['total_return']
                
                # 수익/손실 배치 구분
                if result['total_return'] > 0:
                    profitable_batches += 1
                elif result['total_return'] < 0:
                    loss_batches += 1
            else:
                print(f"[BATCH{batch_num:2d}] ❌ 결과 없음")
        
        print()
        print("="*60)
        print("📈 전체 배치 통계")
        print("="*60)
        print(f"총 배치 수: {len(batch_dirs)}개")
        print(f"성공한 배치: {successful_batches}개")
        
        if successful_batches > 0:
            avg_return = total_return_sum / successful_batches
            avg_trades = total_trades / successful_batches
            print(f"평균 수익률: {avg_return:.2f}%")
            print(f"총 거래횟수: {total_trades:,}회")
            print(f"배치당 평균 거래: {avg_trades:.1f}회")
            print(f"총 필터된 신호: {total_signals:,}개")
        
        print(f"수익 배치: {profitable_batches}개 ({(profitable_batches/len(batch_dirs))*100:.1f}%)")
        print(f"손실 배치: {loss_batches}개 ({(loss_batches/len(batch_dirs))*100:.1f}%)")
        print("="*80)
        
        # 결과를 파일로 저장
        self.save_results()
    
    def save_results(self):
        """결과를 파일로 저장"""
        if not self.batch_results:
            return
        
        # CSV 파일로 저장
        df = pd.DataFrame(self.batch_results)
        csv_file = os.path.join(self.result_dir, f'batch_analysis_{self.timestamp}.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 요약 리포트 파일로 저장
        report_file = os.path.join(self.result_dir, f'batch_analysis_report_{self.timestamp}.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("📊 배치별 백테스팅 결과 분석 리포트\n")
            f.write("="*80 + "\n")
            f.write(f"분석 대상: {self.result_dir}\n")
            f.write(f"타임스탬프: {self.timestamp}\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # 배치별 결과
            f.write("배치별 상세 결과:\n")
            f.write("-" * 80 + "\n")
            
            for result in self.batch_results:
                f.write(f"BATCH{result['batch_num']:2d} | "
                       f"수익률: {result['total_return']:6.2f}% | "
                       f"거래: {result['total_trades']:3d}회 | "
                       f"신호: {result['filtered_signals']:4d}개 | "
                       f"자산: {result['final_value']:,} KRW\n")
            
            f.write("\n")
            
            # 통계
            total_trades = sum(r['total_trades'] for r in self.batch_results)
            total_signals = sum(r['filtered_signals'] for r in self.batch_results)
            total_return_sum = sum(r['total_return'] for r in self.batch_results)
            profitable_batches = sum(1 for r in self.batch_results if r['total_return'] > 0)
            loss_batches = sum(1 for r in self.batch_results if r['total_return'] < 0)
            
            f.write("전체 통계:\n")
            f.write("-" * 40 + "\n")
            f.write(f"총 배치 수: {len(self.batch_results)}개\n")
            f.write(f"평균 수익률: {total_return_sum/len(self.batch_results):.2f}%\n")
            f.write(f"총 거래횟수: {total_trades:,}회\n")
            f.write(f"배치당 평균 거래: {total_trades/len(self.batch_results):.1f}회\n")
            f.write(f"총 필터된 신호: {total_signals:,}개\n")
            f.write(f"수익 배치: {profitable_batches}개 ({(profitable_batches/len(self.batch_results))*100:.1f}%)\n")
            f.write(f"손실 배치: {loss_batches}개 ({(loss_batches/len(self.batch_results))*100:.1f}%)\n")
        
        print(f"\n💾 결과 파일 저장 완료:")
        print(f"  - CSV: {csv_file}")
        print(f"  - 리포트: {report_file}")

def main():
    """메인 함수"""
    if len(sys.argv) != 2:
        print("사용법: python analyze_batch_results.py [결과폴더경로]")
        print("예시: python analyze_batch_results.py RESULT/20250806_014751")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    
    if not os.path.exists(result_dir):
        print(f"❌ 결과 폴더를 찾을 수 없습니다: {result_dir}")
        sys.exit(1)
    
    analyzer = BatchResultAnalyzer(result_dir)
    analyzer.analyze_results()

if __name__ == "__main__":
    main() 