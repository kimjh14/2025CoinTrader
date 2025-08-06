"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📊 2025CoinTrader - 데이터 품질 분석 및 시각화 모듈 (data_analyzer.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • 수집된 암호화폐 데이터의 품질 분석 및 검증
    • 기술적 지표 및 가격 패턴 시각화
    • 다양한 코인 간 상관관계 분석
    • 레이블 분포 및 클래스 균형성 분석
    • 데이터 이상치 탐지 및 처리 방안 제시

🔧 핵심 기능:
    1. 데이터 품질 분석: 결측값, 이상치, 데이터 일관성 검증
    2. 시각화: 가격 차트, 기술적 지표, 상관관계 히트맵
    3. 통계 분석: 변동성, 수익률 분포, 거래량 패턴 분석
    4. 비교 분석: 코인별 성과 및 위험도 비교
    5. 품질 리포트: JSON 형태의 종합 데이터 품질 보고서 생성

📈 생성되는 분석 결과:
    • {코인명}_price_chart.html: 인터랙티브 가격 및 지표 차트
    • {코인명}_correlation_heatmap.png: 특성 간 상관관계 히트맵
    • {코인명}_label_distribution.png: 예측 레이블 분포 차트
    • coin_volatility_comparison.png: 코인별 변동성 비교
    • data_quality_analysis.json: 종합 데이터 품질 리포트

📋 사용 방법:
    analyzer = CryptoDataAnalyzer()
    analyzer.analyze_coin_data('BTC_training_dataset.csv', 'BTC')
    analyzer.create_comparison_analysis(['BTC', 'ETH', 'SOL'])

⚠️ 주의사항:
    • 대용량 데이터 처리 시 메모리 사용량 주의
    • 시각화 파일은 자동으로 data/processed/ 폴더에 저장
    • 한글 폰트 설정으로 인한 시스템별 호환성 고려

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용 한글 폰트
else:
    plt.rcParams['font.family'] = 'AppleGothic'  # Mac용 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

class CryptoDataAnalyzer:
    """가상화폐 데이터 분석 및 시각화 클래스"""
    
    def __init__(self, data_dir=None):
        """초기화"""
        if data_dir is None:
            # 환경 변수에서 결과 폴더 확인
            result_folder = os.environ.get('RESULT_FOLDER_PATH')
            if result_folder and os.path.exists(result_folder):
                # 결과 폴더가 지정된 경우 해당 폴더 내 DATA 사용
                data_dir = os.path.join(result_folder, 'DATA')
            else:
                # 기본 경로 사용
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_dir = os.path.join(current_dir, 'DATA')
        
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        # 분석 대상 코인들
        self.target_coins = ['BTC', 'DOGE', 'SOL', 'XRP', 'SUI', 'HBAR', 'APT']
        
        # 색상 팔레트
        self.colors = {
            'BTC': '#F7931A',
            'DOGE': '#C2A633', 
            'SOL': '#00FFA3',
            'XRP': '#23292F',
            'SUI': '#6FBBC0',
            'HBAR': '#000000',
            'APT': '#00D4AA'
        }
        
    def load_training_data(self, coin: str) -> pd.DataFrame:
        """훈련 데이터 로드"""
        filepath = f"{self.processed_dir}/{coin}_training_dataset.csv"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            print(f"파일이 존재하지 않습니다: {filepath}")
            return pd.DataFrame()
    
    def analyze_price_patterns(self, coin: str):
        """가격 패턴 분석"""
        df = self.load_training_data(coin)
        if df.empty:
            return
        
        print(f"\n=== {coin} 가격 패턴 분석 ===")
        
        # 기본 통계
        print(f"데이터 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        print(f"총 데이터 수: {len(df):,}개")
        print(f"현재가: {df['close'].iloc[-1]:,.0f}원")
        print(f"최고가: {df['high'].max():,.0f}원")
        print(f"최저가: {df['low'].min():,.0f}원")
        print(f"평균가: {df['close'].mean():,.0f}원")
        
        # 변동성 분석
        daily_returns = df['close'].pct_change().dropna()
        print(f"일일 변동성 (표준편차): {daily_returns.std():.4f}")
        print(f"최대 상승: {daily_returns.max():.4f}")
        print(f"최대 하락: {daily_returns.min():.4f}")
        
        # 거래량 분석
        print(f"평균 거래량: {df['volume'].mean():,.0f}")
        print(f"최대 거래량: {df['volume'].max():,.0f}")
        
        return df
    
    def plot_price_chart(self, coin: str, days: int = 7):
        """가격 차트 시각화"""
        df = self.load_training_data(coin)
        if df.empty:
            return
        
        # 최근 N일 데이터만 사용
        recent_df = df.tail(days * 24 * 60)  # 1분봉 기준
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{coin} 가격 차트', '거래량', '기술적 지표'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # 캔들스틱 차트
        fig.add_trace(
            go.Candlestick(
                x=recent_df['timestamp'],
                open=recent_df['open'],
                high=recent_df['high'],
                low=recent_df['low'],
                close=recent_df['close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # 이동평균선
        fig.add_trace(
            go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['ma5'],
                name='MA5',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['ma20'],
                name='MA20',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # 거래량
        fig.add_trace(
            go.Bar(
                x=recent_df['timestamp'],
                y=recent_df['volume'],
                name='거래량',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=recent_df['timestamp'],
                y=recent_df['rsi'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # RSI 기준선
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            title=f'{coin} 가격 분석 ({days}일)',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        # 파일로 저장
        fig.write_html(f"{self.processed_dir}/{coin}_price_chart.html")
        print(f"{coin} 가격 차트 저장 완료")
        
        return fig
    
    def analyze_correlation(self, coin: str):
        """특성 간 상관관계 분석"""
        df = self.load_training_data(coin)
        if df.empty:
            return
        
        # 수치형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        # 상관관계 히트맵
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )
        plt.title(f'{coin} 특성 상관관계 분석')
        plt.tight_layout()
        plt.savefig(f"{self.processed_dir}/{coin}_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 가격과 가장 상관관계가 높은 특성들
        price_corr = correlation_matrix['close'].abs().sort_values(ascending=False)
        print(f"\n=== {coin} 가격과의 상관관계 (상위 10개) ===")
        print(price_corr.head(10))
        
        return correlation_matrix
    
    def analyze_label_distribution(self, coin: str):
        """레이블 분포 분석"""
        df = self.load_training_data(coin)
        if df.empty or 'label' not in df.columns:
            return
        
        # 레이블 분포
        label_counts = df['label'].value_counts()
        label_names = {0: '상승 (Buy)', 1: '하락 (Sell)', 2: '횡보 (Hold)'}
        
        print(f"\n=== {coin} 레이블 분포 ===")
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{label_names.get(label, label)}: {count:,}개 ({percentage:.1f}%)")
        
        # 레이블 분포 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 파이 차트
        colors = ['green', 'red', 'gray']
        ax1.pie(label_counts.values, labels=[label_names.get(l, l) for l in label_counts.index], 
                autopct='%1.1f%%', colors=colors)
        ax1.set_title(f'{coin} 레이블 분포')
        
        # 막대 차트
        ax2.bar(range(len(label_counts)), label_counts.values, color=colors)
        ax2.set_xticks(range(len(label_counts)))
        ax2.set_xticklabels([label_names.get(l, l) for l in label_counts.index])
        ax2.set_title(f'{coin} 레이블 개수')
        ax2.set_ylabel('개수')
        
        plt.tight_layout()
        plt.savefig(f"{self.processed_dir}/{coin}_label_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return label_counts
    
    def analyze_technical_indicators(self, coin: str):
        """기술적 지표 분석"""
        df = self.load_training_data(coin)
        if df.empty:
            return
        
        print(f"\n=== {coin} 기술적 지표 분석 ===")
        
        # MACD 분석
        if 'macd' in df.columns:
            macd_crosses = ((df['macd'] > df['macd_signal']) & 
                           (df['macd'].shift(1) <= df['macd_signal'].shift(1))).sum()
            print(f"MACD 골든크로스 횟수: {macd_crosses}회")
        
        # RSI 분석
        if 'rsi' in df.columns:
            oversold_count = (df['rsi'] < 30).sum()
            overbought_count = (df['rsi'] > 70).sum()
            print(f"RSI 과매도 (<30) 횟수: {oversold_count}회")
            print(f"RSI 과매수 (>70) 횟수: {overbought_count}회")
        
        # 볼린저 밴드 분석
        if 'bb_upper' in df.columns:
            bb_breakout_up = (df['close'] > df['bb_upper']).sum()
            bb_breakout_down = (df['close'] < df['bb_lower']).sum()
            print(f"볼린저 밴드 상단 돌파: {bb_breakout_up}회")
            print(f"볼린저 밴드 하단 돌파: {bb_breakout_down}회")
        
        # 기술적 지표 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MACD
        if 'macd' in df.columns:
            recent_data = df.tail(100)
            axes[0, 0].plot(recent_data['timestamp'], recent_data['macd'], label='MACD')
            axes[0, 0].plot(recent_data['timestamp'], recent_data['macd_signal'], label='Signal')
            axes[0, 0].bar(recent_data['timestamp'], recent_data['macd_histogram'], alpha=0.3, label='Histogram')
            axes[0, 0].set_title('MACD')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RSI
        if 'rsi' in df.columns:
            axes[0, 1].plot(recent_data['timestamp'], recent_data['rsi'], color='purple')
            axes[0, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7)
            axes[0, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7)
            axes[0, 1].set_title('RSI')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 볼린저 밴드
        if 'bb_upper' in df.columns:
            axes[1, 0].plot(recent_data['timestamp'], recent_data['close'], label='가격')
            axes[1, 0].plot(recent_data['timestamp'], recent_data['bb_upper'], '--', alpha=0.7, label='상단')
            axes[1, 0].plot(recent_data['timestamp'], recent_data['bb_middle'], '--', alpha=0.7, label='중간')
            axes[1, 0].plot(recent_data['timestamp'], recent_data['bb_lower'], '--', alpha=0.7, label='하단')
            axes[1, 0].set_title('볼린저 밴드')
            axes[1, 0].legend()
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 거래량
        axes[1, 1].bar(recent_data['timestamp'], recent_data['volume'], alpha=0.7)
        axes[1, 1].set_title('거래량')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.processed_dir}/{coin}_technical_indicators.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_coins(self):
        """코인 간 비교 분석"""
        print("\n=== 코인 간 비교 분석 ===")
        
        comparison_data = {}
        
        for coin in self.target_coins:
            df = self.load_training_data(coin)
            if not df.empty:
                # 수익률 계산
                returns = df['close'].pct_change().dropna()
                
                comparison_data[coin] = {
                    '현재가': df['close'].iloc[-1],
                    '변동성': returns.std(),
                    '최대상승': returns.max(),
                    '최대하락': returns.min(),
                    '평균거래량': df['volume'].mean(),
                    '데이터수': len(df)
                }
        
        # 비교 테이블 생성
        comparison_df = pd.DataFrame(comparison_data).T
        print(comparison_df.round(4))
        
        # 변동성 비교 시각화
        volatilities = [comparison_data[coin]['변동성'] for coin in self.target_coins if coin in comparison_data]
        coin_names = [coin for coin in self.target_coins if coin in comparison_data]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(coin_names, volatilities, color=[self.colors.get(coin, 'blue') for coin in coin_names])
        plt.title('코인별 변동성 비교')
        plt.ylabel('변동성 (표준편차)')
        plt.xticks(rotation=45)
        
        # 값 표시
        for bar, vol in zip(bars, volatilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                    f'{vol:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.processed_dir}/coin_volatility_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return comparison_df
    
    def generate_analysis_report(self):
        """종합 분석 리포트 생성"""
        print("=== 가상화폐 데이터 분석 리포트 생성 시작 ===")
        
        report = {
            '분석_일시': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '분석_대상_코인': self.target_coins,
            '코인별_분석': {}
        }
        
        # 각 코인별 분석
        for coin in self.target_coins:
            print(f"\n{coin} 분석 중...")
            
            coin_report = {}
            
            # 가격 패턴 분석
            df = self.analyze_price_patterns(coin)
            if not df.empty:
                coin_report['기본_통계'] = {
                    '데이터_수': len(df),
                    '현재가': float(df['close'].iloc[-1]),
                    '최고가': float(df['high'].max()),
                    '최저가': float(df['low'].min()),
                    '평균가': float(df['close'].mean()),
                    '변동성': float(df['close'].pct_change().std())
                }
            
            # 레이블 분포 분석
            label_counts = self.analyze_label_distribution(coin)
            if label_counts is not None:
                coin_report['레이블_분포'] = label_counts.to_dict()
            
            # 기술적 지표 분석
            self.analyze_technical_indicators(coin)
            
            # 상관관계 분석
            corr_matrix = self.analyze_correlation(coin)
            if corr_matrix is not None:
                # 가격과 가장 상관관계가 높은 특성들
                price_corr = corr_matrix['close'].abs().sort_values(ascending=False)
                coin_report['가격_상관관계_상위5'] = price_corr.head(5).to_dict()
            
            # 가격 차트 생성
            self.plot_price_chart(coin)
            
            report['코인별_분석'][coin] = coin_report
        
        # 코인 간 비교
        comparison_df = self.compare_coins()
        if not comparison_df.empty:
            report['코인_비교'] = comparison_df.to_dict()
        
        # 리포트 저장
        report_path = f"{self.processed_dir}/analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n분석 리포트 저장 완료: {report_path}")
        
        return report

def main():
    """메인 실행 함수"""
    analyzer = CryptoDataAnalyzer()
    
    # 종합 분석 리포트 생성
    report = analyzer.generate_analysis_report()
    
    print("\n=== 분석 완료 ===")
    print("생성된 파일들:")
    print("- analysis_report.json: 종합 분석 리포트")
    print("- *_price_chart.html: 인터랙티브 가격 차트")
    print("- *_correlation_heatmap.png: 상관관계 히트맵")
    print("- *_label_distribution.png: 레이블 분포")
    print("- *_technical_indicators.png: 기술적 지표")
    print("- coin_volatility_comparison.png: 코인 변동성 비교")

if __name__ == "__main__":
    main() 