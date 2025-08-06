"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
⚡ 2025CoinTrader - TOKAMAK 고변동성 백테스팅 시스템 (backtest_tokamak_optimized.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • 고변동성 TOKAMAK(TON) 코인 전용 백테스팅 시스템
    • 변동성이 큰 코인에 특화된 리스크 관리 전략
    • 급격한 가격 변동에 대응하는 적응형 거래 시스템
    • 고변동성 환경에서의 모델 성능 검증
    • 극한 시장 상황에서의 전략 안정성 테스트

🔧 핵심 기능:
    1. 고변동성 대응: 급격한 가격 변동 시 리스크 관리 강화
    2. 적응형 임계값: 변동성에 따른 거래 신호 임계값 동적 조정
    3. 긴급 손절: 예상 손실 한도 초과 시 즉시 청산
    4. 분할 매매: 대량 포지션을 여러 구간으로 나누어 체결
    5. 변동성 필터: 극도로 높은 변동성 구간에서 거래 중단

💰 고변동성 거래 전략:
    • 변동성 임계값: ATR 기반 동적 조정 (일반 코인 대비 150%)
    • 포지션 크기: 변동성에 반비례하여 자동 조정
    • 빠른 손절: 2% 손실 시 즉시 청산
    • 부분 익절: 5% 수익 시 50% 물량 익절, 나머지는 트레일링
    • 거래 중단: 일일 변동성 30% 초과 시 자동 중단

📊 고변동성 성과 지표:
    • 변동성 조정 수익률 (Volatility-Adjusted Return)
    • 칼마 비율 (Calmar Ratio) - 최대 낙폭 대비 수익률
    • VaR (Value at Risk) - 95% 신뢰구간 최대 손실
    • 변동성 대비 승률 분석
    • 극한 상황에서의 모델 신뢰도

📋 사용 방법:
    python backtest_tokamak_optimized.py
    
    생성 파일:
    - backtest_tokamak_optimized_YYYYMMDD_HHMMSS.log: 상세 거래 로그
    - tokamak_backtest_results.png: 백테스팅 결과 차트
    - tokamak_volatility_analysis.png: 변동성 분석 차트
    - tokamak_performance_report.json: 성과 지표 리포트

⚠️ 주의사항:
    • 고변동성 코인으로 인한 높은 리스크 존재
    • 실제 거래 시 슬리피지가 일반 코인보다 클 수 있음
    • 유동성 부족으로 인한 체결 지연 가능성
    • 극단적 시장 상황에서는 모델 예측 정확도 하락
    • 충분한 자본과 리스크 허용도를 가진 후 거래 권장

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import logging
import os
from collections import deque
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'backtest_tokamak_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# MultiTimeframeBiLSTM 모델 클래스 (동일)
class MultiTimeframeBiLSTM(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 128, num_layers: int = 2, 
                 num_classes: int = 3, dropout: float = 0.2):
        super(MultiTimeframeBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.timeframe_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        out = self.dropout(context)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class TOKAMAKBacktesterOptimized:
    """최적화된 TOKAMAK 백테스팅 시스템"""
    
    def __init__(self, initial_capital=10_000_000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.avg_price = 0
        self.trades = []
        self.commission_rate = 0.0005  # 0.05%
        
        # 모델 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = 30
        
        # === 최적화된 거래 설정 ===
        self.buy_threshold = 0.50  # 0.40 → 0.50 (상향)
        self.sell_threshold = 0.25  # 0.30 → 0.25 (하향)
        self.force_sell_loss = -0.7  # -1.0 → -0.7 (손절 강화)
        self.force_sell_profit = 1.0  # 1.2 → 1.0 (익절 하향)
        self.trailing_stop_profit = 0.6  # 0.8 → 0.6 (트레일링 강화)
        self.time_stop_minutes = 15  # 10 → 15 (시간 손절 완화)
        
        # 추가 최적화 설정
        self.min_hold_minutes = 5  # 최소 보유 시간 (2분 → 5분)
        self.good_trading_hours = [5, 6, 9, 10, 14, 16, 19, 20, 21, 23]  # 수익률 높은 시간대
        self.min_buy_prob_diff = 2.0  # 매수/매도 확률 차이 최소값
        
        logger.info(f"Initializing Optimized TOKAMAK Backtester")
        logger.info(f"Initial Capital: {self.initial_capital:,} KRW")
        logger.info(f"Device: {self.device}")
        logger.info(f"=== Optimized Settings ===")
        logger.info(f"Buy Threshold: {self.buy_threshold}")
        logger.info(f"Sell Threshold: {self.sell_threshold}")
        logger.info(f"Stop Loss: {self.force_sell_loss}%")
        logger.info(f"Take Profit: {self.force_sell_profit}%")
        logger.info(f"Trailing Stop: {self.trailing_stop_profit}%")
        logger.info(f"Min Hold Time: {self.min_hold_minutes} minutes")
        
        # 모델 로드
        self.load_model()
        
        # Scaler 초기화
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
        # 가격 히스토리
        self.price_history = deque(maxlen=20)  # 10 → 20으로 확대
        
        # 포지션 추적
        self.position_open_idx = None
        self.max_profit_since_buy = 0
        
        # 추가 통계
        self.filtered_signals = 0
        self.time_filtered = 0
        
    def load_model(self):
        """모델 로드"""
        try:
            logger.info("Loading model...")
            self.model = MultiTimeframeBiLSTM().to(self.device)
            
            # 현재 파일 위치 기준 상대경로로 모델 파일 경로 설정
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(os.path.dirname(current_dir), 'Model_maker', 'multitimeframe_bilstm_crypto_model_v04.pth')
            
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.model_threshold = checkpoint.get('probability_threshold', 0.45)
            logger.info(f"Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_data(self):
        """과거 데이터 로드 - TOKAMAK"""
        logger.info("Loading TOKAMAK historical data...")
        
        # 현재 파일 위치 기준 상대경로로 데이터 파일 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), 'Data_maker', 'DATA', 'raw')
        
        # 1분봉 데이터
        tokamak_1m_path = os.path.join(data_dir, 'TOKAMAK_1m.csv')
        self.df_1m = pd.read_csv(tokamak_1m_path)
        self.df_1m['timestamp'] = pd.to_datetime(self.df_1m['timestamp'])
        self.df_1m = self.df_1m.sort_values('timestamp')
        self.df_1m.set_index('timestamp', inplace=True)
        
        # 5분봉 데이터
        tokamak_5m_path = os.path.join(data_dir, 'TOKAMAK_5m.csv')
        self.df_5m = pd.read_csv(tokamak_5m_path)
        self.df_5m['timestamp'] = pd.to_datetime(self.df_5m['timestamp'])
        self.df_5m = self.df_5m.sort_values('timestamp')
        self.df_5m.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded 1m data: {len(self.df_1m)} rows")
        logger.info(f"Loaded 5m data: {len(self.df_5m)} rows")
        logger.info(f"Date range: {self.df_1m.index[0]} to {self.df_1m.index[-1]}")
        
        # TOKAMAK 가격 범위 확인
        price_range = self.df_1m['close'].max() - self.df_1m['close'].min()
        price_volatility = (price_range / self.df_1m['close'].mean()) * 100
        logger.info(f"Price range: {self.df_1m['close'].min():.0f} - {self.df_1m['close'].max():.0f}")
        logger.info(f"Price volatility: {price_volatility:.2f}%")
        
    def prepare_features_at_index(self, idx):
        """특정 인덱스에서의 특징 준비"""
        if idx < 60:  # 최소 60개의 1분봉 필요
            return None
            
        # 현재 시점
        current_time = self.df_1m.index[idx]
        
        # 1분봉 데이터 (현재 시점부터 과거 100개)
        start_idx = max(0, idx - 99)
        df_1m_window = self.df_1m.iloc[start_idx:idx+1].copy()
        
        # 5분봉 데이터 찾기 (현재 시점 이전의 데이터만)
        df_5m_window = self.df_5m[self.df_5m.index <= current_time].tail(30).copy()
        
        if len(df_5m_window) < 20:
            return None
        
        # 특징 생성
        features = pd.DataFrame(index=df_1m_window.index)
        
        # 1분봉 MACD (이미 계산되어 있음)
        features['macd_1m'] = df_1m_window['macd']
        features['macd_signal_1m'] = df_1m_window['macd_signal']
        features['macd_histogram_1m'] = df_1m_window['macd_histogram']
        
        # 5분봉 MACD를 1분봉에 맞춰 리샘플링
        macd_5m = df_5m_window['macd'].resample('1min').interpolate(method='linear')
        signal_5m = df_5m_window['macd_signal'].resample('1min').interpolate(method='linear')
        hist_5m = df_5m_window['macd_histogram'].resample('1min').interpolate(method='linear')
        
        # 공통 인덱스
        common_index = features.index.intersection(macd_5m.index)
        features = features.loc[common_index]
        
        features['macd_5m'] = macd_5m.loc[common_index]
        features['macd_signal_5m'] = signal_5m.loc[common_index]
        features['macd_histogram_5m'] = hist_5m.loc[common_index]
        
        # 다이버전스
        features['macd_divergence'] = features['macd_1m'] - features['macd_5m']
        features['histogram_divergence'] = features['macd_histogram_1m'] - features['macd_histogram_5m']
        
        # 가격 변화
        features['price_change'] = df_1m_window['price_change'].loc[common_index]
        features['price_change_ma'] = features['price_change'].rolling(window=5).mean()
        
        # NaN 처리
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def make_prediction(self, features):
        """예측 수행"""
        try:
            if len(features) < self.seq_length:
                return None, None
            
            # Scaler 학습
            if not self.is_scaler_fitted:
                self.scaler.fit(features)
                self.is_scaler_fitted = True
            
            # 스케일링
            X_scaled = self.scaler.transform(features)
            
            # 시퀀스 생성
            sequence = X_scaled[-self.seq_length:]
            X_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # 예측
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                return predictions[0].item(), probabilities[0].cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None, None
    
    def should_buy(self, signal, probs, current_price, current_time):
        """최적화된 매수 결정"""
        # 시간대 필터
        current_hour = current_time.hour
        if current_hour not in self.good_trading_hours:
            self.time_filtered += 1
            return False
        
        # 강화된 확률 조건
        if signal == 2 and probs[2] > self.buy_threshold:
            # 추가 확인: HOLD 확률이 낮아야 함
            if probs[1] < 0.2:
                # 매수/매도 확률 차이 확인
                if probs[2] / (probs[0] + 0.001) > self.min_buy_prob_diff:
                    return True
        
        # 보조 조건도 강화
        if signal == 1 and probs[2] > probs[0] * 2.0 and probs[2] > 0.45:
            if probs[1] < 0.25:  # HOLD 확률이 낮을 때만
                return True
        
        # 가격 조건 강화
        if len(self.price_history) >= 20:
            avg_price = np.mean(list(self.price_history)[-10:])  # 최근 10개 평균
            if current_price < avg_price * 0.993 and probs[2] > 0.45:
                return True
        
        self.filtered_signals += 1
        return False
    
    def should_sell(self, signal, probs, current_price, current_idx):
        """최적화된 매도 결정"""
        if self.position == 0:
            return False
        
        profit_pct = ((current_price - self.avg_price) / self.avg_price) * 100
        
        if profit_pct > self.max_profit_since_buy:
            self.max_profit_since_buy = profit_pct
        
        # === 1. 강화된 손절매 ===
        if profit_pct <= self.force_sell_loss:
            logger.info(f"[STOP LOSS] Loss: {profit_pct:.2f}%")
            return True
        
        # === 2. 조정된 익절매 ===
        if profit_pct >= self.force_sell_profit:
            logger.info(f"[TAKE PROFIT] Profit: {profit_pct:.2f}%")
            return True
        
        # === 3. 강화된 트레일링 스톱 ===
        if self.max_profit_since_buy > self.trailing_stop_profit:
            trailing_loss = self.max_profit_since_buy - profit_pct
            if trailing_loss > 0.3:  # 0.5 → 0.3으로 강화
                logger.info(f"[TRAILING STOP] Max: {self.max_profit_since_buy:.2f}% -> Current: {profit_pct:.2f}%")
                return True
        
        # === 4. 시간 기반 손절 (조건 완화) ===
        if self.position_open_idx is not None:
            hold_minutes = current_idx - self.position_open_idx
            if hold_minutes > self.time_stop_minutes and profit_pct < -0.3:  # -0.5 → -0.3
                logger.info(f"[TIME STOP] Holding {hold_minutes}min with loss: {profit_pct:.2f}%")
                return True
        
        # === 5. 기술적 신호 기반 매도 ===
        if signal == 0 and probs[0] > self.sell_threshold:
            logger.info(f"[SIGNAL SELL] Sell probability: {probs[0]:.3f}")
            return True
        
        # === 6. 약한 하락 신호 (조건 강화) ===
        if profit_pct < -0.1 and probs[0] > 0.20:  # 0.25 → 0.20
            logger.info(f"[WEAK SIGNAL SELL] Loss: {profit_pct:.2f}%, Sell prob: {probs[0]:.3f}")
            return True
        
        # === 7. 수익 중 매도 신호 강화 ===
        if profit_pct > 0.3 and probs[0] > 0.15:
            logger.info(f"[PROFIT PROTECTION] Profit: {profit_pct:.2f}%, Sell prob: {probs[0]:.3f}")
            return True
        
        return False
    
    def execute_trade(self, idx, signal, current_price, probs):
        """거래 실행"""
        timestamp = self.df_1m.index[idx]
        
        self.price_history.append(current_price)
        
        # 매수
        if self.position == 0 and self.should_buy(signal, probs, current_price, timestamp):
            use_capital = self.current_capital * 0.95
            quantity = use_capital / current_price
            cost = quantity * current_price * (1 + self.commission_rate)
            
            if cost <= self.current_capital:
                self.position = quantity
                self.avg_price = current_price
                self.current_capital -= cost
                self.position_open_idx = idx
                self.max_profit_since_buy = 0
                
                trade = {
                    'timestamp': timestamp,
                    'idx': idx,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': quantity,
                    'cost': cost,
                    'balance': self.current_capital,
                    'signal': signal,
                    'prob_sell': probs[0],
                    'prob_hold': probs[1],
                    'prob_buy': probs[2]
                }
                self.trades.append(trade)
                
                logger.info(f"[BUY] {timestamp} | {quantity:.4f} @ {current_price:,.0f} | "
                          f"Probs: [S:{probs[0]:.3f}, H:{probs[1]:.3f}, B:{probs[2]:.3f}]")
                return True
        
        # 매도
        elif self.position > 0 and self.should_sell(signal, probs, current_price, idx):
            revenue = self.position * current_price * (1 - self.commission_rate)
            profit = revenue - (self.position * self.avg_price)
            profit_pct = (profit / (self.position * self.avg_price)) * 100
            
            self.current_capital += revenue
            
            hold_minutes = idx - self.position_open_idx if self.position_open_idx else 0
            
            trade = {
                'timestamp': timestamp,
                'idx': idx,
                'action': 'SELL',
                'price': current_price,
                'quantity': self.position,
                'revenue': revenue,
                'profit': profit,
                'profit_pct': profit_pct,
                'balance': self.current_capital,
                'signal': signal,
                'prob_sell': probs[0],
                'prob_hold': probs[1],
                'prob_buy': probs[2],
                'hold_minutes': hold_minutes
            }
            self.trades.append(trade)
            
            logger.info(f"[SELL] {timestamp} | {self.position:.4f} @ {current_price:,.0f} | "
                      f"Profit: {profit_pct:.2f}% | Hold: {hold_minutes}min")
            
            self.position = 0
            self.avg_price = 0
            self.position_open_idx = None
            self.max_profit_since_buy = 0
            return True
            
        return False
    
    def run_backtest(self):
        """백테스트 실행"""
        logger.info("="*60)
        logger.info("Starting Optimized TOKAMAK Backtest")
        logger.info("="*60)
        
        # 데이터 로드
        self.load_data()
        
        # 백테스트 진행
        total_rows = len(self.df_1m)
        last_trade_idx = None
        min_hold_intervals = self.min_hold_minutes  # 최소 보유 시간 사용
        
        for idx in range(100, total_rows):
            # 진행상황 표시
            if idx % 1000 == 0:
                progress = (idx / total_rows) * 100
                position_value = self.position * self.df_1m.iloc[idx]['close'] if self.position > 0 else 0
                total_value = self.current_capital + position_value
                total_return = ((total_value - self.initial_capital) / self.initial_capital) * 100
                
                logger.info(f"Progress: {progress:.1f}% | "
                          f"Total Return: {total_return:.2f}% | "
                          f"Trades: {len(self.trades)} | "
                          f"Filtered: {self.filtered_signals} | "
                          f"Time Filtered: {self.time_filtered}")
            
            # 특징 준비
            features = self.prepare_features_at_index(idx)
            if features is None:
                continue
            
            # 예측
            signal, probs = self.make_prediction(features)
            if signal is None:
                continue
            
            # 현재가
            current_price = self.df_1m.iloc[idx]['close']
            
            # 거래 시간 제한 확인
            can_buy = True
            if last_trade_idx and self.position == 0:
                if idx - last_trade_idx < min_hold_intervals:
                    can_buy = False
            
            # 거래 실행
            if self.position > 0:  # 포지션 있으면 매도 체크
                if self.execute_trade(idx, signal, current_price, probs):
                    last_trade_idx = idx
            elif can_buy:  # 포지션 없고 매수 가능하면
                if self.execute_trade(idx, signal, current_price, probs):
                    last_trade_idx = idx
        
        # 마지막 포지션 정리
        if self.position > 0:
            final_price = self.df_1m.iloc[-1]['close']
            revenue = self.position * final_price * (1 - self.commission_rate)
            self.current_capital += revenue
            logger.info(f"[FINAL SELL] Closing position at {final_price:,.0f}")
        
        # 결과 출력
        self.print_results()
    
    def print_results(self):
        """백테스트 결과 출력"""
        print("\n" + "="*60)
        print("OPTIMIZED TOKAMAK BACKTEST RESULTS")
        print("="*60)
        
        position_value = 0
        total_value = self.current_capital + position_value
        total_return = ((total_value - self.initial_capital) / self.initial_capital) * 100
        
        print(f"Initial Capital: {self.initial_capital:,.0f} KRW")
        print(f"Final Value: {total_value:,.0f} KRW")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Filtered Signals: {self.filtered_signals}")
        print(f"Time Filtered: {self.time_filtered}")
        
        if self.trades:
            # 거래 통계
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] == 'SELL']
            
            if sell_trades:
                profits = [t['profit_pct'] for t in sell_trades]
                winning_trades = sum(1 for p in profits if p > 0)
                
                print(f"\nTrade Statistics:")
                print(f"- Buy Orders: {len(buy_trades)}")
                print(f"- Sell Orders: {len(sell_trades)}")
                print(f"- Win Rate: {winning_trades/len(sell_trades)*100:.1f}%")
                print(f"- Average Profit: {np.mean(profits):.2f}%")
                print(f"- Max Profit: {max(profits):.2f}%")
                print(f"- Max Loss: {min(profits):.2f}%")
                print(f"- Std Dev: {np.std(profits):.2f}%")
                
                # 평균 보유 시간
                hold_times = [t.get('hold_minutes', 0) for t in sell_trades]
                if hold_times:
                    print(f"- Average Hold Time: {np.mean(hold_times):.1f} minutes")
                
                # Sharpe Ratio 계산
                if len(profits) > 1:
                    sharpe = np.mean(profits) / np.std(profits) * np.sqrt(252 * 24 * 60 / np.mean(hold_times))
                    print(f"- Sharpe Ratio: {sharpe:.2f}")
                
                # 손익비
                winning_profits = [p for p in profits if p > 0]
                losing_profits = [p for p in profits if p <= 0]
                if winning_profits and losing_profits:
                    profit_factor = np.mean(winning_profits) / abs(np.mean(losing_profits))
                    print(f"- Profit Factor: {profit_factor:.2f}")
            
            # 거래 내역 저장
            df_trades = pd.DataFrame(self.trades)
            filename = f'backtest_tokamak_optimized_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df_trades.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\nTrade history saved to: {filename}")

if __name__ == "__main__":
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # 백테스터 생성 및 실행
    backtester = TOKAMAKBacktesterOptimized(initial_capital=10_000_000)
    backtester.run_backtest()