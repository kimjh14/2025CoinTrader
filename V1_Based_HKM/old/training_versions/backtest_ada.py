"""
ADA 과거 데이터 백테스팅 시스템
upbit_paper_trading_v3_improved.py 기반
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
        logging.FileHandler(f'backtest_ada_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
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

class ADABacktester:
    """ADA 백테스팅 시스템"""
    
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
        
        # 거래 설정 (V3와 동일)
        self.buy_threshold = 0.40
        self.sell_threshold = 0.30
        self.force_sell_loss = -1.0
        self.force_sell_profit = 1.2
        self.trailing_stop_profit = 0.8
        self.time_stop_minutes = 10
        
        logger.info(f"Initializing ADA Backtester")
        logger.info(f"Initial Capital: {self.initial_capital:,} KRW")
        logger.info(f"Device: {self.device}")
        
        # 모델 로드
        self.load_model()
        
        # Scaler 초기화
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
        # 가격 히스토리
        self.price_history = deque(maxlen=10)
        
        # 포지션 추적
        self.position_open_idx = None
        self.max_profit_since_buy = 0
        
    def load_model(self):
        """모델 로드"""
        try:
            logger.info("Loading model...")
            self.model = MultiTimeframeBiLSTM().to(self.device)
            
            checkpoint = torch.load('D:\\2025CoinTrader\\Model_maker\\multitimeframe_bilstm_crypto_model_v04.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.model_threshold = checkpoint.get('probability_threshold', 0.45)
            logger.info(f"Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_data(self):
        """과거 데이터 로드"""
        logger.info("Loading historical data...")
        
        # 1분봉 데이터
        self.df_1m = pd.read_csv('D:\\2025CoinTrader\\Data_maker\\DATA\\raw\\ADA_1m.csv')
        self.df_1m['timestamp'] = pd.to_datetime(self.df_1m['timestamp'])
        self.df_1m = self.df_1m.sort_values('timestamp')
        self.df_1m.set_index('timestamp', inplace=True)
        
        # 5분봉 데이터
        self.df_5m = pd.read_csv('D:\\2025CoinTrader\\Data_maker\\DATA\\raw\\ADA_5m.csv')
        self.df_5m['timestamp'] = pd.to_datetime(self.df_5m['timestamp'])
        self.df_5m = self.df_5m.sort_values('timestamp')
        self.df_5m.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded 1m data: {len(self.df_1m)} rows")
        logger.info(f"Loaded 5m data: {len(self.df_5m)} rows")
        logger.info(f"Date range: {self.df_1m.index[0]} to {self.df_1m.index[-1]}")
        
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
    
    def should_buy(self, signal, probs, current_price):
        """매수 결정"""
        if signal == 2 and probs[2] > self.buy_threshold:
            return True
        
        if signal == 1 and probs[2] > probs[0] * 1.5 and probs[2] > 0.35:
            return True
        
        if len(self.price_history) >= 10:
            avg_price = np.mean(self.price_history)
            if current_price < avg_price * 0.995 and probs[2] > 0.30:
                return True
        
        return False
    
    def should_sell(self, signal, probs, current_price, current_idx):
        """매도 결정"""
        if self.position == 0:
            return False
        
        profit_pct = ((current_price - self.avg_price) / self.avg_price) * 100
        
        if profit_pct > self.max_profit_since_buy:
            self.max_profit_since_buy = profit_pct
        
        # 손절매
        if profit_pct <= self.force_sell_loss:
            logger.info(f"[STOP LOSS] Loss: {profit_pct:.2f}%")
            return True
        
        # 익절매
        if profit_pct >= self.force_sell_profit:
            logger.info(f"[TAKE PROFIT] Profit: {profit_pct:.2f}%")
            return True
        
        # 트레일링 스톱
        if self.max_profit_since_buy > self.trailing_stop_profit:
            trailing_loss = self.max_profit_since_buy - profit_pct
            if trailing_loss > 0.5:
                logger.info(f"[TRAILING STOP] Max: {self.max_profit_since_buy:.2f}% -> Current: {profit_pct:.2f}%")
                return True
        
        # 시간 기반 손절
        if self.position_open_idx is not None:
            hold_minutes = current_idx - self.position_open_idx
            if hold_minutes > self.time_stop_minutes and profit_pct < -0.5:
                logger.info(f"[TIME STOP] Holding {hold_minutes}min with loss: {profit_pct:.2f}%")
                return True
        
        # 신호 기반 매도
        if signal == 0 and probs[0] > self.sell_threshold:
            logger.info(f"[SIGNAL SELL] Sell probability: {probs[0]:.3f}")
            return True
        
        # 약한 하락 신호
        if profit_pct < 0 and probs[0] > 0.25:
            logger.info(f"[WEAK SIGNAL SELL] Loss: {profit_pct:.2f}%, Sell prob: {probs[0]:.3f}")
            return True
        
        return False
    
    def execute_trade(self, idx, signal, current_price, probs):
        """거래 실행"""
        timestamp = self.df_1m.index[idx]
        
        self.price_history.append(current_price)
        
        # 매수
        if self.position == 0 and self.should_buy(signal, probs, current_price):
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
        logger.info("Starting ADA Backtest")
        logger.info("="*60)
        
        # 데이터 로드
        self.load_data()
        
        # 백테스트 진행
        total_rows = len(self.df_1m)
        last_trade_idx = None
        min_hold_intervals = 2  # 최소 2분 간격
        
        for idx in range(100, total_rows):  # 최소 100개 데이터 필요
            # 진행상황 표시
            if idx % 1000 == 0:
                progress = (idx / total_rows) * 100
                position_value = self.position * self.df_1m.iloc[idx]['close'] if self.position > 0 else 0
                total_value = self.current_capital + position_value
                total_return = ((total_value - self.initial_capital) / self.initial_capital) * 100
                
                logger.info(f"Progress: {progress:.1f}% | "
                          f"Total Return: {total_return:.2f}% | "
                          f"Trades: {len(self.trades)}")
            
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
        print("BACKTEST RESULTS")
        print("="*60)
        
        position_value = 0
        total_value = self.current_capital + position_value
        total_return = ((total_value - self.initial_capital) / self.initial_capital) * 100
        
        print(f"Initial Capital: {self.initial_capital:,.0f} KRW")
        print(f"Final Value: {total_value:,.0f} KRW")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        
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
            
            # 거래 내역 저장
            df_trades = pd.DataFrame(self.trades)
            filename = f'backtest_ada_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df_trades.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\nTrade history saved to: {filename}")

if __name__ == "__main__":
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # 백테스터 생성 및 실행
    backtester = ADABacktester(initial_capital=10_000_000)
    backtester.run_backtest()