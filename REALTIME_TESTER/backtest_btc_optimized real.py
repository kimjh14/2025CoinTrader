"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📈 2025CoinTrader - BTC 최적화 백테스팅 시스템 (backtest_btc_optimized.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • 훈련된 Bi-LSTM 모델을 사용한 BTC 코인 백테스팅
    • 과거 데이터를 이용한 매매 전략 성능 검증
    • 최적화된 리스크 관리 및 포지션 사이징
    • 상세한 거래 로그 및 성과 분석
    • 실전 거래 전 전략 검증 도구

🔧 핵심 기능:
    1. 모델 기반 예측: 다중 타임프레임 Bi-LSTM 모델로 BTC 가격 예측
    2. 거래 전략: 예측 신호 기반 자동 매수/매도 결정
    3. 리스크 관리: 손절선, 익절선, 포지션 사이징 최적화
    4. 성과 분석: 수익률, 샤프 비율, 최대 낙폭, 승률 계산
    5. 상세 로깅: 모든 거래 과정 및 결과 상세 기록

💰 거래 전략:
    • 예측 신뢰도 > 65% 시 거래 신호 생성 (BTC는 더 보수적 접근)
    • 상승 예측 시 매수, 하락 예측 시 매도/공매도
    • 동적 손절선: ATR 기반 변동성 고려
    • 트레일링 스톱: 수익 보호를 위한 추적 손절
    • 포지션 크기: 변동성과 자산 규모 고려한 켈리 공식 적용

📊 성과 지표:
    • 총 수익률 및 연환산 수익률
    • 샤프 비율 (위험 대비 수익)
    • 최대 낙폭 (Maximum Drawdown)
    • 승률 및 평균 수익/손실 비율
    • 거래 빈도 및 보유 기간 분석

📋 사용 방법:
    python backtest_btc_optimized.py
    
    생성 파일:
    - backtest_btc_optimized_YYYYMMDD_HHMMSS.log: 상세 거래 로그
    - btc_backtest_results.png: 백테스팅 결과 차트
    - btc_performance_report.json: 성과 지표 리포트

⚠️ 주의사항:
    • multitimeframe_bilstm_crypto_model_v04.pth 파일이 필요
    • 과거 성과가 미래 수익을 보장하지 않음
    • 실제 거래 시 슬리피지 및 수수료 고려 필요
    • 충분한 백테스팅 기간으로 전략 검증 권장

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

# 로깅 설정 (REALTIME_DATA 폴더에 저장)
current_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(current_dir, 'REALTIME_DATA')

# 디렉토리가 없으면 생성
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

result_timestamp = os.environ.get('RESULT_TIMESTAMP', datetime.now().strftime("%Y%m%d_%H%M%S"))
log_file_path = os.path.join(log_dir, f'backtest_btc_optimized_{result_timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True  # 기존 로깅 설정 덮어쓰기
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

class BTCBacktesterOptimized:
    """최적화된 BTC 백테스팅 시스템"""
    
    def __init__(self, initial_capital=10_000_000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position = 0
        self.avg_price = 0
        self.trades = []
        self.commission_rate = 0.0005  # 0.05%
        self.slippage_rate = 0.00025  # 0.025% 슬리피지 (매수/매도 시 모두 불리하게 체결) 
        # (틱당 1000원 일때 0.025% 는 BTC 가격이 400만원일때 기준임, 현재가는 1억5천만원으로 0.025%는 37500원 --> 틱당 1000원 이므로 37.5틱 수준임)
        
        # 모델 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = 30
        
        # === BTC 최적화된 거래 설정 ===
        self.buy_threshold = 0.65  # BTC는 더 보수적 (0.50 → 0.65)
        self.sell_threshold = 0.30  # 매도도 더 보수적 (0.25 → 0.30)
        self.force_sell_loss = -0.8  # 손절 완화 (-0.7 → -0.8)
        self.force_sell_profit = 2.0  # 익절 상향 (1.5 → 2.0)
        self.trailing_stop_profit = 0.8  # 트레일링 완화 (0.6 → 0.8)
        self.time_stop_minutes = 20  # 시간 손절 완화 (15 → 20)
        
        # BTC 전용 추가 최적화 설정
        self.min_hold_minutes = 1  # 최소 거래 간격 3분
        self.good_trading_hours = list(range(24))  # 24시간 거래 가능 (암호화폐 특성 반영)
        self.min_buy_prob_diff = 1.8  # 확률 차이 완화 (2.0 → 1.8)
        self.volatility_threshold = 0.5  # BTC 변동성 임계값
        
        logger.info(f"Initializing Optimized BTC Backtester")
        logger.info(f"Initial Capital: {self.initial_capital:,} KRW")
        logger.info(f"Device: {self.device}")
        logger.info(f"=== BTC Optimized Settings ===")
        logger.info(f"Buy Threshold: {self.buy_threshold}")
        logger.info(f"Sell Threshold: {self.sell_threshold}")
        logger.info(f"Stop Loss: {self.force_sell_loss}%")
        logger.info(f"Slippage Rate: ±{self.slippage_rate*100:.2f}% (매수/매도 시 모두 불리한 체결 반영)")
        logger.info(f"Take Profit: {self.force_sell_profit}%")
        logger.info(f"Trailing Stop: {self.trailing_stop_profit}%")
        logger.info(f"Min Hold Time: {self.min_hold_minutes} minutes")
        
        # 모델 로드
        self.load_model()
        
        # Scaler 초기화
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
        # 가격 히스토리
        self.price_history = deque(maxlen=25)  # BTC는 더 긴 히스토리
        
        # 포지션 추적
        self.position_open_idx = None
        self.max_profit_since_buy = 0
        
        # 추가 통계
        self.filtered_signals = 0
        self.time_filtered = 0
        self.volatility_filtered = 0
        
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
        """과거 데이터 로드"""
        logger.info("Loading historical BTC data...")
        
        # REALTIME_DATA 폴더 경로 설정
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'REALTIME_DATA')
        logger.info(f"REALTIME_DATA 폴더에서 데이터 로드: {data_dir}")
        
        # 데이터 디렉토리 존재 확인
        if not os.path.exists(data_dir):
            logger.error(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        
        # 1분봉 데이터
        btc_1m_path = os.path.join(data_dir, 'BTC_1m_realtime.csv')
        if not os.path.exists(btc_1m_path):
            logger.error(f"1분봉 데이터 파일이 존재하지 않습니다: {btc_1m_path}")
            raise FileNotFoundError(f"1분봉 데이터 파일을 찾을 수 없습니다: {btc_1m_path}")
        
        self.df_1m = pd.read_csv(btc_1m_path)
        self.df_1m['timestamp'] = pd.to_datetime(self.df_1m['timestamp'])
        self.df_1m = self.df_1m.sort_values('timestamp')
        self.df_1m.set_index('timestamp', inplace=True)
        
        # 5분봉 데이터
        btc_5m_path = os.path.join(data_dir, 'BTC_5m_realtime.csv')
        if not os.path.exists(btc_5m_path):
            logger.error(f"5분봉 데이터 파일이 존재하지 않습니다: {btc_5m_path}")
            raise FileNotFoundError(f"5분봉 데이터 파일을 찾을 수 없습니다: {btc_5m_path}")
        
        self.df_5m = pd.read_csv(btc_5m_path)
        self.df_5m['timestamp'] = pd.to_datetime(self.df_5m['timestamp'])
        self.df_5m = self.df_5m.sort_values('timestamp')
        self.df_5m.set_index('timestamp', inplace=True)
        
        logger.info(f"Loaded BTC 1m data: {len(self.df_1m)} rows")
        logger.info(f"Loaded BTC 5m data: {len(self.df_5m)} rows")
        logger.info(f"Date range: {self.df_1m.index[0]} to {self.df_1m.index[-1]}")
        print()
        
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
    
    def calculate_volatility(self, current_price):
        """변동성 계산"""
        if len(self.price_history) < 10:
            return 0
        
        prices = list(self.price_history)
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        return np.std(returns) * 100
    
    def should_buy(self, signal, probs, current_price, current_time):
        """BTC 최적화된 매수 결정"""
        # 시간대 필터
        current_hour = current_time.hour
        if current_hour not in self.good_trading_hours:
            self.time_filtered += 1
            return False
        
        # 변동성 필터 (BTC는 변동성이 높을 때 더 신중하게)
        volatility = self.calculate_volatility(current_price)
        if volatility > self.volatility_threshold:
            self.volatility_filtered += 1
            return False
        
        # BTC 강화된 확률 조건
        if signal == 2 and probs[2] > self.buy_threshold:
            # 추가 확인: HOLD 확률이 낮아야 함
            if probs[1] < 0.15:  # BTC는 더 엄격 (0.2 → 0.15)
                # 매수/매도 확률 차이 확인
                if probs[2] / (probs[0] + 0.001) > self.min_buy_prob_diff:
                    return True
        
        # 보조 조건 강화
        if signal == 1 and probs[2] > probs[0] * 2.2 and probs[2] > 0.55:  # BTC 더 엄격
            if probs[1] < 0.20:
                return True
        
        # BTC 가격 조건 (더 보수적)
        if len(self.price_history) >= 25:
            avg_price = np.mean(list(self.price_history)[-15:])  # 더 긴 기간 평균
            if current_price < avg_price * 0.995 and probs[2] > 0.60:  # 더 엄격한 조건
                return True
        
        self.filtered_signals += 1
        return False
    
    def should_sell(self, signal, probs, current_data, current_idx):
        """BTC 최적화된 매도 결정 (고가/저가 반영)"""
        if self.position == 0:
            return False, None
        
        # 현재 분봉의 OHLC 데이터
        current_price = current_data['close']
        high_price = current_data['high']
        low_price = current_data['low']
        
        # 현재 종가 기준 수익률
        profit_pct = ((current_price - self.avg_price) / self.avg_price) * 100
        
        # 고가/저가 기준 수익률
        high_profit_pct = ((high_price - self.avg_price) / self.avg_price) * 100
        low_profit_pct = ((low_price - self.avg_price) / self.avg_price) * 100
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # 트레일링 스톱을 위한 최고점 업데이트 (고가 기준)
        # ═══════════════════════════════════════════════════════════════════════════════
        # 1분봉에서는 시간 순서를 알 수 없으므로 고가로 최고점을 갱신한다.
        # 예시: 분봉에서 고가가 +1.9%였다면, 실제로 어느 시점에서든 1.9%까지 올랐던 것이므로
        # 트레일링 스톱의 기준점을 1.9%로 설정하는 것이 맞다.
        if high_profit_pct > self.max_profit_since_buy:
            self.max_profit_since_buy = high_profit_pct
        
        # === 1. BTC 손절매 (저가 기준) ===
        if low_profit_pct <= self.force_sell_loss:
            # 손절가 계산
            stop_loss_price = self.avg_price * (1 + self.force_sell_loss / 100)
            logger.info(f"[BTC STOP LOSS] 저가 {low_price:,.0f}원에서 손절선 {stop_loss_price:,.0f}원 터치")
            return True, stop_loss_price
        
        # === 2. BTC 익절매 (고가 기준) ===
        if high_profit_pct >= self.force_sell_profit:
            # 익절가 계산
            take_profit_price = self.avg_price * (1 + self.force_sell_profit / 100)
            logger.info(f"[BTC TAKE PROFIT] 고가 {high_price:,.0f}원에서 익절선 {take_profit_price:,.0f}원 터치")
            return True, take_profit_price
        
        # ═══════════════════════════════════════════════════════════════════════════════
        # === 3. BTC 트레일링 스톱 (비활성화) ===
        # ═══════════════════════════════════════════════════════════════════════════════
        # 트레일링 스톱을 1분봉 데이터로 정확히 구현하는 것은 불가능한 이유:
        # 
        # 1. 시간 순서 불명: 고가와 저가가 언제 발생했는지 알 수 없음
        #    예시) 고가 101,500원, 저가 100,500원이 기록된 분봉에서
        #    - 고가 → 저가 순서인지, 저가 → 고가 순서인지 모름
        #    - 트레일링 실행 시점이 완전히 달라짐
        # 
        # 2. 매수 분봉에서의 즉시 매도 가능성:
        #    예시) 매수가 100,000원인 분봉에서
        #    - 고가 101,500원 (+1.5%) → 트레일링 활성화
        #    - 저가 100,500원 (+0.5%) → 1.0%p 하락으로 트레일링 실행
        #    - 결과: 매수와 동시에 매도 (현실적으로 불가능)
        # 
        # 3. 부정확한 가정의 누적:
        #    - 어떤 가정을 해도 실제 상황과 차이 발생
        #    - 보수적 가정 → 과도한 트레일링 실행
        #    - 낙관적 가정 → 비현실적 결과
        # 
        # 결론: 트레일링 스톱은 초단위 또는 틱 데이터가 있어야만 정확히 구현 가능
        # 1분봉 백테스팅에서는 다른 매도 조건들(손절/익절/AI신호)로 대체
        
        # if self.max_profit_since_buy > self.trailing_stop_profit:
        #     trailing_loss = self.max_profit_since_buy - low_profit_pct
        #     if trailing_loss > 0.4:
        #         trailing_stop_price = self.avg_price * (1 + (self.max_profit_since_buy - 0.4) / 100)
        #         logger.info(f"[BTC TRAILING STOP] Max: {self.max_profit_since_buy:.2f}% -> Low: {low_profit_pct:.2f}%")
        #         return True, trailing_stop_price
        
        # === 4. BTC 시간 기반 손절 ===
        if self.position_open_idx is not None:
            hold_minutes = current_idx - self.position_open_idx
            if hold_minutes > self.time_stop_minutes and profit_pct < -0.2:
                logger.info(f"[BTC TIME STOP] Holding {hold_minutes}min with loss: {profit_pct:.2f}%")
                return True, current_price
        
        # === 5. BTC 기술적 신호 기반 매도 ===
        if signal == 0 and probs[0] > self.sell_threshold:
            logger.info(f"[BTC SIGNAL SELL] Sell probability: {probs[0]:.3f}")
            return True, current_price
        
        # === 6. BTC 약한 하락 신호 ===
        if profit_pct < 0 and probs[0] > 0.25:
            logger.info(f"[BTC WEAK SIGNAL SELL] Loss: {profit_pct:.2f}%, Sell prob: {probs[0]:.3f}")
            return True, current_price
        
        # === 7. BTC 수익 중 매도 신호 ===
        if profit_pct > 0.5 and probs[0] > 0.20:
            logger.info(f"[BTC PROFIT PROTECTION] Profit: {profit_pct:.2f}%, Sell prob: {probs[0]:.3f}")
            return True, current_price
        
        return False, None
    
    def execute_trade(self, idx, signal, current_data, probs):
        """
        거래 실행 (고가/저가 반영)
        
        ═══════════════════════════════════════════════════════════════════════════════
        실시간 자동매매 현실성 반영을 위한 추가 고려사항들 (현재 미반영)
        ═══════════════════════════════════════════════════════════════════════════════
        
        현재 백테스트에 반영되지 않은 현실적 요소들:
        
        1. 처리 지연 (Processing Delay):
           - 분봉 완성 → 데이터 수신 → AI 예측 → 매수 결정: 2-5초 소요
           - 실제로는 10:00:00이 아닌 10:00:03~05에 매수 주문
           
        2. 시가-체결가 갭 (Momentum Slippage):
           - 09:58~59초 급등/급락 시 → 10:00:xx초 체결가는 시가와 차이
           - 상승 모멘텀 중 매수: 체결가 > 시가 (불리한 체결)
           - 예시) 시가 100만원 → 실제 체결 100.5만원 (0.5% 슬리피지)
           
        3. 거래량 제약 (Liquidity Constraints):
           - 큰 주문 시 여러 호가에 분할 체결 → 평균 체결가 상승
           - 변동성 큰 시점에 호가 스프레드 확대 (매도1호가 - 매수1호가)
           
        4. 네트워크 지연 및 거래소 처리 시간:
           - API 호출 → 주문 접수 → 매칭 엔진 → 체결 확인: 추가 1-2초
           - 급변동 시 거래소 과부하로 처리 시간 증가
           
        5. 주문 실패 가능성:
           - 급변동 시 주문 거부 (거래소 자체 보호)
           - 호가 급변으로 체결 실패 → 재주문 필요
           - 기술적 문제 (API 오류, 네트워크 단절, 서버 다운 등)
           
        6. 수수료 및 세금:
           - 현재는 0.05% 고정이지만 실제로는 VIP 등급에 따라 차등
           - 대량 거래 시 추가 수수료, 세금 고려 필요
           
        ⚠️ 결론: 현재 백테스트는 "완벽한 조건"을 가정
        실제 자동매매 시 수익률이 백테스트 대비 10-30% 낮아질 가능성
        특히 단타 전략일수록 이런 마이크로 비용들의 누적 효과가 크다.
        """
        timestamp = self.df_1m.index[idx]
        current_price = current_data['close']
        
        self.price_history.append(current_price)
        
        # 매수 (슬리피지 반영)
        if self.position == 0 and self.should_buy(signal, probs, current_price, timestamp):
            # ═══════════════════════════════════════════════════════════════════════════════
            # 슬리피지 반영: 실제 체결가는 시가보다 0.5% 높게 설정
            # ═══════════════════════════════════════════════════════════════════════════════
            # 이유: 실시간 자동매매에서 다음과 같은 상황들로 인해 불리하게 체결됨
            # 1. 처리 지연: 분봉 완성 → AI 예측 → 주문 실행까지 5-10초 소요
            # 2. 모멘텀 슬리피지: 상승 예측 → 매수 주문 시점에 이미 가격 상승
            # 3. 호가 스프레드: 매수 주문은 매도1호가에 체결 (시가보다 높음)
            # 4. 거래량 충격: 큰 주문 시 여러 호가에 걸쳐 체결
            
            slippage_adjusted_price = current_price * (1 + self.slippage_rate)  # 슬리피지만큼 높게 매수
            
            use_capital = self.current_capital * 0.98  # 98% 자본 활용
            quantity = use_capital / slippage_adjusted_price  # 슬리피지 적용된 가격으로 수량 계산
            cost = quantity * slippage_adjusted_price * (1 + self.commission_rate)  # 실제 비용
            
            if cost <= self.current_capital:
                self.position = quantity
                self.avg_price = slippage_adjusted_price  # 손절/익절 기준가는 실제 체결가
                self.current_capital -= cost
                self.position_open_idx = idx
                self.max_profit_since_buy = 0
                
                trade = {
                    'timestamp': timestamp,
                    'idx': idx,
                    'action': 'BUY',
                    'price': slippage_adjusted_price,  # 실제 체결가 기록
                    'original_price': current_price,   # 시가 참고용
                    'slippage': self.slippage_rate * 100,  # 슬리피지 0.5% 기록
                    'quantity': quantity,
                    'cost': cost,
                    'balance': self.current_capital,
                    'signal': signal,
                    'prob_sell': probs[0],
                    'prob_hold': probs[1],
                    'prob_buy': probs[2]
                }
                self.trades.append(trade)
                
                logger.info(f"[BTC BUY] {timestamp} | {quantity:.6f} @ {slippage_adjusted_price:,.0f} "
                          f"(시가: {current_price:,.0f}, 슬리피지: +{self.slippage_rate*100:.2f}%) | "
                          f"Probs: [S:{probs[0]:.3f}, H:{probs[1]:.3f}, B:{probs[2]:.3f}]")
                return True
        
        # 매도 (고가/저가 반영)
        elif self.position > 0:
            should_sell_result, sell_price = self.should_sell(signal, probs, current_data, idx)
            
            if should_sell_result:
                # 실제 매도가 결정 (손절/익절의 경우 해당 가격, 그 외는 종가)
                base_sell_price = sell_price if sell_price else current_price
                # 매도 시 슬리피지 적용 (불리하게 체결)
                actual_sell_price = base_sell_price * (1 - self.slippage_rate)
                
                revenue = self.position * actual_sell_price * (1 - self.commission_rate)
                profit = revenue - (self.position * self.avg_price)
                profit_pct = (profit / (self.position * self.avg_price)) * 100
                
                self.current_capital += revenue
                
                hold_minutes = idx - self.position_open_idx if self.position_open_idx else 0
                
                trade = {
                    'timestamp': timestamp,
                    'idx': idx,
                    'action': 'SELL',
                    'price': actual_sell_price,
                    'close_price': current_price,  # 참고용 종가
                    'high_price': current_data['high'],  # 참고용 고가
                    'low_price': current_data['low'],    # 참고용 저가
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
                
                logger.info(f"[BTC SELL] {timestamp} | {self.position:.6f} @ {actual_sell_price:,.0f} "
                          f"(기준가: {base_sell_price:,.0f}, 슬리피지: -{self.slippage_rate*100:.2f}%) | "
                          f"Profit: {profit_pct:.2f}% | Hold: {hold_minutes}min")
                
                # 자본 현황 표시 (올바른 계산)
                # 매수 시점의 잔액 = 매도후 잔액 - 매도 수익 + 매수 비용
                buy_cost = self.position * self.avg_price * (1 + self.commission_rate)  # 실제 매수 비용
                buy_balance = self.current_capital - revenue + buy_cost  # 매수 직전 잔액
                buy_ratio = (buy_balance / self.initial_capital) * 100
                sell_ratio = (self.current_capital / self.initial_capital) * 100
                
                logger.info(f"💰 자본현황 | 초기자본: {self.initial_capital:,.0f}원 | "
                          f"매수전: {buy_balance:,.0f}원 ({buy_ratio:.2f}%) → "
                          f"매도후: {self.current_capital:,.0f}원 ({sell_ratio:.2f}%)")
                logger.info("")  # 거래 사이클 완료 후 로그 파일에 공백줄 추가
                print()  # 콘솔 출력용 공백줄
                
                self.position = 0
                self.avg_price = 0
                self.position_open_idx = None
                self.max_profit_since_buy = 0
                return True
            
        return False
    
    def run_backtest(self):
        """백테스트 실행"""
        logger.info("="*60)
        logger.info("Starting Optimized BTC Backtest")
        logger.info("="*60)
        
        # 데이터 로드
        self.load_data()
        
        # 백테스트 진행
        total_rows = len(self.df_1m)
        last_trade_idx = None
        min_hold_intervals = self.min_hold_minutes
        
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
                          f"Time Filtered: {self.time_filtered} | "
                          f"Volatility Filtered: {self.volatility_filtered}")
                print()  # 진행상황 출력 후 완전한 공백줄 추가
            
            # 특징 준비
            features = self.prepare_features_at_index(idx)
            if features is None:
                continue
            
            # 예측
            signal, probs = self.make_prediction(features)
            if signal is None:
                continue
            
            # 현재 분봉 OHLC 데이터
            current_data = {
                'open': self.df_1m.iloc[idx]['open'],
                'high': self.df_1m.iloc[idx]['high'],
                'low': self.df_1m.iloc[idx]['low'],
                'close': self.df_1m.iloc[idx]['close']
            }
            
            # 거래 시간 제한 확인 (3분 간격 + 같은 분봉 제한)
            can_buy = True
            if last_trade_idx and self.position == 0:
                # 1. 같은 분봉에서 매도 후 다시 매수하는 것은 불가능 (현실적 제약)
                # 2. 추가로 3분 간격 제한 적용
                if idx <= last_trade_idx:  # 같은 분봉이면 매수 불가
                    can_buy = False
                elif idx - last_trade_idx < self.min_hold_minutes:  # 3분 간격 제한
                    can_buy = False
            
            # 거래 실행
            if self.position > 0:  # 포지션 있으면 매도 체크
                if self.execute_trade(idx, signal, current_data, probs):
                    last_trade_idx = idx
            elif can_buy:  # 포지션 없고 매수 가능하면
                if self.execute_trade(idx, signal, current_data, probs):
                    last_trade_idx = idx
                    # 매수 직후 같은 분봉에서 손절/익절 체크 (현실 반영)
                    if self.position > 0:  # 매수가 성공했으면
                        sell_result, sell_price = self.should_sell(signal, probs, current_data, idx)
                        if sell_result:
                            # 같은 분봉에서 매도 실행
                            base_sell_price = sell_price if sell_price else current_data['close']
                            # 매도 시 슬리피지 적용 (불리하게 체결)
                            actual_sell_price = base_sell_price * (1 - self.slippage_rate)
                            revenue = self.position * actual_sell_price * (1 - self.commission_rate)
                            profit = revenue - (self.position * self.avg_price)
                            profit_pct = (profit / (self.position * self.avg_price)) * 100
                            
                            self.current_capital += revenue
                            
                            # 거래 기록
                            timestamp = self.df_1m.index[idx]
                            trade = {
                                'timestamp': timestamp,
                                'idx': idx,
                                'action': 'SELL',
                                'price': actual_sell_price,
                                'close_price': current_data['close'],
                                'high_price': current_data['high'],
                                'low_price': current_data['low'],
                                'quantity': self.position,
                                'revenue': revenue,
                                'profit': profit,
                                'profit_pct': profit_pct,
                                'balance': self.current_capital,
                                'signal': signal,
                                'prob_sell': probs[0],
                                'prob_hold': probs[1],
                                'prob_buy': probs[2],
                                'hold_minutes': 0,  # 같은 분봉에서 매수+매도
                                'same_candle_trade': True  # 같은 분봉 거래 표시
                            }
                            self.trades.append(trade)
                            
                            logger.info(f"[BTC SAME CANDLE SELL] {timestamp} | {self.position:.6f} @ {actual_sell_price:,.0f} "
                                      f"(기준가: {base_sell_price:,.0f}, 슬리피지: -{self.slippage_rate*100:.2f}%) | "
                                      f"Profit: {profit_pct:.2f}% | Hold: 0min (같은 분봉 매매)")
                            
                            # 자본 현황 표시 (같은 분봉 매매 - 올바른 계산)
                            # 매수 시점의 잔액 = 매도후 잔액 - 매도 수익 + 매수 비용
                            buy_cost = self.position * self.avg_price * (1 + self.commission_rate)  # 실제 매수 비용
                            buy_balance = self.current_capital - revenue + buy_cost  # 매수 직전 잔액
                            buy_ratio = (buy_balance / self.initial_capital) * 100
                            sell_ratio = (self.current_capital / self.initial_capital) * 100
                            
                            logger.info(f"💰 자본현황 | 초기자본: {self.initial_capital:,.0f}원 | "
                                      f"매수전: {buy_balance:,.0f}원 ({buy_ratio:.2f}%) → "
                                      f"매도후: {self.current_capital:,.0f}원 ({sell_ratio:.2f}%)")
                            logger.info("")  # 같은 분봉 매매 완료 후 로그 파일에 공백줄 추가
                            print()  # 콘솔 출력용 공백줄
                            
                            # 포지션 정리
                            self.position = 0
                            self.avg_price = 0
                            self.position_open_idx = None
                            self.max_profit_since_buy = 0
        
        # 마지막 포지션 정리
        if self.position > 0:
            base_final_price = self.df_1m.iloc[-1]['close']
            # 매도 시 슬리피지 적용 (불리하게 체결)
            final_price = base_final_price * (1 - self.slippage_rate)
            revenue = self.position * final_price * (1 - self.commission_rate)
            self.current_capital += revenue
            logger.info(f"[BTC FINAL SELL] Closing position at {final_price:,.0f} "
                      f"(기준가: {base_final_price:,.0f}, 슬리피지: -{self.slippage_rate*100:.2f}%)")
            logger.info("")  # 마지막 거래 후 로그 파일에 공백줄 추가
            print()  # 콘솔 출력용 공백줄
        
        # 결과 출력
        self.print_results()
    
    def print_results(self):
        """백테스트 결과 출력"""
        print("\n" + "="*60)
        print("BTC OPTIMIZED BACKTEST RESULTS")
        print("="*60)
        
        position_value = 0
        total_value = self.current_capital + position_value
        total_return = ((total_value - self.initial_capital) / self.initial_capital) * 100
        
        # 결과를 콘솔과 로그에 모두 출력
        result_lines = [
            "="*60,
            "BTC OPTIMIZED BACKTEST RESULTS",
            "="*60,
            f"Initial Capital: {self.initial_capital:,.0f} KRW",
            f"Final Value: {total_value:,.0f} KRW",
            f"Total Return: {total_return:.2f}%",
            f"Total Trades: {len(self.trades)}",
            f"Filtered Signals: {self.filtered_signals}",
            f"Time Filtered: {self.time_filtered}",
            f"Volatility Filtered: {self.volatility_filtered}"
        ]
        
        # 콘솔 출력
        print(f"Initial Capital: {self.initial_capital:,.0f} KRW")
        print(f"Final Value: {total_value:,.0f} KRW")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Filtered Signals: {self.filtered_signals}")
        print(f"Time Filtered: {self.time_filtered}")
        print(f"Volatility Filtered: {self.volatility_filtered}")
        
        # 로그 파일에 기록
        for line in result_lines:
            logger.info(line)
        
        # 결과를 반환 (메인 로그에서 사용하기 위해)
        self.final_results = {
            'initial_capital': self.initial_capital,
            'final_value': total_value,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'filtered_signals': self.filtered_signals,
            'time_filtered': self.time_filtered,
            'volatility_filtered': self.volatility_filtered
        }
        
        if self.trades:
            # 거래 통계
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] == 'SELL']
            
            if sell_trades:
                profits = [t['profit_pct'] for t in sell_trades]
                winning_trades = sum(1 for p in profits if p > 0)
                
                print(f"\nBTC Trade Statistics:")
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
            
            # 거래 내역 저장 (REALTIME_DATA 폴더에 저장)
            df_trades = pd.DataFrame(self.trades)
            
            # 로그와 동일한 폴더 사용
            current_dir = os.path.dirname(os.path.abspath(__file__))
            result_dir = os.path.join(current_dir, 'REALTIME_DATA')
            
            # 디렉토리가 없으면 생성
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            
            result_timestamp = os.environ.get('RESULT_TIMESTAMP', datetime.now().strftime("%Y%m%d_%H%M%S"))
            filename = os.path.join(result_dir, f'backtest_btc_optimized_results_{result_timestamp}.csv')
            
            df_trades.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\nBTC Trade history saved to: {filename}")

if __name__ == "__main__":
    # GPU 정보 출력
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # BTC 백테스터 생성 및 실행
    backtester = BTCBacktesterOptimized(initial_capital=10_000_000)
    backtester.run_backtest()