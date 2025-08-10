"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
📈 2025CoinTrader - 실시간 AI 예측 및 매매신호 생성기 (realtime_predictor.py)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

🎯 주요 역할:
    • 실시간 수집된 데이터를 AI 모델로 분석하여 다음 분봉 예측
    • 매 분마다 매매 신호 생성 (상승/하락/횡보 예측)
    • 기존 백테스터와 동일한 로직으로 매수/매도 결정
    • 실시간 데이터 수집기와 연동하여 자동 분석

🔧 핵심 기능:
    1. AI 모델 로딩: 훈련된 Bi-LSTM 모델 로드 및 초기화
    2. 실시간 예측: 최근 데이터로 다음 분봉 예측 (상승/하락/횡보)
    3. 매매 신호 생성: 예측 결과를 매수/매도/홀드 신호로 변환
    4. 신뢰도 평가: 예측 신뢰도에 따른 거래 강도 조절

📊 입력 데이터:
    • 실시간 수집기의 1분봉, 5분봉 데이터
    • 최근 10개 분봉의 OHLCV + 기술적 지표
    • 시퀀스 길이: 30개 (모델 요구사항)

🎮 출력 신호:
    • 예측 없음: "📊 [시간] 신호 없음"
    • 매수 신호: "🟢 [시간] 매수 신호 - 상승 예측 (신뢰도: XX%)"
    • 매도 신호: "🔴 [시간] 매도 신호 - 하락 예측 (신뢰도: XX%)"
    • 홀드 신호: "🟡 [시간] 홀드 신호 - 횡보 예측 (신뢰도: XX%)"

⚠️ 주의사항:
    • AI 모델 파일이 필요: multitimeframe_bilstm_crypto_model_v04.pth
    • 충분한 과거 데이터 필요 (최소 30개 분봉)
    • 예측 결과는 참고용이며 투자 책임은 사용자에게 있음

═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 상위 디렉토리에서 모델 클래스 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MultiTimeframeBiLSTM 모델 클래스 정의 (기존과 동일)
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

class RealtimePredictor:
    """실시간 AI 예측 및 매매신호 생성 클래스"""
    
    def __init__(self):
        """초기화"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = 30  # LSTM 시퀀스 길이
        self.model = None
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
        # 매매 신호 설정 (기존 백테스터와 유사)
        self.buy_threshold = 0.65   # 매수 신뢰도 임계값
        self.sell_threshold = 0.30  # 매도 신뢰도 임계값
        self.min_confidence = 0.55  # 최소 예측 신뢰도
        
        print(f"🤖 실시간 AI 예측기 초기화 완료 (Device: {self.device})")
        
        # 모델 로드
        self.load_model()
        
    def load_model(self):
        """AI 모델 로드"""
        try:
            print("🔄 AI 모델 로딩 중...")
            
            # 모델 파일 경로
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(os.path.dirname(current_dir), 'Model_maker', 'multitimeframe_bilstm_crypto_model_v04.pth')
            
            if not os.path.exists(model_path):
                print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
                print("📝 Model_maker 폴더에서 모델을 먼저 훈련하세요.")
                return
            
            # 모델 로드
            self.model = MultiTimeframeBiLSTM().to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 모델 설정값
            self.model_threshold = checkpoint.get('probability_threshold', 0.45)
            
            print(f"✅ AI 모델 로딩 완료!")
            print(f"📊 모델 임계값: {self.model_threshold}")
            
        except Exception as e:
            print(f"❌ AI 모델 로딩 실패: {str(e)}")
            self.model = None
    
    def prepare_features_for_current_time(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame):
        """백테스터와 동일한 특징 생성 로직"""
        try:
            if len(df_1m) < 100:  # 백테스터와 동일한 최소 요구사항
                print(f"⚠️ 1분봉 데이터 부족: {len(df_1m)}개 (최소 100개 필요)")
                return None
            
            # 현재 시점 (가장 마지막 데이터)
            if isinstance(df_1m.index, pd.DatetimeIndex):
                current_time = df_1m.index[-1]
            else:
                # timestamp 컬럼에서 추출
                current_time = pd.to_datetime(df_1m['timestamp'].iloc[-1])
            
            # 1분봉 데이터 (현재 시점부터 과거 100개)
            df_1m_window = df_1m.tail(100).copy()
            
            # timestamp를 인덱스로 설정 (RangeIndex 문제 해결)
            if 'timestamp' in df_1m_window.columns:
                df_1m_window['timestamp'] = pd.to_datetime(df_1m_window['timestamp'])
                df_1m_window = df_1m_window.set_index('timestamp')
            
            # 5분봉 데이터 찾기 (현재 시점 이전의 데이터만)
            if 'timestamp' in df_5m.columns and not isinstance(df_5m.index, pd.DatetimeIndex):
                df_5m_temp = df_5m.copy()
                df_5m_temp['timestamp'] = pd.to_datetime(df_5m_temp['timestamp'])
                df_5m_temp = df_5m_temp.set_index('timestamp')
                df_5m_window = df_5m_temp[df_5m_temp.index <= current_time].tail(30).copy()
            else:
                df_5m_window = df_5m[df_5m.index <= current_time].tail(30).copy()
            
            if len(df_5m_window) < 20:
                print(f"⚠️ 5분봉 데이터 부족: {len(df_5m_window)}개 (최소 20개 필요)")
                return None
            
            # 특징 생성 (백테스터와 동일)
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
            
        except Exception as e:
            print(f"❌ 특징 데이터 준비 실패: {str(e)}")
            return None
    
    def prepare_model_input(self, features: pd.DataFrame):
        """특징 데이터를 모델 입력 형태로 변환"""
        try:
            if len(features) < self.seq_length:
                print(f"⚠️ 특징 데이터 부족: {len(features)}개 (최소 {self.seq_length}개 필요)")
                return None
            
            # 마지막 30개 시퀀스 사용
            feature_values = features.tail(self.seq_length).values
            
            # NaN 처리
            if np.isnan(feature_values).any():
                feature_values = np.nan_to_num(feature_values, nan=0.0)
            
            # 스케일링 (첫 번째 데이터로 학습)
            if not self.is_scaler_fitted:
                self.scaler.fit(feature_values)
                self.is_scaler_fitted = True
            
            data_scaled = self.scaler.transform(feature_values)
            
            # 배치 차원 추가 [1, seq_length, features]
            model_input = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)
            
            return model_input
            
        except Exception as e:
            print(f"❌ 모델 입력 데이터 준비 실패: {str(e)}")
            return None
    
    def predict_next_candle(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> dict:
        """다음 분봉 예측 (백테스터와 동일한 로직)"""
        if self.model is None:
            return {"signal": "NO_MODEL", "confidence": 0.0, "prediction": "모델 없음"}
        
        try:
            # 백테스터와 동일한 특징 생성
            features = self.prepare_features_for_current_time(df_1m, df_5m)
            if features is None:
                return {"signal": "NO_DATA", "confidence": 0.0, "prediction": "특징 생성 실패"}
            
            # 모델 입력 데이터 준비
            model_input = self.prepare_model_input(features)
            if model_input is None:
                return {"signal": "NO_DATA", "confidence": 0.0, "prediction": "모델 입력 준비 실패"}
            
            # AI 예측 수행
            with torch.no_grad():
                outputs = self.model(model_input)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                max_probability = torch.max(probabilities).item()
            
            # 예측 결과 해석
            class_names = ["매도", "홀드", "매수"]  # 백테스터와 동일한 순서
            prediction = class_names[predicted_class]
            confidence = max_probability * 100
            
            # 매매 신호 생성 (백테스터의 should_buy 로직과 동일)
            signal = self.generate_trading_signal(predicted_class, confidence, probabilities[0].cpu().numpy(), df_1m)
            
            return {
                "signal": signal,
                "confidence": confidence,
                "prediction": prediction,
                "class": predicted_class,
                "probabilities": probabilities[0].cpu().numpy(),
                "features": features  # 디버깅용
            }
            
        except Exception as e:
            print(f"❌ AI 예측 실패: {str(e)}")
            return {"signal": "ERROR", "confidence": 0.0, "prediction": f"오류: {str(e)}"}
    
    def calculate_volatility(self, df_1m: pd.DataFrame) -> float:
        """변동성 계산 (백테스터와 동일)"""
        if len(df_1m) < 25:
            return 0
        
        # 최근 25개 가격으로 변동성 계산
        recent_prices = df_1m['close'].tail(25).values
        returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] for i in range(1, len(recent_prices))]
        return np.std(returns) * 100
    
    def generate_trading_signal(self, predicted_class: int, confidence: float, probabilities: np.ndarray, df_1m: pd.DataFrame) -> str:
        """예측 결과를 매매 신호로 변환 (백테스터의 should_buy와 완전히 동일한 로직)"""
        probs = probabilities  # [매도, 홀드, 매수] 확률
        signal = predicted_class  # 0: 매도, 1: 홀드, 2: 매수
        current_price = df_1m['close'].iloc[-1]
        
        # timestamp 컬럼에서 현재 시간 추출
        if 'timestamp' in df_1m.columns:
            current_time = pd.to_datetime(df_1m['timestamp'].iloc[-1])
        else:
            # 인덱스가 DatetimeIndex인 경우
            if isinstance(df_1m.index, pd.DatetimeIndex):
                current_time = df_1m.index[-1]
            else:
                # 기본값으로 현재 시간 사용
                current_time = datetime.now()
        
        # 1. 시간대 필터 (24시간 거래 가능하므로 항상 True)
        good_trading_hours = list(range(24))
        current_hour = current_time.hour
        if current_hour not in good_trading_hours:
            return "TIME_FILTERED"
        
        # 2. 변동성 필터 (BTC는 변동성이 높을 때 더 신중하게)
        volatility = self.calculate_volatility(df_1m)
        volatility_threshold = 0.5
        if volatility > volatility_threshold:
            return "VOLATILITY_FILTERED"
        
        # 3. BTC 강화된 확률 조건 (백테스터와 완전히 동일)
        if signal == 2 and probs[2] > self.buy_threshold:  # 0.65
            # 추가 확인: HOLD 확률이 낮아야 함
            if probs[1] < 0.15:  # BTC는 더 엄격 (0.2 → 0.15)
                # 매수/매도 확률 차이 확인
                min_buy_prob_diff = 1.8
                if probs[2] / (probs[0] + 0.001) > min_buy_prob_diff:
                    return "BUY"
        
        # 4. 보조 조건 강화
        if signal == 1 and probs[2] > probs[0] * 2.2 and probs[2] > 0.55:  # BTC 더 엄격
            if probs[1] < 0.20:
                return "WEAK_BUY"
        
        # 5. BTC 가격 조건 (더 보수적) - 25개 이상의 데이터가 있을 때만
        if len(df_1m) >= 25:
            avg_price = np.mean(df_1m['close'].tail(15).values)  # 더 긴 기간 평균
            if current_price < avg_price * 0.995 and probs[2] > 0.60:  # 더 엄격한 조건
                return "PRICE_BUY"
        
        # 6. 매도 신호 확인
        if signal == 0 and probs[0] > self.sell_threshold:  # 0.30
            return "SELL"
        elif signal == 0:
            return "WEAK_SELL"
        
        # 7. 홀드 신호
        if signal == 1:
            return "HOLD"
            
        return "NO_SIGNAL"
    
    def format_prediction_output(self, result: dict, current_time: datetime) -> str:
        """예측 결과를 출력 형식으로 변환"""
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        if result["signal"] == "NO_MODEL":
            return f"📊 [{time_str}] 신호 없음 - AI 모델 로딩 실패"
        elif result["signal"] == "NO_DATA":
            return f"📊 [{time_str}] 신호 없음 - {result['prediction']}"
        elif result["signal"] == "ERROR":
            return f"📊 [{time_str}] 신호 없음 - {result['prediction']}"
        elif result["signal"] == "TIME_FILTERED":
            return f"📊 [{time_str}] 신호 없음 - 거래 시간 제한 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "VOLATILITY_FILTERED":
            return f"📊 [{time_str}] 신호 없음 - 높은 변동성 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "BUY":
            return f"🟢 [{time_str}] 강력한 매수 신호 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "WEAK_BUY":
            return f"🟡 [{time_str}] 약한 매수 신호 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "PRICE_BUY":
            return f"🟢 [{time_str}] 가격 기반 매수 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "SELL":
            return f"🔴 [{time_str}] 강력한 매도 신호 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "WEAK_SELL":
            return f"🟠 [{time_str}] 약한 매도 신호 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "HOLD":
            return f"🟡 [{time_str}] 홀드 신호 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "NO_SIGNAL":
            return f"📊 [{time_str}] 신호 없음 - 매매 조건 미충족 (신뢰도: {result['confidence']:.1f}%)"
        else:
            return f"📊 [{time_str}] 알 수 없는 신호: {result['signal']}"
    
    def analyze_realtime_data(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> str:
        """실시간 데이터 분석 및 매매신호 생성 (메인 함수)"""
        current_time = datetime.now()
        
        # AI 예측 수행
        prediction_result = self.predict_next_candle(df_1m, df_5m)
        
        # 결과 출력 형식으로 변환
        output = self.format_prediction_output(prediction_result, current_time)
        
        return output

def load_real_data():
    """실제 CSV 파일에서 데이터 로드"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'REALTIME_DATA')
        
        # 1분봉 데이터 로드
        btc_1m_path = os.path.join(data_dir, 'BTC_1m_realtime.csv')
        if not os.path.exists(btc_1m_path):
            print(f"❌ 1분봉 데이터 파일이 없습니다: {btc_1m_path}")
            return None, None
        
        df_1m = pd.read_csv(btc_1m_path)
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
        df_1m = df_1m.sort_values('timestamp')
        df_1m.set_index('timestamp', inplace=True)
        
        # 5분봉 데이터 로드
        btc_5m_path = os.path.join(data_dir, 'BTC_5m_realtime.csv')
        if not os.path.exists(btc_5m_path):
            print(f"❌ 5분봉 데이터 파일이 없습니다: {btc_5m_path}")
            return None, None
        
        df_5m = pd.read_csv(btc_5m_path)
        df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
        df_5m = df_5m.sort_values('timestamp')
        df_5m.set_index('timestamp', inplace=True)
        
        print(f"📊 실제 데이터 로드 완료!")
        print(f"1분봉: {len(df_1m)}개 ({df_1m.index[0]} ~ {df_1m.index[-1]})")
        print(f"5분봉: {len(df_5m)}개 ({df_5m.index[0]} ~ {df_5m.index[-1]})")
        
        return df_1m, df_5m
        
    except Exception as e:
        print(f"❌ 실제 데이터 로드 실패: {str(e)}")
        return None, None

def main():
    """테스트용 메인 함수"""
    predictor = RealtimePredictor()
    
    # 실제 데이터로 테스트
    print("\n🧪 실제 데이터로 테스트 중...")
    
    df_1m, df_5m = load_real_data()
    
    if df_1m is None or df_5m is None:
        print("\n⚠️ 실제 데이터를 로드할 수 없어 샘플 데이터로 테스트합니다.")
        
        # 백테스터와 동일한 형식의 더미 데이터 생성
        sample_data_1m = {
            'close': np.random.random(100) * 1000 + 159000,
            'volume': np.random.random(100) * 1000,
            'macd': np.random.random(100) * 100 - 50,
            'macd_signal': np.random.random(100) * 100 - 50,
            'macd_histogram': np.random.random(100) * 50 - 25,
            'rsi': np.random.random(100) * 100,
            'bb_upper': np.random.random(100) * 1000 + 159500,
            'bb_lower': np.random.random(100) * 1000 + 158500,
            'ma20': np.random.random(100) * 1000 + 159000,
            'atr': np.random.random(100) * 1000,
            'price_change': np.random.random(100) * 2 - 1
        }
        
        time_index_1m = pd.date_range(start='2025-01-01 10:00:00', periods=100, freq='1min')
        df_1m = pd.DataFrame(sample_data_1m, index=time_index_1m)
        
        sample_data_5m = {
            'close': np.random.random(30) * 1000 + 159000,
            'volume': np.random.random(30) * 1000,
            'macd': np.random.random(30) * 100 - 50,
            'macd_signal': np.random.random(30) * 100 - 50,
            'macd_histogram': np.random.random(30) * 50 - 25,
        }
        
        time_index_5m = pd.date_range(start='2025-01-01 09:00:00', periods=30, freq='5min')
        df_5m = pd.DataFrame(sample_data_5m, index=time_index_5m)
    
    # 마지막 데이터 시점 확인
    last_1m_time = df_1m.index[-1]
    last_5m_time = df_5m.index[-1]
    
    print(f"\n📅 데이터 현황:")
    print(f"마지막 1분봉: {last_1m_time}")
    print(f"마지막 5분봉: {last_5m_time}")
    print(f"다음 예측 대상: {last_1m_time + pd.Timedelta(minutes=1)}")
    
    # 예측 테스트
    result = predictor.analyze_realtime_data(df_1m, df_5m)
    print(f"\n{result}")

if __name__ == "__main__":
    main()