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
            if not df_5m.empty:
                if 'timestamp' in df_5m.columns and not isinstance(df_5m.index, pd.DatetimeIndex):
                    df_5m_temp = df_5m.copy()
                    df_5m_temp['timestamp'] = pd.to_datetime(df_5m_temp['timestamp'])
                    df_5m_temp = df_5m_temp.set_index('timestamp')
                    df_5m_window = df_5m_temp[df_5m_temp.index <= current_time].tail(30).copy()
                else:
                    df_5m_window = df_5m[df_5m.index <= current_time].tail(30).copy()
                
                if len(df_5m_window) < 20:
                    print(f"⚠️ 5분봉 데이터 부족: {len(df_5m_window)}개 (최소 20개 필요)")
                    # 5분봉이 부족해도 더미 데이터로 처리
                    dummy_data = {
                        'macd': [0] * 30,
                        'macd_signal': [0] * 30,
                        'macd_histogram': [0] * 30,
                    }
                    time_index = pd.date_range(end=current_time, periods=30, freq='5min')
                    df_5m_window = pd.DataFrame(dummy_data, index=time_index)
            else:
                # 5분봉이 없으면 더미 데이터 생성
                dummy_data = {
                    'macd': [0] * 30,
                    'macd_signal': [0] * 30,
                    'macd_histogram': [0] * 30,
                }
                time_index = pd.date_range(end=current_time, periods=30, freq='5min')
                df_5m_window = pd.DataFrame(dummy_data, index=time_index)
            
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
        if 'close' in df_1m.columns:
            recent_prices = df_1m['close'].tail(25).values
        else:
            recent_prices = df_1m.iloc[:, 4].tail(25).values  # close 컬럼이 5번째라고 가정
        
        returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] for i in range(1, len(recent_prices))]
        return np.std(returns) * 100
    
    def generate_trading_signal(self, predicted_class: int, confidence: float, probabilities: np.ndarray, df_1m: pd.DataFrame) -> str:
        """예측 결과를 매매 신호로 변환 (백테스터의 should_buy와 완전히 동일한 로직)"""
        probs = probabilities  # [매도, 홀드, 매수] 확률
        signal = predicted_class  # 0: 매도, 1: 홀드, 2: 매수
        
        if 'close' in df_1m.columns:
            current_price = df_1m['close'].iloc[-1]
        else:
            current_price = df_1m.iloc[-1, 4]  # close 컬럼이 5번째라고 가정
            
        if 'timestamp' in df_1m.columns:
            current_time = pd.to_datetime(df_1m['timestamp'].iloc[-1])
        else:
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
            if 'close' in df_1m.columns:
                avg_price = np.mean(df_1m['close'].tail(15).values)  # 더 긴 기간 평균
            else:
                avg_price = np.mean(df_1m.iloc[-15:, 4].values)
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
            return f"📊 [{time_str}] 신호 없음 - 데이터 부족 (최소 30개 분봉 필요)"
        elif result["signal"] == "LOW_CONFIDENCE":
            return f"📊 [{time_str}] 신호 없음 - 낮은 신뢰도 ({result['confidence']:.1f}% < {self.min_confidence*100}%)"
        elif result["signal"] == "ERROR":
            return f"📊 [{time_str}] 신호 없음 - {result['prediction']}"
        elif result["signal"] == "BUY":
            return f"🟢 [{time_str}] 매수 신호 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "WEAK_BUY":
            return f"🟡 [{time_str}] 약한 매수 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "SELL":
            return f"🔴 [{time_str}] 매도 신호 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "WEAK_SELL":
            return f"🟠 [{time_str}] 약한 매도 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
        elif result["signal"] == "HOLD":
            return f"🟡 [{time_str}] 홀드 신호 - {result['prediction']} 예측 (신뢰도: {result['confidence']:.1f}%)"
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
    
    def load_realtime_data(self, csv_path: str) -> pd.DataFrame:
        """실시간 수집된 CSV 데이터 로드"""
        try:
            if not os.path.exists(csv_path):
                print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"✅ 실시간 데이터 로드 완료: {len(df)}개 분봉")
            return df
            
        except Exception as e:
            print(f"❌ 실시간 데이터 로드 실패: {str(e)}")
            return pd.DataFrame()
    
    def calculate_actual_label(self, current_close: float, next_close: float, threshold: float = 0.002) -> int:
        """실제 다음 분봉 결과를 레이블로 변환"""
        if pd.isna(current_close) or pd.isna(next_close):
            return -1  # 무효 데이터
        
        return_rate = (next_close - current_close) / current_close
        
        if return_rate > threshold:
            return 0  # 상승
        elif return_rate < -threshold:
            return 1  # 하락
        else:
            return 2  # 횡보
    
    def backtest_realtime_predictions(self, csv_1m_path: str, csv_5m_path: str, test_minutes: int = 30) -> dict:
        """실시간 수집 데이터로 예측 정확도 백테스트"""
        print(f"\n🔄 실시간 예측 정확도 백테스트 시작...")
        print(f"📊 테스트 대상: 최근 {test_minutes}개 분봉")
        print("="*60)
        
        # 1분봉 실시간 데이터 로드
        df_1m = self.load_realtime_data(csv_1m_path)
        if df_1m.empty or len(df_1m) < 100 + test_minutes:
            print(f"❌ 1분봉 데이터 부족: {len(df_1m)}개 (최소 {100 + test_minutes}개 필요)")
            return {}
        
        # 5분봉 실시간 데이터 로드
        df_5m = self.load_realtime_data(csv_5m_path)
        if df_5m.empty:
            print(f"⚠️ 5분봉 데이터 없음. 빈 DataFrame으로 처리합니다.")
            df_5m = pd.DataFrame()
        else:
            print(f"✅ 5분봉 데이터 로드 완료: {len(df_5m)}개")
            print(f"📅 5분봉 범위: {df_5m['timestamp'].min()} ~ {df_5m['timestamp'].max()}")
        
        predictions = []
        actuals = []
        results = []
        
        # 최근 데이터에서 백테스트 수행
        total_data_len = len(df_1m)
        start_idx = max(100, total_data_len - test_minutes - 1)
        end_idx = total_data_len - 1
        
        print(f"📈 백테스트 범위: 인덱스 {start_idx} ~ {end_idx} ({end_idx - start_idx + 1}개)")
        print()
        
        for i in range(start_idx, end_idx):
            try:
                # 현재 시점까지의 1분봉 데이터 (처음부터 현재까지 모든 데이터)
                current_1m_data = df_1m.iloc[:i+1].copy()
                
                if len(current_1m_data) < 100:  # 백테스터와 동일한 최소 요구사항
                    continue
                
                # 현재 시간과 다음 시간의 종가
                current_close = df_1m.iloc[i]['close']
                next_close = df_1m.iloc[i+1]['close'] if i+1 < len(df_1m) else None
                current_timestamp = df_1m.iloc[i]['timestamp']
                
                if next_close is None:
                    break
                
                # 현재 시점까지의 5분봉 데이터 (있는 경우)
                if not df_5m.empty:
                    current_time = pd.to_datetime(current_timestamp)
                    # 현재 시점 이전의 5분봉 데이터만 사용
                    current_5m_data = df_5m[df_5m['timestamp'] <= current_time].copy()
                else:
                    current_5m_data = pd.DataFrame()
                
                # AI 예측 수행 (실제 5분봉 데이터 사용)
                prediction_result = self.predict_next_candle(current_1m_data, current_5m_data)
                
                if prediction_result['signal'] not in ['BUY', 'SELL', 'HOLD', 'WEAK_BUY', 'WEAK_SELL', 'PRICE_BUY', 'NO_SIGNAL']:
                    continue
                
                # 실제 결과 계산
                actual_label = self.calculate_actual_label(current_close, next_close)
                if actual_label == -1:
                    continue
                
                # 예측 클래스 추출
                predicted_class = prediction_result.get('class', -1)
                if predicted_class == -1:
                    continue
                
                predictions.append(predicted_class)
                actuals.append(actual_label)
                
                # 예측 결과 저장
                is_correct = (predicted_class == actual_label)
                class_names = ["매도", "홀드", "매수"]  # 백테스터와 동일한 순서
                
                result_info = {
                    'timestamp': current_timestamp,
                    'current_price': current_close,
                    'next_price': next_close,
                    'predicted_class': predicted_class,
                    'predicted_label': class_names[predicted_class],
                    'actual_class': actual_label,
                    'actual_label': class_names[actual_label],
                    'confidence': prediction_result['confidence'],
                    'is_correct': is_correct,
                    'price_change': ((next_close - current_close) / current_close) * 100
                }
                results.append(result_info)
                
                # 실시간 결과 출력
                status_emoji = "✅" if is_correct else "❌"
                print(f"{status_emoji} [{current_timestamp.strftime('%H:%M')}] 예측: {class_names[predicted_class]} | 실제: {class_names[actual_label]} | 신뢰도: {prediction_result['confidence']:.1f}% | 가격변화: {result_info['price_change']:+.2f}%")
                
            except Exception as e:
                print(f"❌ 백테스트 오류 (인덱스 {i}): {str(e)}")
                continue
        
        if not predictions:
            print("❌ 유효한 예측 결과가 없습니다.")
            return {}
        
        # 정확도 계산
        total_predictions = len(predictions)
        correct_predictions = sum(1 for pred, actual in zip(predictions, actuals) if pred == actual)
        accuracy = (correct_predictions / total_predictions) * 100
        
        # 클래스별 정확도 계산
        class_names = ["매도", "홀드", "매수"]  # 백테스터와 동일한 순서
        class_accuracies = {}
        class_counts = {}
        
        for class_idx, class_name in enumerate(class_names):
            class_predictions = [pred for pred, actual in zip(predictions, actuals) if pred == class_idx]
            class_corrects = [1 for pred, actual in zip(predictions, actuals) if pred == class_idx and pred == actual]
            
            class_counts[class_name] = len(class_predictions)
            if len(class_predictions) > 0:
                class_accuracies[class_name] = (len(class_corrects) / len(class_predictions)) * 100
            else:
                class_accuracies[class_name] = 0
        
        # 결과 출력
        print("\n" + "="*60)
        print(f"📊 백테스트 결과 요약")
        print("="*60)
        print(f"🎯 전체 정확도: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
        print()
        print(f"📈 클래스별 정확도:")
        for class_name in class_names:
            count = class_counts[class_name]
            acc = class_accuracies[class_name]
            print(f"   {class_name}: {acc:.2f}% ({count}회 예측)")
        
        # 신뢰도별 정확도 분석
        high_conf_results = [r for r in results if r['confidence'] >= 70]
        if high_conf_results:
            high_conf_correct = sum(1 for r in high_conf_results if r['is_correct'])
            high_conf_accuracy = (high_conf_correct / len(high_conf_results)) * 100
            print(f"\n🔥 높은 신뢰도(≥70%) 예측: {high_conf_accuracy:.2f}% ({high_conf_correct}/{len(high_conf_results)})")
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'class_counts': class_counts,
            'results': results
        }

def main():
    """백테스트용 메인 함수"""
    predictor = RealtimePredictor()
    
    if predictor.model is None:
        print("❌ AI 모델 로딩 실패. 백테스트를 진행할 수 없습니다.")
        return
    
    # 실시간 수집 데이터 경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_1m_path = os.path.join(current_dir, 'REALTIME_DATA', 'BTC_1m_realtime.csv')
    csv_5m_path = os.path.join(current_dir, 'REALTIME_DATA', 'BTC_5m_realtime.csv')
    
    print(f"📂 사용할 데이터 파일:")
    print(f"   1분봉: {csv_1m_path}")
    print(f"   5분봉: {csv_5m_path}")
    
    # 백테스트 실행
    backtest_results = predictor.backtest_realtime_predictions(csv_1m_path, csv_5m_path, test_minutes=30)
    
    if backtest_results:
        print(f"\n✅ 백테스트 완료!")
        print(f"💡 이 결과는 실시간 수집된 최근 30개 분봉 데이터를 기반으로 합니다.")
        print(f"💡 각 시점에서 다음 1분봉의 상승/하락/횡보를 예측한 정확도입니다.")
        print(f"💡 백테스터와 동일한 알고리즘으로 1분봉+5분봉 데이터를 사용합니다.")
    else:
        print(f"\n❌ 백테스트 실패")
        print(f"💡 실시간 데이터 파일을 확인하고 충분한 데이터가 있는지 확인하세요.")
        print(f"💡 1분봉 경로: {csv_1m_path}")
        print(f"💡 5분봉 경로: {csv_5m_path}")

if __name__ == "__main__":
    main()