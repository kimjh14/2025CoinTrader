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
    
    def prepare_model_input(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> np.ndarray:
        """모델 입력용 데이터 준비"""
        try:
            if len(df_1m) < self.seq_length:
                print(f"⚠️ 데이터 부족: {len(df_1m)}개 (최소 {self.seq_length}개 필요)")
                return None
            
            # 최근 30개 분봉만 사용
            recent_1m = df_1m.tail(self.seq_length).copy()
            
            # 기술적 지표 선택 (모델 훈련 시와 동일한 컬럼)
            feature_columns = [
                'close', 'volume', 'macd', 'macd_signal', 'macd_histogram',
                'rsi', 'bb_upper', 'bb_lower', 'ma20', 'atr'
            ]
            
            # 존재하는 컬럼만 선택
            available_columns = [col for col in feature_columns if col in recent_1m.columns]
            if len(available_columns) < 5:
                print(f"⚠️ 필요한 기술적 지표 부족: {available_columns}")
                return None
                
            # 데이터 추출
            data = recent_1m[available_columns].values
            
            # NaN 처리
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=0.0)
            
            # 스케일링 (첫 번째 데이터로 학습)
            if not self.is_scaler_fitted:
                self.scaler.fit(data)
                self.is_scaler_fitted = True
            
            data_scaled = self.scaler.transform(data)
            
            # 배치 차원 추가 [1, seq_length, features]
            model_input = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)
            
            return model_input
            
        except Exception as e:
            print(f"❌ 모델 입력 데이터 준비 실패: {str(e)}")
            return None
    
    def predict_next_candle(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame) -> dict:
        """다음 분봉 예측"""
        if self.model is None:
            return {"signal": "NO_MODEL", "confidence": 0.0, "prediction": "모델 없음"}
        
        try:
            # 모델 입력 데이터 준비
            model_input = self.prepare_model_input(df_1m, df_5m)
            if model_input is None:
                return {"signal": "NO_DATA", "confidence": 0.0, "prediction": "데이터 부족"}
            
            # AI 예측 수행
            with torch.no_grad():
                outputs = self.model(model_input)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                max_probability = torch.max(probabilities).item()
            
            # 예측 결과 해석
            class_names = ["상승", "하락", "횡보"]
            prediction = class_names[predicted_class]
            confidence = max_probability * 100
            
            # 신뢰도 검증
            if confidence < self.min_confidence * 100:
                return {"signal": "LOW_CONFIDENCE", "confidence": confidence, "prediction": prediction}
            
            # 매매 신호 생성
            signal = self.generate_trading_signal(predicted_class, confidence)
            
            return {
                "signal": signal,
                "confidence": confidence,
                "prediction": prediction,
                "class": predicted_class,
                "probabilities": probabilities[0].cpu().numpy()
            }
            
        except Exception as e:
            print(f"❌ AI 예측 실패: {str(e)}")
            return {"signal": "ERROR", "confidence": 0.0, "prediction": f"오류: {str(e)}"}
    
    def generate_trading_signal(self, predicted_class: int, confidence: float) -> str:
        """예측 결과를 매매 신호로 변환"""
        confidence_ratio = confidence / 100.0
        
        if predicted_class == 0:  # 상승 예측
            if confidence_ratio >= self.buy_threshold:
                return "BUY"
            else:
                return "WEAK_BUY"
        elif predicted_class == 1:  # 하락 예측
            if confidence_ratio >= (1 - self.sell_threshold):  # 70% 이상
                return "SELL" 
            else:
                return "WEAK_SELL"
        else:  # 횡보 예측
            return "HOLD"
    
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

def main():
    """테스트용 메인 함수"""
    predictor = RealtimePredictor()
    
    # 샘플 데이터로 테스트
    print("\n🧪 샘플 데이터로 테스트 중...")
    
    # 더미 데이터 생성 (실제로는 realtime_data_collector에서 받음)
    sample_data = {
        'close': np.random.random(30) * 1000 + 159000,
        'volume': np.random.random(30) * 1000,
        'macd': np.random.random(30) * 100 - 50,
        'macd_signal': np.random.random(30) * 100 - 50,
        'macd_histogram': np.random.random(30) * 50 - 25,
        'rsi': np.random.random(30) * 100,
        'bb_upper': np.random.random(30) * 1000 + 159500,
        'bb_lower': np.random.random(30) * 1000 + 158500,
        'ma20': np.random.random(30) * 1000 + 159000,
        'atr': np.random.random(30) * 1000
    }
    
    df_1m = pd.DataFrame(sample_data)
    df_5m = pd.DataFrame(sample_data[:6])  # 5분봉은 더 적음
    
    # 예측 테스트
    result = predictor.analyze_realtime_data(df_1m, df_5m)
    print(result)

if __name__ == "__main__":
    main()