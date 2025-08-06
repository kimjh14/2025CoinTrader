import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pickle
import os
import multiprocessing
import logging
from tqdm import tqdm

# GPU/CPU 최적화 설정
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = 'cuda'
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    batch_size = 256
else:
    torch.set_num_threads(multiprocessing.cpu_count())
    device = 'cpu'
    print(f"Using {torch.get_num_threads()} CPU threads for PyTorch")
    batch_size = 128

print(f"CPU cores: {multiprocessing.cpu_count()}")
print(f"PyTorch threads: {torch.get_num_threads()}")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BalancedSensitivePreprocessor:
    """
    균형잡힌 민감도를 가진 데이터 전처리 클래스
    - 0.05% 임계값 사용 (거래비용 대비 합리적)
    - 빠른 반응을 위한 MACD 파라미터
    - 적절한 거래 신호 생성
    """
    
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        # 균형잡힌 임계값 (거래비용 0.05% 고려)
        self.threshold = 0.05  # 0.05%
        
    def calculate_macd(self, data: pd.Series, 
                      fast_period: int = 8,   # 빠른 반응
                      slow_period: int = 17,  # 빠른 반응
                      signal_period: int = 5) -> Dict[str, pd.Series]:
        """빠른 반응을 위한 MACD 지표 계산"""
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        # 정규화
        macd_line = (macd_line - macd_line.mean()) / (macd_line.std() + 1e-8)
        signal_line = (signal_line - signal_line.mean()) / (signal_line.std() + 1e-8)
        macd_histogram = (macd_histogram - macd_histogram.mean()) / (macd_histogram.std() + 1e-8)
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_histogram
        }
    
    def create_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        균형잡힌 3-클래스 레이블 생성
        - 클래스 0 (하락): 1분 뒤 수익률 < -0.05%
        - 클래스 1 (보류): -0.05% ≤ 1분 뒤 수익률 ≤ 0.05%
        - 클래스 2 (상승): 1분 뒤 수익률 > 0.05%
        """
        returns = (data['close'].shift(-1) - data['close']) / data['close'] * 100
        labels = pd.Series(1, index=data.index)  # 기본값: 보류
        labels[returns < -self.threshold] = 0  # 하락
        labels[returns > self.threshold] = 2   # 상승
        
        # 클래스 분포 로깅
        class_dist = labels.value_counts(normalize=True).sort_index()
        logger.info(f"균형잡힌 레이블 분포 (±{self.threshold}%): 하락={class_dist.get(0, 0):.2%}, 보류={class_dist.get(1, 0):.2%}, 상승={class_dist.get(2, 0):.2%}")
        
        # 거래 신호 개수 계산
        trading_signals = len(labels[labels != 1])
        logger.info(f"생성된 거래 신호: {trading_signals}개 ({trading_signals/len(labels)*100:.2%})")
        
        return labels
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """MACD 특징 생성"""
        features = pd.DataFrame(index=data.index)
        
        # MACD 지표 계산
        macd_dict = self.calculate_macd(data['close'])
        features['macd'] = macd_dict['macd']
        features['macd_signal'] = macd_dict['signal']
        features['macd_histogram'] = macd_dict['histogram']
        
        # 추가: 가격 변화율
        features['price_change'] = data['close'].pct_change() * 100
        features['price_change_ma'] = features['price_change'].rolling(5).mean()
        
        # NaN 처리
        features = features.bfill().ffill()
        
        return features
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 생성"""
        sequences = []
        sequence_labels = []
        
        for i in range(self.sequence_length, len(features)):
            sequences.append(features[i-self.sequence_length:i])
            sequence_labels.append(labels[i])
        
        return np.array(sequences), np.array(sequence_labels)
    
    def split_data(self, features: pd.DataFrame, labels: pd.Series, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict:
        """데이터 Train/Validation/Test 분할"""
        # DataFrame -> numpy 변환
        features_np = features.values
        labels_np = labels.values
        
        # NaN 처리
        features_np = np.nan_to_num(features_np, nan=0.0)
        
        n_samples = len(labels_np)
        logger.info(f"처리된 데이터 샘플 수: {n_samples}")
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # 정규화
        X_train = features_np[:train_end]
        X_val = features_np[train_end:val_end]
        X_test = features_np[val_end:]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        y_train = labels_np[:train_end]
        y_val = labels_np[train_end:val_end]
        y_test = labels_np[val_end:]

        # LSTM용 시퀀스 데이터
        train_sequences, train_labels = self.create_sequences(X_train_scaled, y_train)
        val_sequences, val_labels = self.create_sequences(X_val_scaled, y_val)
        test_sequences, test_labels = self.create_sequences(X_test_scaled, y_test)

        return {
            'X_train': train_sequences,
            'y_train': train_labels,
            'X_val': val_sequences,
            'y_val': val_labels,
            'X_test': test_sequences,
            'y_test': test_labels,
            'feature_names': ['macd', 'macd_signal', 'macd_histogram', 'price_change', 'price_change_ma']
        }


class BalancedBiLSTM(nn.Module):
    """균형잡힌 Bi-LSTM 모델"""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 128, num_layers: int = 2, 
                 num_classes: int = 3, dropout: float = 0.2):
        super(BalancedBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 출력층
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # LSTM 처리
        lstm_out, _ = self.lstm(x)
        
        # Attention weights 계산
        attention_weights = self.attention(lstm_out)
        
        # Attention 적용
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Dropout and fully connected
        out = self.dropout(context)
        out = self.fc(out)
        
        return out


class BalancedLSTMPredictor:
    """
    균형잡힌 Bi-LSTM 모델 클래스
    - 적절한 확률 임계값
    - 균형잡힌 손실 함수
    - 실용적인 거래 전략
    """
    
    def __init__(self, input_size: int = 5, 
                 hidden_size: int = 128, num_layers: int = 2, 
                 num_classes: int = 3, learning_rate: float = 0.001):
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.probability_threshold = 0.4  # 적절한 확률 임계값
        
        # GPU 메모리 최적화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model = BalancedBiLSTM(input_size, hidden_size, num_layers, num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 모델을 DataParallel로 감싸서 멀티 GPU 지원
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Multi-GPU enabled: {torch.cuda.device_count()} GPUs")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 256) -> Dict:
        """모델 훈련"""
        # 병렬처리 최적화 설정
        if torch.cuda.is_available():
            num_workers = min(8, multiprocessing.cpu_count())
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False
        
        # 데이터 로더 생성
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
        
        # 검증 데이터
        X_val_tensor = torch.FloatTensor(X_val).to(self.device, non_blocking=True)
        y_val_tensor = torch.LongTensor(y_val).to(self.device, non_blocking=True)
        
        # 균형잡힌 클래스 가중치
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        
        # 클래스 분포 확인
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"훈련 데이터 클래스 분포: {dict(zip(unique, counts))}")
        logger.info(f"클래스 가중치: {class_weights}")
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # 훈련 기록
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_signal_ratios = []
        val_f1_scores = []
        
        logger.info(f"균형잡힌 Bi-LSTM 훈련 시작 (임계값 ±{0.05}%): {epochs} epochs")
        
        # Training loop
        best_f1_score = 0
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_X, batch_y in train_pbar:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
                
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Validation
            self.model.eval()
            val_loss = 0
            all_val_preds = []
            all_val_probs = []
            
            with torch.no_grad():
                val_batch_size = batch_size * 2
                for i in range(0, len(X_val_tensor), val_batch_size):
                    end_idx = min(i + val_batch_size, len(X_val_tensor))
                    
                    batch_X = X_val_tensor[i:end_idx]
                    batch_y = y_val_tensor[i:end_idx]
                    
                    val_outputs = self.model(batch_X)
                    val_loss += self.criterion(val_outputs, batch_y).item()
                    
                    # 예측
                    _, preds = torch.max(val_outputs, 1)
                    all_val_preds.extend(preds.cpu().numpy())
                    
                    # 확률
                    probs = torch.softmax(val_outputs, dim=1)
                    all_val_probs.extend(probs.cpu().numpy())
            
            # 평가 지표 계산
            val_accuracy = accuracy_score(y_val, all_val_preds)
            val_f1 = f1_score(y_val, all_val_preds, average='macro')
            signal_ratio = np.sum(np.array(all_val_preds) != 1) / len(all_val_preds)
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / ((len(X_val_tensor) + val_batch_size - 1) // val_batch_size))
            val_accuracies.append(val_accuracy)
            val_f1_scores.append(val_f1)
            val_signal_ratios.append(signal_ratio)
            
            # Log results
            logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, '
                       f'Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, '
                       f'Signal Ratio: {signal_ratio:.2%}')
            
            # F1 score 기반 저장
            if val_f1 > best_f1_score:
                best_f1_score = val_f1
                patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'probability_threshold': self.probability_threshold
                }, 'best_balanced_model_v03.pth')
                logger.info(f"Best model saved with F1 score: {best_f1_score:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
            
            # Scheduler step
            scheduler.step(val_f1)
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_signal_ratios': val_signal_ratios,
            'val_f1_scores': val_f1_scores
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            batch_size = 256
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                
                batch_X = torch.FloatTensor(X[i:end_idx]).to(self.device, non_blocking=True)
                outputs = self.model(batch_X)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """예측 확률"""
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            batch_size = 256
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                
                batch_X = torch.FloatTensor(X[i:end_idx]).to(self.device, non_blocking=True)
                outputs = self.model(batch_X)
                proba = torch.softmax(outputs, dim=1)
                probabilities.extend(proba.cpu().numpy())
        
        return np.array(probabilities)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """모델 평가"""
        y_pred = self.predict(X_test)
        
        # 거래 신호 통계
        signal_count = np.sum(y_pred != 1)
        signal_ratio = signal_count / len(y_pred)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_scores = f1_score(y_test, y_pred, average=None)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_scores': f1_scores,
            'f1_macro': f1_macro,
            'confusion_matrix': conf_matrix,
            'signal_count': signal_count,
            'signal_ratio': signal_ratio,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'probability_threshold': self.probability_threshold
        }, filepath)


def load_and_filter_data(coin: str, start_date: str, end_date: str, data_dir: str = './data/raw') -> pd.DataFrame:
    """1분봉 데이터만 로드 및 날짜 필터링"""
    filepath = os.path.join(data_dir, f'{coin}_1m.csv')
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    # 날짜 필터링
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    return df


def main():
    """메인 실행 코드"""
    # 설정
    START_DATE = '2025-05-03'
    END_DATE = '2025-07-30'
    COINS = ['APT', 'BTC', 'DOGE', 'HBAR', 'SOL', 'XRP']  # 훈련용
    VAL_COIN = 'SUI'  # 검증용
    
    logger.info("=== 균형잡힌 암호화폐 예측 모델 V03 훈련 시작 ===")
    logger.info(f"훈련 코인: {COINS}")
    logger.info(f"검증 코인: {VAL_COIN}")
    logger.info(f"기간: {START_DATE} ~ {END_DATE}")
    logger.info("특징: 0.05% 임계값, 균형잡힌 거래 전략")
    
    print("\n1. 데이터 로드 및 전처리 시작...")
    
    # 데이터 전처리기 초기화
    preprocessor = BalancedSensitivePreprocessor(sequence_length=10)
    
    # 훈련 데이터 준비
    logger.info("훈련 데이터 준비 중...")
    
    # 각 코인별로 데이터 처리 및 결합
    all_coin_data = []
    total_signals = 0
    
    for coin in tqdm(COINS, desc="코인 데이터 처리"):
        logger.info(f"\n   - {coin} 데이터 처리 중...")
        data = load_and_filter_data(coin, START_DATE, END_DATE)
        
        # 특징 생성
        features = preprocessor.prepare_features(data)
        labels = preprocessor.create_labels(data)
        
        # 거래 신호 수 확인
        coin_signals = len(labels[labels != 1])
        total_signals += coin_signals
        logger.info(f"   {coin} 거래 신호: {coin_signals}개")
        
        # 각 코인의 데이터를 하나의 딕셔너리로 저장
        coin_data = {
            'features': features,
            'labels': labels,
            'coin': coin
        }
        all_coin_data.append(coin_data)
    
    logger.info(f"\n총 훈련 거래 신호: {total_signals}개")
    
    # 모든 코인의 데이터를 순차적으로 결합
    train_features_list = []
    train_labels_list = []
    
    for coin_data in all_coin_data:
        train_features_list.append(coin_data['features'])
        train_labels_list.append(coin_data['labels'])
    
    # 결합
    train_features = pd.concat(train_features_list, axis=0, ignore_index=True)
    train_labels = pd.concat(train_labels_list, axis=0, ignore_index=True)
    
    logger.info(f"훈련 데이터 결합 완료: {train_features.shape}")
    
    # 검증 데이터 준비
    logger.info(f"\n   - {VAL_COIN} 검증 데이터 처리 중...")
    val_data = load_and_filter_data(VAL_COIN, START_DATE, END_DATE)
    val_features = preprocessor.prepare_features(val_data)
    val_labels = preprocessor.create_labels(val_data)
    
    val_signals = len(val_labels[val_labels != 1])
    logger.info(f"   {VAL_COIN} 거래 신호: {val_signals}개")
    logger.info(f"검증 데이터 준비 완료: {val_features.shape}")
    
    # 데이터 분할
    logger.info("\n데이터 분할 시작...")
    data_splits = preprocessor.split_data(train_features, train_labels)
    
    # 검증 데이터 처리
    val_features_scaled = preprocessor.scaler.transform(val_features)
    val_sequences, val_seq_labels = preprocessor.create_sequences(val_features_scaled, val_labels.values)
    
    logger.info("\n2. 균형잡힌 Bi-LSTM 모델 훈련...")
    input_size = len(data_splits['feature_names'])  # 5 (MACD + 가격 변화율)
    lstm_model = BalancedLSTMPredictor(input_size=input_size)
    
    training_history = lstm_model.train(
        data_splits['X_train'],
        data_splits['y_train'],
        data_splits['X_val'],
        data_splits['y_val'],
        epochs=50,
        batch_size=batch_size
    )
    
    # 모델 평가
    lstm_eval = lstm_model.evaluate(val_sequences, val_seq_labels)
    logger.info(f"\n   균형잡힌 Bi-LSTM Accuracy: {lstm_eval['accuracy']:.4f}")
    logger.info(f"   거래 신호 수: {lstm_eval['signal_count']}개")
    logger.info(f"   거래 신호 비율: {lstm_eval['signal_ratio']:.2%}")
    logger.info(f"   Macro F1 Score: {lstm_eval['f1_macro']:.4f}")
    logger.info(f"   F1-scores: {lstm_eval['f1_scores']}")
    print("\n분류 보고서:")
    print(lstm_eval['classification_report'])
    
    # 예측 분포 확인
    predictions = lstm_model.predict(val_sequences)
    pred_dist = pd.Series(predictions).value_counts(normalize=True).sort_index()
    logger.info(f"\n예측 분포: 하락={pred_dist.get(0, 0):.2%}, 보류={pred_dist.get(1, 0):.2%}, 상승={pred_dist.get(2, 0):.2%}")
    
    # 시각화
    logger.info("\n3. 결과 시각화...")
    
    # 훈련 히스토리 시각화
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(training_history['train_losses'], label='Train Loss')
    plt.plot(training_history['val_losses'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 2)
    plt.plot(training_history['val_accuracies'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 3)
    plt.plot(training_history['val_f1_scores'], label='F1 Score', color='green')
    plt.title('Validation F1 Score (Macro)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 4)
    plt.plot([x * 100 for x in training_history['val_signal_ratios']], label='Signal Ratio', color='purple')
    plt.title('Validation Signal Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Signal Ratio (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('balanced_training_history_v03.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(lstm_eval['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Balanced Model V03')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0.5, 1.5, 2.5], ['Down', 'Hold', 'Up'])
    plt.yticks([0.5, 1.5, 2.5], ['Down', 'Hold', 'Up'])
    plt.savefig('balanced_confusion_matrix_v03.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 모델 저장
    logger.info("\n4. 모델 저장...")
    lstm_model.save_model('balanced_bilstm_crypto_model_v03.pth')
    with open('balanced_preprocessor_v03.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    logger.info("\n=== 균형잡힌 모델 V03 훈련 완료! ===")
    logger.info(f"최종 거래 신호: {lstm_eval['signal_count']}개 ({lstm_eval['signal_ratio']:.2%})")
    logger.info(f"임계값: ±0.05% (거래비용 대비 합리적)")


if __name__ == "__main__":
    main()