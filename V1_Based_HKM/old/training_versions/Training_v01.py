import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

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
    batch_size = 128
else:
    torch.set_num_threads(multiprocessing.cpu_count())
    device = 'cpu'
    print(f"Using {torch.get_num_threads()} CPU threads for PyTorch")
    batch_size = 64

print(f"CPU cores: {multiprocessing.cpu_count()}")
print(f"PyTorch threads: {torch.get_num_threads()}")

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != ce_loss.device:
                self.alpha = self.alpha.to(ce_loss.device)
            focal_loss = self.alpha[targets] * focal_loss
        
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class ImprovedDataPreprocessor:
    """
    개선된 데이터 전처리 클래스
    - 동적 임계값 사용 (분위수 기반)
    - 개선된 NaN 처리
    - SMOTE를 통한 클래스 균형
    """
    
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.lower_threshold = None
        self.upper_threshold = None
        
    def calculate_macd(self, data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """MACD 지표 계산"""
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_histogram
        }
    
    def create_labels(self, data: pd.DataFrame, is_training: bool = True) -> pd.Series:
        """
        동적 임계값을 사용한 3-클래스 레이블 생성
        - 훈련 시: 분위수 기반 임계값 계산 및 저장
        - 검증/테스트 시: 저장된 임계값 사용
        """
        returns = (data['close'].shift(-1) - data['close']) / data['close'] * 100
        
        if is_training:
            # 동적 임계값 계산 (훈련 데이터에서만)
            # 각 클래스가 약 33%씩 차지하도록 분위수 사용
            self.lower_threshold = returns.quantile(0.33)
            self.upper_threshold = returns.quantile(0.67)
            logger.info(f"동적 임계값 설정: 하락 < {self.lower_threshold:.4f}%, 상승 > {self.upper_threshold:.4f}%")
        else:
            # 검증/테스트 시 훈련에서 계산된 임계값 사용
            if self.lower_threshold is None or self.upper_threshold is None:
                raise ValueError("임계값이 설정되지 않았습니다. 먼저 훈련 데이터로 create_labels를 호출하세요.")
        
        labels = pd.Series(1, index=data.index)  # 기본값: 보류
        labels[returns < self.lower_threshold] = 0  # 하락
        labels[returns > self.upper_threshold] = 2   # 상승
        
        # 클래스 분포 로깅
        if is_training:
            class_dist = labels.value_counts(normalize=True).sort_index()
            logger.info(f"클래스 분포: 하락={class_dist.get(0, 0):.2%}, 보류={class_dist.get(1, 0):.2%}, 상승={class_dist.get(2, 0):.2%}")
        
        return labels
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """MACD 특징 생성"""
        features = pd.DataFrame(index=data.index)
        
        # MACD 지표 계산
        macd_dict = self.calculate_macd(data['close'])
        features['macd'] = macd_dict['macd']
        features['macd_signal'] = macd_dict['signal']
        features['macd_histogram'] = macd_dict['histogram']
        
        # NaN 처리: 첫 번째 유효한 값으로 백필
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
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE를 사용한 클래스 균형 맞추기"""
        # 시퀀스 데이터를 2D로 변환
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(n_samples, n_timesteps * n_features)
        
        # SMOTE 적용
        smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y)) - 1))
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
        
        # 다시 3D로 변환
        X_resampled = X_resampled.reshape(-1, n_timesteps, n_features)
        
        logger.info(f"SMOTE 적용 후: {len(X)} -> {len(X_resampled)} 샘플")
        logger.info(f"클래스 분포: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def split_data(self, features: pd.DataFrame, labels: pd.Series, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15,
                   use_smote: bool = True) -> Dict:
        """데이터 Train/Validation/Test 분할"""
        # DataFrame -> numpy 변환
        features_np = features.values
        labels_np = labels.values
        
        # NaN 처리 (보류가 아닌 각 클래스의 비율에 따라 채움)
        if np.any(np.isnan(labels_np)):
            # 유효한 레이블의 분포 계산
            valid_labels = labels_np[~np.isnan(labels_np)]
            class_probs = np.bincount(valid_labels.astype(int)) / len(valid_labels)
            # NaN을 확률적으로 채움
            nan_mask = np.isnan(labels_np)
            n_nans = np.sum(nan_mask)
            if n_nans > 0:
                random_labels = np.random.choice(len(class_probs), size=n_nans, p=class_probs)
                labels_np[nan_mask] = random_labels
        
        features_np = np.nan_to_num(features_np, nan=0.0)
        
        n_samples = len(labels_np)
        logger.info(f"처리된 데이터 샘플 수: {n_samples}")
        
        # 최소 샘플 수 보장
        min_samples_per_split = 100
        if n_samples < min_samples_per_split * 3:
            raise ValueError(f"데이터가 너무 적습니다. 최소 {min_samples_per_split * 3}개 필요, 현재 {n_samples}개")
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # 분할 크기 확인
        train_size = train_end
        val_size = val_end - train_end
        test_size = n_samples - val_end
        
        logger.info(f"데이터 분할: Train={train_size}, Val={val_size}, Test={test_size}")

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
        
        # SMOTE 적용 (훈련 데이터에만)
        if use_smote and len(np.unique(train_labels)) == 3:
            min_class_count = min(np.bincount(train_labels.astype(int)))
            if min_class_count > 5:  # SMOTE는 최소 6개 샘플 필요
                train_sequences, train_labels = self.apply_smote(train_sequences, train_labels)

        return {
            'X_train': train_sequences,
            'y_train': train_labels,
            'X_val': val_sequences,
            'y_val': val_labels,
            'X_test': test_sequences,
            'y_test': test_labels,
            'feature_names': ['macd', 'macd_signal', 'macd_histogram']
        }


class ImprovedBiLSTM(nn.Module):
    """개선된 Bi-LSTM 모델"""
    
    def __init__(self, input_size: int = 3, hidden_size: int = 128, num_layers: int = 2, 
                 num_classes: int = 3, dropout: float = 0.2):
        super(ImprovedBiLSTM, self).__init__()
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


class ImprovedBiLSTMPredictor:
    """
    개선된 Bi-LSTM 모델 클래스
    - Focal Loss 사용
    - F1 score 기반 early stopping
    - 개선된 클래스 가중치
    """
    
    def __init__(self, input_size: int = 3, 
                 hidden_size: int = 128, num_layers: int = 2, 
                 num_classes: int = 3, learning_rate: float = 0.001):
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        # GPU 메모리 최적화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for training")
        
        self.model = ImprovedBiLSTM(input_size, hidden_size, num_layers, num_classes).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # 모델을 DataParallel로 감싸서 멀티 GPU 지원
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Multi-GPU enabled: {torch.cuda.device_count()} GPUs")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 64) -> Dict:
        """모델 훈련 (개선된 손실 함수와 평가 지표)"""
        # 병렬처리 최적화 설정
        if torch.cuda.is_available():
            num_workers = min(8, multiprocessing.cpu_count())
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False
        
        logger.info(f"Using {num_workers} workers for data loading")
        
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
        
        # 검증 데이터는 미리 GPU로 전송
        X_val_tensor = torch.FloatTensor(X_val).to(self.device, non_blocking=True)
        y_val_tensor = torch.LongTensor(y_val).to(self.device, non_blocking=True)
        
        # sklearn의 compute_class_weight 사용하여 클래스 가중치 계산
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        # Focal Loss 사용
        self.criterion = FocalLoss(alpha=class_weights_tensor, gamma=2)
        
        logger.info(f"클래스 분포: {np.bincount(y_train.astype(int))}")
        logger.info(f"클래스 가중치: {class_weights}")
        
        # Optimizer and scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        # 훈련 기록
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_f1_scores = []
        
        logger.info(f"개선된 Bi-LSTM 훈련 시작: {epochs} epochs, batch_size={batch_size}")
        
        # Training loop with F1 score based early stopping
        best_val_f1 = 0
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
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
                
                # 정확도 계산
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_val_preds = []
            all_val_labels = []
            
            with torch.no_grad():
                val_batch_size = batch_size * 2
                for i in range(0, len(X_val_tensor), val_batch_size):
                    end_idx = min(i + val_batch_size, len(X_val_tensor))
                    
                    batch_X = X_val_tensor[i:end_idx]
                    batch_y = y_val_tensor[i:end_idx]
                    
                    val_outputs = self.model(batch_X)
                    val_loss += self.criterion(val_outputs, batch_y).item()
                    
                    _, val_predicted = val_outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += val_predicted.eq(batch_y).sum().item()
                    
                    all_val_preds.extend(val_predicted.cpu().numpy())
                    all_val_labels.extend(batch_y.cpu().numpy())
            
            # F1 score 계산
            val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / ((len(X_val_tensor) + val_batch_size - 1) // val_batch_size))
            val_accuracies.append(val_correct / val_total)
            val_f1_scores.append(val_f1)
            
            # Log results
            logger.info(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, '
                       f'Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}, '
                       f'Val F1: {val_f1:.4f}')
            
            # F1 score 기반 Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, 'best_improved_bilstm_model_v01.pth')
                logger.info(f"Best model saved with validation F1 score: {best_val_f1:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                # 최적 모델 로드
                checkpoint = torch.load('best_improved_bilstm_model_v01.pth')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                break
            
            # Scheduler step
            scheduler.step()
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_f1_scores': val_f1_scores
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            batch_size = 128
            for i in range(0, len(X), batch_size):
                end_idx = min(i + batch_size, len(X))
                
                batch_X = torch.FloatTensor(X[i:end_idx]).to(self.device, non_blocking=True)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """예측 확률"""
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            batch_size = 128
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
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_scores = f1_score(y_test, y_pred, average=None)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_scores': f1_scores,
            'f1_macro': f1_macro,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_report(y_test, y_pred)
        }
    
    def save_model(self, filepath: str) -> None:
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str) -> None:
        """모델 로드"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class ImprovedBacktestEngine:
    """개선된 백테스팅 클래스"""
    
    def __init__(self, initial_capital: float = 10000, transaction_cost: float = 0.0005):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def simulate_trading(self, predictions: np.ndarray, prices: pd.Series, 
                        timestamps: pd.DatetimeIndex) -> Dict:
        """거래 시뮬레이션"""
        capital = self.initial_capital
        position = 0  # 0: 현금, 1: 포지션 보유
        trades = []
        equity_curve = []
        
        for i in range(len(predictions)):
            current_price = prices.iloc[i]
            prediction = predictions[i]
            
            # 현재 자산 가치
            if position == 1 and i > 0:
                capital = capital * (current_price / prices.iloc[i-1])
            
            # 거래 신호
            if prediction == 2 and position == 0:  # 매수
                capital *= (1 - self.transaction_cost)
                position = 1
                trades.append({
                    'timestamp': timestamps[i],
                    'action': 'buy',
                    'price': current_price,
                    'capital': capital
                })
            elif prediction == 0 and position == 1:  # 매도
                capital *= (1 - self.transaction_cost)
                position = 0
                trades.append({
                    'timestamp': timestamps[i],
                    'action': 'sell',
                    'price': current_price,
                    'capital': capital
                })
            
            equity_curve.append(capital)
        
        # 성과 지표 계산
        equity_curve = np.array(equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        sharpe_ratio = np.sqrt(252 * 24 * 60) * returns.mean() / (returns.std() + 1e-8)  # 연간화
        
        # 최대 손실폭 (MDD)
        cumulative = equity_curve / self.initial_capital
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 승률
        winning_trades = sum(1 for t in trades if t['action'] == 'sell' and 
                           t['capital'] > trades[trades.index(t)-1]['capital'])
        win_rate = winning_trades / max(len([t for t in trades if t['action'] == 'sell']), 1)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades),
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def plot_results(self, backtest_results: Dict, evaluation_results: Dict) -> None:
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 누적 수익률
        ax1 = axes[0, 0]
        ax1.plot(backtest_results['equity_curve'], linewidth=2, color='blue')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Cumulative Returns')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. 성과 지표
        ax2 = axes[0, 1]
        metrics = ['Total Return', 'Sharpe Ratio', 'Win Rate', 'Max Drawdown']
        values = [
            backtest_results['total_return'] * 100,
            backtest_results['sharpe_ratio'],
            backtest_results['win_rate'] * 100,
            abs(backtest_results['max_drawdown']) * 100
        ]
        
        bars = ax2.bar(metrics, values, color=['green', 'blue', 'orange', 'red'])
        ax2.set_ylabel('Values')
        ax2.set_title('Performance Metrics')
        ax2.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. F1 Scores
        ax3 = axes[1, 0]
        classes = ['Down', 'Hold', 'Up']
        f1_scores = evaluation_results['f1_scores']
        bars = ax3.bar(classes, f1_scores, color=['red', 'gray', 'green'])
        ax3.set_title('F1 Scores by Class')
        ax3.set_ylabel('F1 Score')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # 값 표시
        for bar, score in zip(bars, f1_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 4. 혼동 행렬
        ax4 = axes[1, 1]
        sns.heatmap(evaluation_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax4)
        ax4.set_title('Confusion Matrix')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_xticklabels(['Down', 'Hold', 'Up'])
        ax4.set_yticklabels(['Down', 'Hold', 'Up'])
        
        plt.tight_layout()
        plt.savefig('improved_model_results_v01.png', dpi=300, bbox_inches='tight')
        plt.show()


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
    
    logger.info("=== 개선된 암호화폐 예측 모델 V01 훈련 시작 ===")
    logger.info(f"훈련 코인: {COINS}")
    logger.info(f"검증 코인: {VAL_COIN}")
    logger.info(f"기간: {START_DATE} ~ {END_DATE}")
    logger.info("개선사항: 동적 임계값, Focal Loss, SMOTE, F1 기반 Early Stopping")
    
    print("\n1. 데이터 로드 및 전처리 시작...")
    
    # 데이터 전처리기 초기화
    preprocessor = ImprovedDataPreprocessor(sequence_length=10)
    
    # 훈련 데이터 준비
    logger.info("훈련 데이터 준비 중...")
    
    # 각 코인별로 데이터 처리 및 결합
    all_coin_data = []
    
    for coin in tqdm(COINS, desc="코인 데이터 처리"):
        logger.info(f"   - {coin} 데이터 처리 중...")
        data = load_and_filter_data(coin, START_DATE, END_DATE)
        
        # 특징 생성
        features = preprocessor.prepare_features(data)
        labels = preprocessor.create_labels(data, is_training=True)  # 훈련 데이터로 임계값 설정
        
        # 각 코인의 데이터를 하나의 딕셔너리로 저장
        coin_data = {
            'features': features,
            'labels': labels,
            'coin': coin
        }
        all_coin_data.append(coin_data)
    
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
    logger.info(f"   - {VAL_COIN} 검증 데이터 처리 중...")
    val_data = load_and_filter_data(VAL_COIN, START_DATE, END_DATE)
    val_features = preprocessor.prepare_features(val_data)
    val_labels = preprocessor.create_labels(val_data, is_training=False)  # 저장된 임계값 사용
    
    logger.info(f"검증 데이터 준비 완료: {val_features.shape}")
    
    # 데이터 분할 (SMOTE 적용)
    logger.info("데이터 분할 시작...")
    data_splits = preprocessor.split_data(train_features, train_labels, use_smote=True)
    
    # 검증 데이터 처리
    val_features_scaled = preprocessor.scaler.transform(val_features)
    val_sequences, val_seq_labels = preprocessor.create_sequences(val_features_scaled, val_labels.values)
    
    logger.info("\n2. 개선된 Bi-LSTM 모델 훈련...")
    input_size = len(data_splits['feature_names'])  # 3 (MACD 지표들)
    lstm_model = ImprovedBiLSTMPredictor(input_size=input_size)
    
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
    logger.info(f"\n   개선된 Bi-LSTM Accuracy: {lstm_eval['accuracy']:.4f}")
    logger.info(f"   개선된 Bi-LSTM F1-scores: {lstm_eval['f1_scores']}")
    logger.info(f"   개선된 Bi-LSTM Macro F1: {lstm_eval['f1_macro']:.4f}")
    print("\n분류 보고서:")
    print(lstm_eval['classification_report'])
    
    logger.info("\n3. 백테스팅...")
    backtest_engine = ImprovedBacktestEngine()
    
    # Bi-LSTM 백테스팅
    logger.info("개선된 Bi-LSTM 백테스팅 중...")
    lstm_predictions = lstm_model.predict(val_sequences)
    
    # 예측 분포 확인
    pred_dist = pd.Series(lstm_predictions).value_counts(normalize=True).sort_index()
    logger.info(f"예측 분포: 하락={pred_dist.get(0, 0):.2%}, 보류={pred_dist.get(1, 0):.2%}, 상승={pred_dist.get(2, 0):.2%}")
    
    lstm_backtest = backtest_engine.simulate_trading(
        lstm_predictions,
        val_data['close'].iloc[10:],  # 시퀀스 길이만큼 제외
        val_data.index[10:]
    )
    
    logger.info("\n4. 성능 결과:")
    logger.info(f"   총 수익률: {lstm_backtest['total_return']*100:.2f}%")
    logger.info(f"   샤프 지수: {lstm_backtest['sharpe_ratio']:.4f}")
    logger.info(f"   최대 손실폭: {lstm_backtest['max_drawdown']*100:.2f}%")
    logger.info(f"   승률: {lstm_backtest['win_rate']*100:.2f}%")
    logger.info(f"   거래 횟수: {lstm_backtest['num_trades']}")
    
    # 시각화
    logger.info("\n5. 결과 시각화...")
    backtest_engine.plot_results(lstm_backtest, lstm_eval)
    
    # 훈련 히스토리 시각화
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(training_history['train_losses'], label='Train Loss')
    plt.plot(training_history['val_losses'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(training_history['val_accuracies'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(training_history['val_f1_scores'], label='Validation F1 Score', color='green')
    plt.title('Validation F1 Score (Macro)')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_training_history_v01.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 모델 저장
    logger.info("\n6. 모델 저장...")
    lstm_model.save_model('improved_bilstm_crypto_model_v01.pth')
    with open('preprocessor_v01.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    logger.info("\n=== 개선된 모델 V01 훈련 완료! ===")


if __name__ == "__main__":
    main()