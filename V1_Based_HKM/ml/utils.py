"""
공통 유틸리티 함수들
"""
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple

def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
    """모델 성능 지표 계산"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions, average='macro', zero_division=0)
    recall = recall_score(actuals, predictions, average='macro', zero_division=0)
    f1 = f1_score(actuals, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def save_model_checkpoint(model: torch.nn.Module, optimizer, epoch: int, loss: float, path: str):
    """모델 체크포인트 저장"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model_checkpoint(model: torch.nn.Module, optimizer, path: str) -> Dict:
    """모델 체크포인트 로드"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return {
        'epoch': epoch,
        'loss': loss
    }

def format_currency(amount: float) -> str:
    """통화 형식으로 포맷팅"""
    return f"₩{amount:,.0f}"

def calculate_returns(prices: pd.Series) -> pd.Series:
    """수익률 계산"""
    return prices.pct_change().fillna(0)