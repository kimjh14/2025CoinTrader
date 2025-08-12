"""
LSTM 모델 정의 모듈
"""
import torch
import torch.nn as nn
import numpy as np

class MultiTimeframeBiLSTM(nn.Module):
    """멀티 타임프레임 Bi-LSTM 모델"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, num_classes=3):
        super(MultiTimeframeBiLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bi-LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 출력 레이어
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # * 2 for bidirectional
        
    def forward(self, x):
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 마지막 타임스텝의 출력 사용
        last_output = lstm_out[:, -1, :]
        
        # 드롭아웃 적용
        output = self.dropout(last_output)
        
        # 최종 출력
        output = self.fc(output)
        
        return output