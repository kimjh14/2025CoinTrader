"""
데이터 전처리 모듈
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List

class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, scaler_type='standard'):
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
    
    def create_sequences(self, data: np.ndarray, sequence_length: int, target_col: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 시퀀스 생성"""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:(i + sequence_length)]
            target = data[i + sequence_length, target_col]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정규화"""
        normalized_data = self.scaler.fit_transform(data)
        return pd.DataFrame(normalized_data, columns=data.columns, index=data.index)
    
    def prepare_multitimeframe_data(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, df_15m: pd.DataFrame) -> pd.DataFrame:
        """멀티 타임프레임 데이터 준비"""
        # 각 타임프레임 데이터를 1분 단위로 리샘플링하여 병합
        # 구현 세부사항은 기존 코드 참조
        pass