"""
PnL 계산 공용 유틸리티

체결 규칙: 다음 봉 시가 체결 (Next-Open)
- 시그널 발생 시 다음 봉의 시가에서 진입/청산
- 포지션 변경 시 수수료 부과
- 롱/숏 포지션 전환 시 이중 수수료 (청산 + 진입)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def calculate_next_open_pnl(
    prices: pd.DataFrame,
    positions: np.ndarray,
    fee: float = 0.0005,
    initial_capital: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    다음 봉 시가 체결 기준 PnL 계산
    
    Parameters:
    -----------
    prices : pd.DataFrame
        OHLCV 데이터 (open, close 컬럼 필요)
    positions : np.ndarray
        포지션 배열 (-1: 숏, 0: 중립, 1: 롱)
    fee : float
        거래 수수료 (0.0005 = 0.05%)
    initial_capital : float
        초기 자본
    
    Returns:
    --------
    equity_curve : np.ndarray
        누적 자산 곡선
    returns : np.ndarray
        각 시점별 수익률
    trades : np.ndarray
        거래 발생 시점 (0: 없음, 1: 거래)
    """
    n = len(positions)
    equity = np.ones(n) * initial_capital
    returns = np.zeros(n)
    trades = np.zeros(n)
    
    # 시가와 종가 가격 배열
    open_prices = prices['open'].values if 'open' in prices.columns else prices['close'].values
    close_prices = prices['close'].values
    
    # 포지션 시프트 (시그널 다음 봉에서 실행)
    pos_shift = np.roll(positions, 1)
    pos_shift[0] = 0
    
    current_pos = 0
    entry_price = 0
    
    for i in range(1, n):
        prev_pos = current_pos
        target_pos = pos_shift[i]
        
        # 포지션 변경 여부 확인
        if target_pos != prev_pos:
            trades[i] = 1
            
            # 기존 포지션 청산
            if prev_pos != 0:
                exit_price = open_prices[i]
                if prev_pos == 1:  # 롱 청산
                    pnl = (exit_price - entry_price) / entry_price - fee
                else:  # 숏 청산
                    pnl = (entry_price - exit_price) / entry_price - fee
                equity[i] = equity[i-1] * (1 + pnl)
            else:
                equity[i] = equity[i-1]
            
            # 새 포지션 진입
            if target_pos != 0:
                entry_price = open_prices[i]
                equity[i] *= (1 - fee)  # 진입 수수료
            
            current_pos = target_pos
            
        # 포지션 유지 중
        elif current_pos != 0:
            if current_pos == 1:  # 롱 포지션
                daily_return = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
            else:  # 숏 포지션
                daily_return = -(close_prices[i] - close_prices[i-1]) / close_prices[i-1]
            
            equity[i] = equity[i-1] * (1 + daily_return)
        else:
            equity[i] = equity[i-1]
        
        # 수익률 계산
        returns[i] = (equity[i] - equity[i-1]) / equity[i-1] if equity[i-1] > 0 else 0
    
    return equity, returns, trades


def calculate_equity_from_proba(
    proba: np.ndarray,
    prices: pd.DataFrame,
    th_long: float,
    th_short: float,
    fee: float,
    class_index: Dict[int, int],
    allow_short: bool = True
) -> float:
    """
    확률값과 임계치를 기반으로 최종 자산 계산
    train.py의 임계치 최적화에서 사용
    
    Parameters:
    -----------
    proba : np.ndarray
        모델 예측 확률 (n_samples, n_classes)
    prices : pd.DataFrame
        OHLCV 가격 데이터
    th_long : float
        롱 진입 임계치
    th_short : float
        숏 진입 임계치
    fee : float
        거래 수수료
    class_index : dict
        클래스 매핑 {-1: idx, 0: idx, 1: idx}
    allow_short : bool
        숏 포지션 허용 여부
    
    Returns:
    --------
    final_equity : float
        최종 자산 (배수)
    """
    # 확률값에서 포지션 결정
    positions = np.zeros(len(proba))
    
    for i in range(len(proba)):
        p_long = proba[i, class_index[1]] if 1 in class_index else 0
        p_short = proba[i, class_index[-1]] if -1 in class_index else 0
        
        if p_long >= th_long:
            positions[i] = 1
        elif allow_short and p_short >= th_short:
            positions[i] = -1
        else:
            positions[i] = 0
    
    # PnL 계산
    equity_curve, _, _ = calculate_next_open_pnl(prices, positions, fee)
    
    return equity_curve[-1]


def calculate_risk_metrics(equity_curve: np.ndarray, periods_per_year: int = 365*24*60) -> Dict[str, float]:
    """
    리스크 지표 계산
    
    Parameters:
    -----------
    equity_curve : np.ndarray
        누적 자산 곡선
    periods_per_year : int
        연간 기간 수 (분봉 기준 = 365*24*60)
    
    Returns:
    --------
    metrics : dict
        MDD, Sharpe, Calmar 등 리스크 지표
    """
    if len(equity_curve) < 2:
        return {
            "mdd": 0,
            "sharpe": 0,
            "calmar": 0,
            "max_consecutive_losses": 0,
            "max_consecutive_wins": 0
        }
    
    # 수익률 계산
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[~np.isnan(returns)]
    
    # Maximum Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    mdd = np.min(drawdown)
    
    # Sharpe Ratio (연율화)
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
    else:
        sharpe = 0
    
    # Calmar Ratio
    total_periods = len(equity_curve)
    years = total_periods / periods_per_year
    if years > 0 and mdd < 0:
        annual_return = (equity_curve[-1] / equity_curve[0]) ** (1/years) - 1
        calmar = annual_return / abs(mdd)
    else:
        calmar = 0
    
    # 연속 손실/이익 계산
    consecutive_losses = 0
    consecutive_wins = 0
    max_consecutive_losses = 0
    max_consecutive_wins = 0
    
    for r in returns:
        if r < 0:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        elif r > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
        else:
            consecutive_losses = 0
            consecutive_wins = 0
    
    return {
        "mdd": float(mdd),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "max_consecutive_losses": int(max_consecutive_losses),
        "max_consecutive_wins": int(max_consecutive_wins)
    }