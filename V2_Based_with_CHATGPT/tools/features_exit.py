# tools/features_exit.py
from typing import Dict, List
import numpy as np
import pandas as pd
from .position_state import PositionState

EXIT_FEATURES_DEFAULT = [
    "r_unreal", "bars_held", "mfe", "dd_from_peak",
    "atr_rel", "rsi_14", "bb_width"
]

def build_exit_features_onebar(df: pd.DataFrame, i: int, state: PositionState) -> Dict[str, float]:
    """
    df: DataFrame with at least: open, high, low, close, and optional atr_14, rsi_14, bb_width
    i : current index
    """
    close = float(df["close"].iloc[i])
    high  = float(df["high"].iloc[i])
    low   = float(df["low"].iloc[i])

    state.update(high=high, low=low)

    side = state.side
    entry = state.entry_price

    r_unreal = (close/entry - 1.0) * side
    mfe = ((state.peak_price/entry) - 1.0) * side
    dd_from_peak = (mfe - r_unreal)

    atr_rel = float((df["atr_14"].iloc[i] if "atr_14" in df.columns else 0.0) / (close + 1e-9))
    rsi14 = float(df["rsi_14"].iloc[i] if "rsi_14" in df.columns else 50.0)
    bb_width = float(df["bb_width"].iloc[i] if "bb_width" in df.columns else 0.0)

    return {
        "r_unreal": r_unreal,
        "bars_held": int(state.bars_held),
        "mfe": mfe,
        "dd_from_peak": dd_from_peak,
        "atr_rel": atr_rel,
        "rsi_14": rsi14,
        "bb_width": bb_width,
    }

def features_to_row(feat: Dict[str, float], order: List[str]) -> np.ndarray:
    return np.array([[feat.get(k, 0.0) for k in order]], dtype=float)
