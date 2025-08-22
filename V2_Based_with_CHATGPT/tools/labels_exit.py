# tools/labels_exit.py
from typing import List, Dict
import numpy as np
import pandas as pd

def class_index_from(classes: List[int]) -> Dict[int, int]:
    return {
        -1: classes.index(-1) if -1 in classes else None,
         0: classes.index(0)  if  0 in classes else None,
         1: classes.index(1)  if  1 in classes else None,
    }

def build_entries_from_proba(
    proba: np.ndarray,
    classes: List[int],
    long_enter: float,
    short_enter: float,
    allow_short: bool,
    cooldown: int = 30
) -> List[Dict]:
    ci = class_index_from(classes)
    idx_long = ci[1]
    idx_short = ci[-1]
    entries = []
    i = 0
    n = len(proba)
    while i < n:
        p_long = proba[i, idx_long] if idx_long is not None else 0.0
        p_short = proba[i, idx_short] if (allow_short and idx_short is not None) else 0.0

        if p_long >= long_enter:
            entries.append({"t0": i, "side": 1})
            i += cooldown
            continue
        if allow_short and p_short >= short_enter:
            entries.append({"t0": i, "side": -1})
            i += cooldown
            continue
        i += 1
    return entries

def label_optimal_exits(
    close: np.ndarray,
    entries: List[Dict],
    horizon: int = 60,
    fee: float = 0.0005,
    delta: float = 0.0005
) -> pd.DataFrame:
    rows = []
    n = len(close)
    for e in entries:
        t0 = e["t0"]
        side = e["side"]
        if t0+1 >= n: 
            continue
        end = min(n-1, t0 + horizon)
        entry_price = close[t0]
        best_t = None
        best_r = -1e9

        for t in range(t0+1, end+1):
            if side == 1:
                r = (close[t]/entry_price - 1.0) - 2*fee
            else:
                r = (entry_price/close[t] - 1.0) - 2*fee
            if r > best_r + delta:
                best_r = r
                best_t = t

        if best_t is None:
            continue

        for t in range(t0+1, min(end, best_t)+1):
            rows.append({
                "t": t, "side": side, "close_label": 1 if t == best_t else 0,
                "entry_price": float(entry_price), "t0": t0
            })

    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
