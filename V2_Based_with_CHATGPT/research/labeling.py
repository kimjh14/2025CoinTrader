import pandas as pd
import numpy as np

def triple_barrier(df: pd.DataFrame, atr_col:str="atr", k_up:float=0.8, k_dn:float=0.6, horizon:int=10):
    df = df.copy()
    up_bar = df["close"] + k_up * df[atr_col]
    dn_bar = df["close"] - k_dn * df[atr_col]

    n = len(df)
    tp_up = [pd.NaT]*n
    tp_dn = [pd.NaT]*n
    labels = [0]*n
    horizon_end = [pd.NaT]*n

    for i in range(n):
        end_idx = min(i + horizon, n-1)
        # horizon end time
        horizon_end[i] = df.loc[end_idx, "timestamp"]
        # loop forward (small horizon keeps cost OK)
        hit_up = hit_dn = None
        for j in range(i+1, end_idx+1):
            if df.loc[j, "high"] >= up_bar.iloc[i]:
                hit_up = df.loc[j, "timestamp"]
                break
            if df.loc[j, "low"] <= dn_bar.iloc[i]:
                hit_dn = df.loc[j, "timestamp"]
                break
        tp_up[i] = hit_up
        tp_dn[i] = hit_dn
        if hit_up is not None and hit_dn is None:
            labels[i] = 1
        elif hit_dn is not None and hit_up is None:
            labels[i] = -1
        else:
            labels[i] = 0

    df["tp_up"] = tp_up
    df["tp_dn"] = tp_dn
    df["horizon_end"] = horizon_end
    df["label_3c"] = labels
    return df