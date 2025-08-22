# tools/position_state.py
from dataclasses import dataclass

@dataclass
class PositionState:
    entry_idx: int
    entry_price: float
    side: int            # 1=LONG, -1=SHORT
    bars_held: int = 0
    peak_price: float = None
    trough_price: float = None

    def __post_init__(self):
        self.peak_price = self.entry_price
        self.trough_price = self.entry_price

    def update(self, high: float, low: float):
        self.bars_held += 1
        if self.side == 1:
            if high > self.peak_price: self.peak_price = high
            if low < self.trough_price: self.trough_price = low
        else:
            if low < self.trough_price: self.trough_price = low
            if high > self.peak_price: self.peak_price = high
