from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    direction: int  # 1 for long, -1 for short
    signal_strength: int = 1  # 1 = weak, 2 = strong
    initial_stop: float = 0.0
    trailing_stop: float = 0.0
    highest_price: float = 0.0  # For long trailing
    lowest_price: float = 999999.0  # For short trailing
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_bars: int = 0

    def __post_init__(self):
        self.initial_stop = self.stop_loss
        self.trailing_stop = self.stop_loss
        if self.direction == 1:
            self.highest_price = self.entry_price
        else:
            self.lowest_price = self.entry_price

    @property
    def is_open(self):
        return self.exit_time is None

    @property
    def risk_amount(self):
        return abs(self.entry_price - self.initial_stop) * abs(self.position_size)

    def update_trailing_stop(self, current_high, current_low, atr, trailing_mult=0.8):
        """Update trailing stop based on current price action"""
        if self.direction == 1:  # Long trade
            if current_high > self.highest_price:
                self.highest_price = current_high
                new_trailing_stop = current_high - (atr * trailing_mult)
                if new_trailing_stop > self.trailing_stop:
                    self.trailing_stop = new_trailing_stop
                    self.stop_loss = self.trailing_stop
        else:  # Short trade
            if current_low < self.lowest_price:
                self.lowest_price = current_low
                new_trailing_stop = current_low + (atr * trailing_mult)
                if new_trailing_stop < self.trailing_stop:
                    self.trailing_stop = new_trailing_stop
                    self.stop_loss = self.trailing_stop

    def close_trade(self, exit_time, exit_price, exit_reason):
        """Close the trade"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        # Calculate PnL based on direction
        if self.direction == 1:  # Long
            self.pnl = (exit_price - self.entry_price) * self.position_size
        else:  # Short
            self.pnl = (self.entry_price - exit_price) * abs(self.position_size)

        self.pnl_pct = self.pnl / (self.entry_price * abs(self.position_size)) * 100
