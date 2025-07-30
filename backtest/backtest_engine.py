# backtest_engine.py (UPDATED)

import pandas as pd
import numpy as np
from typing import List, Optional
from .trade import Trade


class BacktestEngine:
    """Momentum Backtesting Engine - Revised for improved trailing stop and duration handling"""

    def __init__(self, signals_df, params):
        self.df = signals_df.copy()
        self.params = params
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.equity_curve = []
        self.initial_capital = params.get('initial_capital', 10000)
        self.current_capital = self.initial_capital
        self.commission = params.get('commission', 0.0003)
        self.slippage = params.get('slippage', 0.0001)

        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.reset_index(drop=True)

    def calculate_position_size(self, entry_price, stop_loss):
        risk_per_trade = self.params.get('risk_per_trade', 0.015)
        risk_capital = self.current_capital * risk_per_trade
        price_risk_per_unit = abs(entry_price - stop_loss)
        if price_risk_per_unit <= 0:
            return 0

        position_size_units = risk_capital / price_risk_per_unit
        max_position_value = self.current_capital * 0.95
        position_value = position_size_units * entry_price

        if position_value > max_position_value:
            position_size_units = max_position_value / entry_price

        return position_size_units

    def apply_costs(self, trade_value):
        commission_cost = abs(trade_value) * self.commission
        slippage_cost = abs(trade_value) * self.slippage
        return commission_cost + slippage_cost

    def run(self):
        print(f"Starting backtest with ${self.initial_capital:,.2f} initial capital...")
        try:
            df_array = self.df.to_dict('records')

            for i in range(len(self.df)):
                current_bar = self.df.iloc[i]
                current_bar_dict = df_array[i]

                self._process_bar(i, current_bar, current_bar_dict)

                current_equity = self.current_capital
                if self.current_trade and self.current_trade.is_open:
                    unrealized_pnl = self._calculate_unrealized_pnl(current_bar, self.current_trade)
                    current_equity += unrealized_pnl

                self.equity_curve.append({
                    'timestamp': current_bar.name if hasattr(current_bar, 'name') else i,
                    'equity': current_equity,
                    'trades_count': len(self.trades),
                    'open_trade': self.current_trade is not None
                })

            if self.current_trade and self.current_trade.is_open:
                last_bar = self.df.iloc[-1]
                close_price = last_bar['Close'] if 'Close' in last_bar else last_bar.get('close', 0)
                self._close_trade(last_bar.name if hasattr(last_bar, 'name') else len(self.df) - 1, close_price, 'End of data')

        except Exception as e:
            print(f"Error during backtest execution: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_empty_results()

        final_capital = self.current_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100

        print(f"Backtest completed:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Capital: ${final_capital:,.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Total Trades: {len(self.trades)}")

        return self._generate_results()

    def _calculate_unrealized_pnl(self, current_bar, trade):
        if not trade or not trade.is_open:
            return 0
        current_price = current_bar.get('Close') or current_bar.get('close', 0)
        if current_price == 0:
            return 0
        pnl_per_unit = current_price - trade.entry_price if trade.direction == 1 else trade.entry_price - current_price
        return pnl_per_unit * abs(trade.position_size)

    def _process_bar(self, i, bar, bar_dict):
        try:
            signal = bar_dict.get('Signal', 0)
            if signal != 0 and self.current_trade is None:
                self._open_trade(bar, bar_dict)
            if self.current_trade and self.current_trade.is_open:
                self._check_exits(i, bar, bar_dict)
        except Exception as e:
            print(f"Error processing bar {i}: {e}")

    def _open_trade(self, bar, bar_dict):
        try:
            entry_price = bar_dict.get('Entry_Price', 0)
            stop_loss = bar_dict.get('Stop_Loss', 0)
            take_profit = bar_dict.get('Take_Profit', 0)
            direction = bar_dict.get('Signal', 0)
            signal_strength = bar_dict.get('Signal_Strength', 1)

            if entry_price == 0 or stop_loss == 0:
                print(f"Invalid trade parameters: entry={entry_price}, stop={stop_loss}")
                return

            actual_entry_price = entry_price * (1 + self.slippage) if direction == 1 else entry_price * (1 - self.slippage)
            position_size_units = self.calculate_position_size(actual_entry_price, stop_loss)

            if position_size_units > 0:
                position_value = position_size_units * actual_entry_price
                required_capital = position_value + self.apply_costs(position_value)

                if required_capital <= self.current_capital:
                    actual_position_size = position_size_units if direction == 1 else -position_size_units
                    timestamp = bar.name if hasattr(bar, 'name') else pd.Timestamp.now()
                    self.current_trade = Trade(
                        entry_time=timestamp,
                        entry_price=actual_entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size=actual_position_size,
                        direction=direction,
                        signal_strength=signal_strength
                    )
                    self.current_capital -= self.apply_costs(position_value)
                    print(f"Opened {'Long' if direction == 1 else 'Short'} trade: Entry=${actual_entry_price:.2f}, Size={abs(actual_position_size):.2f}")
        except Exception as e:
            print(f"Error opening trade: {e}")

    def _check_exits(self, i, bar, bar_dict):
        if not self.current_trade or not self.current_trade.is_open:
            return

        try:
            trade = self.current_trade
            trade.duration_bars = i

            exit_price = None
            exit_reason = None

            high_price = bar_dict.get('High', 0)
            low_price = bar_dict.get('Low', 0)
            close_price = bar_dict.get('Close', 0)
            atr = self.df['ATR'].iloc[i] if 'ATR' in self.df.columns else 0

            # Trailing stop update
            if self.params.get('trailing_stop', True):
                if trade.direction == 1 and (high_price - trade.entry_price) > atr:
                    trade.update_trailing_stop(high_price, low_price, atr, trailing_mult=0.8)
                elif trade.direction == -1 and (trade.entry_price - low_price) > atr:
                    trade.update_trailing_stop(high_price, low_price, atr, trailing_mult=0.8)

            # Exit logic
            if trade.direction == 1:
                if low_price <= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = 'Stop Loss'
                elif high_price >= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = 'Take Profit'
                elif close_price < trade.trailing_stop:
                    exit_price = close_price
                    exit_reason = 'Trailing Stop'

            elif trade.direction == -1:
                if high_price >= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = 'Stop Loss'
                elif low_price <= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = 'Take Profit'
                elif close_price > trade.trailing_stop:
                    exit_price = close_price
                    exit_reason = 'Trailing Stop'

            # Max duration
            if exit_price is None and trade.duration_bars >= self.params.get('max_trade_duration', 24):
                exit_price = close_price
                exit_reason = 'Max Duration'

            if exit_price is not None and exit_price > 0:
                timestamp = bar.name if hasattr(bar, 'name') else pd.Timestamp.now()
                self._close_trade(timestamp, exit_price, exit_reason)

        except Exception as e:
            print(f"Error checking exits: {e}")

    def _close_trade(self, exit_time, exit_price, exit_reason):
        if not self.current_trade:
            return
        try:
            trade = self.current_trade

            actual_exit_price = exit_price * (1 - self.slippage) if trade.direction == 1 else exit_price * (1 + self.slippage)
            trade.close_trade(exit_time, actual_exit_price, exit_reason)

            pnl_per_unit = actual_exit_price - trade.entry_price if trade.direction == 1 else trade.entry_price - actual_exit_price
            gross_pnl = pnl_per_unit * abs(trade.position_size)
            position_value = abs(trade.position_size) * actual_exit_price
            exit_costs = self.apply_costs(position_value)
            net_pnl = gross_pnl - exit_costs

            trade.pnl = net_pnl
            if trade.entry_price > 0 and trade.position_size != 0:
                trade.pnl_pct = (net_pnl / (trade.entry_price * abs(trade.position_size))) * 100

            self.current_capital += net_pnl
            self.trades.append(trade)
            print(f"Closed {'Long' if trade.direction == 1 else 'Short'} trade: P&L=${net_pnl:.2f}, Reason={exit_reason}")
            self.current_trade = None
        except Exception as e:
            print(f"Error closing trade: {e}")

    def _generate_results(self):
        try:
            trades_df = pd.DataFrame([{
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'direction': 'Long' if t.direction == 1 else 'Short',
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'position_size': abs(t.position_size),
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'duration_bars': t.duration_bars,
                'exit_reason': t.exit_reason,
                'signal_strength': t.signal_strength
            } for t in self.trades])

            equity_df = pd.DataFrame(self.equity_curve)
            return {
                'trades': trades_df,
                'equity_curve': equity_df,
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
                'total_pnl': self.current_capital - self.initial_capital
            }
        except Exception as e:
            print(f"Error generating results: {e}")
            return self._generate_empty_results()

    def _generate_empty_results(self):
        return {
            'trades': pd.DataFrame(),
            'equity_curve': pd.DataFrame([{
                'timestamp': pd.Timestamp.now(),
                'equity': self.initial_capital,
                'trades_count': 0,
                'open_trade': False
            }]),
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital,
            'total_return': 0.0,
            'total_pnl': 0.0
        }