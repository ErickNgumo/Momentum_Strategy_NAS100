# strategies/improved_momentum.py
# ====================================
# Simplified and Improved NAS100 Momentum Strategy
# ====================================

import pandas as pd
import numpy as np
from .indicators import TechnicalIndicators


class ImprovedMomentumStrategy:
    """Simplified momentum strategy focusing on clean breakouts with trend confirmation"""

    def __init__(self, params):
        self.params = params
        self.indicators = TechnicalIndicators()

    def generate_signals(self, df):
        """Generate trading signals with simplified, effective logic"""
        if df is None or len(df) == 0:
            print("Error: Empty or None DataFrame provided")
            return pd.DataFrame()

        try:
            df = df.copy()

            # Add indicators with simplified parameters
            df = self.indicators.add_all_indicators(
                df,
                lookback_high=self.params.get("lookback_high", 20),
                lookback_low=self.params.get("lookback_low", 20)
            )

            # Initialize signal columns
            df['Signal'] = 0
            df['Entry_Price'] = np.nan
            df['Stop_Loss'] = np.nan
            df['Take_Profit'] = np.nan
            df['Signal_Strength'] = 0
            df['Exit_Reason'] = ''

            # Core conditions for momentum strategy
            trend_up = df['EMA_21'] > df['EMA_50']
            trend_down = df['EMA_21'] < df['EMA_50']

            # Major trend filter (optional but recommended)
            if self.params.get('use_trend_filter', True):
                major_trend_up = df['EMA_50'] > df['EMA_200']
                major_trend_down = df['EMA_50'] < df['EMA_200']
            else:
                major_trend_up = pd.Series(True, index=df.index)
                major_trend_down = pd.Series(True, index=df.index)

            # Volume confirmation
            volume_ok = df['Volume_Ratio'] > self.params.get('min_volume_ratio', 1.0)
            volume_surge = df['Volume_Ratio'] > self.params.get('volume_multiplier', 1.3)

            # Trend strength (ADX)
            strong_trend = df['ADX'] > self.params.get('adx_threshold', 25)

            # ATR filter for volatility
            min_atr = self.params.get('min_atr', 2.0)
            atr_ok = df['ATR'] > min_atr

            # === LONG SIGNALS ===
            # Clean breakout above recent high
            breakout_up = df['Close'] > df['Rolling_High'].shift(1)

            # Previous bar was NOT already above the high (avoid late entries)
            clean_break_up = df['Close'].shift(1) <= df['Rolling_High'].shift(1)

            # Momentum confirmation
            momentum_up = df['MACD'] > df['MACD_Signal']

            # Combine conditions for long signals
            long_conditions = (
                    breakout_up &  # Price breaks above recent high
                    clean_break_up &  # Clean breakout (not already broken)
                    trend_up &  # Short-term trend is up
                    major_trend_up &  # Major trend is up (if enabled)
                    volume_ok &  # Minimum volume requirement
                    strong_trend &  # ADX shows trending market
                    atr_ok &  # Sufficient volatility
                    momentum_up  # MACD momentum confirmation
            )

            # Strong vs weak long signals
            strong_long = long_conditions & volume_surge  # Volume surge for strong signals
            weak_long = long_conditions & ~volume_surge  # Regular volume for weak signals

            # === SHORT SIGNALS ===
            # Clean breakdown below recent low
            breakout_down = df['Close'] < df['Rolling_Low'].shift(1)

            # Previous bar was NOT already below the low
            clean_break_down = df['Close'].shift(1) >= df['Rolling_Low'].shift(1)

            # Momentum confirmation
            momentum_down = df['MACD'] < df['MACD_Signal']

            # Combine conditions for short signals
            short_conditions = (
                    breakout_down &  # Price breaks below recent low
                    clean_break_down &  # Clean breakdown
                    trend_down &  # Short-term trend is down
                    major_trend_down &  # Major trend is down (if enabled)
                    volume_ok &  # Minimum volume requirement
                    strong_trend &  # ADX shows trending market
                    atr_ok &  # Sufficient volatility
                    momentum_down  # MACD momentum confirmation
            )

            # Strong vs weak short signals
            strong_short = short_conditions & volume_surge
            weak_short = short_conditions & ~volume_surge

            # === ASSIGN SIGNALS ===
            # Strong signals (signal strength = 2)
            df.loc[strong_long, ['Signal', 'Signal_Strength']] = [1, 2]
            df.loc[strong_short, ['Signal', 'Signal_Strength']] = [-1, 2]

            # Weak signals (signal strength = 1) - only if no strong signal
            weak_long_final = weak_long & (df['Signal'] == 0)
            weak_short_final = weak_short & (df['Signal'] == 0)

            df.loc[weak_long_final, ['Signal', 'Signal_Strength']] = [1, 1]
            df.loc[weak_short_final, ['Signal', 'Signal_Strength']] = [-1, 1]

            # === SET ENTRY PRICES ===
            signal_mask = df['Signal'] != 0
            df.loc[signal_mask, 'Entry_Price'] = df.loc[signal_mask, 'Close']

            # === CALCULATE STOP LOSS AND TAKE PROFIT ===
            self._calculate_exit_levels(df)

            return df

        except Exception as e:
            print(f"Error in generate_signals: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_signals_df(df) if df is not None else pd.DataFrame()

    def _calculate_exit_levels(self, df):
        """Calculate stop loss and take profit levels"""
        signal_mask = df['Signal'] != 0

        # Get parameters
        stop_loss_r = self.params.get('stop_loss_r', 1.0)
        take_profit_r = self.params.get('take_profit_r', 2.5)
        atr_mult = self.params.get('atr_mult', 2.0)

        for idx in df[signal_mask].index:
            entry_price = df.at[idx, 'Close']
            atr = df.at[idx, 'ATR']
            signal_strength = df.at[idx, 'Signal_Strength']
            direction = df.at[idx, 'Signal']

            # Adjust multipliers based on signal strength
            strength_multiplier = 0.8 if signal_strength == 2 else 1.2
            stop_multiplier = atr_mult * strength_multiplier

            # Calculate stop loss and take profit
            if direction == 1:  # Long
                stop_loss = entry_price - (atr * stop_multiplier * stop_loss_r)
                take_profit = entry_price + (atr * stop_multiplier * take_profit_r)
            else:  # Short
                stop_loss = entry_price + (atr * stop_multiplier * stop_loss_r)
                take_profit = entry_price - (atr * stop_multiplier * take_profit_r)

            df.at[idx, 'Stop_Loss'] = stop_loss
            df.at[idx, 'Take_Profit'] = take_profit

    def _create_empty_signals_df(self, df):
        """Create empty signals dataframe with required columns"""
        df = df.copy() if df is not None else pd.DataFrame()
        if len(df) > 0:
            df['Signal'] = 0
            df['Entry_Price'] = np.nan
            df['Stop_Loss'] = np.nan
            df['Take_Profit'] = np.nan
            df['Signal_Strength'] = 0
            df['Exit_Reason'] = ''
        return df

    def get_strategy_stats(self, df):
        """Get strategy statistics"""
        try:
            if df is None or len(df) == 0:
                return self._empty_stats()

            # Count signals
            long_signals = len(df[df['Signal'] == 1])
            short_signals = len(df[df['Signal'] == -1])
            total_signals = long_signals + short_signals

            # Market conditions
            if 'EMA_50' in df.columns and 'EMA_200' in df.columns:
                bullish_pct = (df['EMA_50'] > df['EMA_200']).mean() * 100
                bearish_pct = (df['EMA_50'] < df['EMA_200']).mean() * 100
            else:
                bullish_pct = bearish_pct = 0

            # Average ATR
            avg_atr = df['ATR'].mean() if 'ATR' in df.columns else 0

            # High ADX percentage
            adx_threshold = self.params.get('adx_threshold', 25)
            high_adx_pct = (df['ADX'] > adx_threshold).mean() * 100 if 'ADX' in df.columns else 0

            return {
                'total_signals': total_signals,
                'long_signals': long_signals,
                'short_signals': short_signals,
                'avg_atr': round(avg_atr, 2),
                'bullish_trend_pct': round(bullish_pct, 1),
                'bearish_trend_pct': round(bearish_pct, 1),
                'high_adx_pct': round(high_adx_pct, 1)
            }

        except Exception as e:
            print(f"Error calculating strategy stats: {e}")
            return self._empty_stats()

    def _empty_stats(self):
        """Return empty statistics dictionary"""
        return {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'avg_atr': 0,
            'bullish_trend_pct': 0,
            'bearish_trend_pct': 0,
            'high_adx_pct': 0
        }

