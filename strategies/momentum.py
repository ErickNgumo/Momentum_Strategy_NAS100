# strategies/momentum.py - FIXED VERSION
# ==========================================

import pandas as pd
import numpy as np
from .indicators import TechnicalIndicators


class MomentumStrategy:
    """Fixed NAS100 Momentum Strategy with proper boolean indexing"""

    def __init__(self, params):
        self.params = params
        self.indicators = TechnicalIndicators()

    def generate_signals(self, df):
        """Generate trading signals with scoring-based flexibility"""
        if df is None or len(df) == 0:
            print("Error: Empty or None DataFrame provided")
            return pd.DataFrame()

        try:
            df = df.copy()

            # Add indicators
            df = self.indicators.add_all_indicators(
                df,
                lookback_high=self.params.get("lookback_high", 15),
                lookback_low=self.params.get("lookback_low", 15)
            )

            # Initialize signal columns
            df['Signal'] = 0
            df['Entry_Price'] = np.nan
            df['Stop_Loss'] = np.nan
            df['Take_Profit'] = np.nan
            df['Signal_Strength'] = 0
            df['Exit_Reason'] = ''

            # Fallbacks for missing indicators
            required_columns = ['ATR', 'ADX', 'EMA_Separation', 'Volume_Above_Avg']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0

            # Convert bool indicators
            bool_columns = ['Volume_Above_Avg', 'Volume_Surge', 'Bullish_Alignment', 'Bearish_Alignment',
                            'Bullish_Momentum', 'Bearish_Momentum', 'High_Break_Strong', 'Low_Break_Strong',
                            'High_Break_Weak', 'Low_Break_Weak', 'Prev_Close_Below_High', 'Prev_Close_Above_Low']
            for col in bool_columns:
                if col in df.columns:
                    df[col] = df[col].astype(bool)
                else:
                    df[col] = pd.Series([False] * len(df), index=df.index)

            # Base features
            volume_ok = df['Volume_Above_Avg']
            volume_surge = df['Volume_Surge']
            atr_filter = df['ATR'] > self.params.get('min_atr', 3.0)
            trend_strength_ok = df['EMA_Separation'] > self.params.get('trend_strength', 0.2)
            adx = df['ADX'].fillna(0)
            di_plus = df.get('DI_Plus', pd.Series(0, index=df.index))
            di_minus = df.get('DI_Minus', pd.Series(0, index=df.index))
            rsi = df.get('RSI', pd.Series(50, index=df.index))
            ema_200 = df.get('EMA_200', df['Close'])

            # Long features
            long_score = (
                    df['Bullish_Alignment'].astype(int) +
                    df['Bullish_Momentum'].astype(int) +
                    df['High_Break_Strong'].astype(int) +
                    df['Volume_Surge'].astype(int) +
                    ((adx > self.params.get('adx_threshold', 25)) | (di_plus > di_minus)).astype(int) +
                    atr_filter.astype(int) +
                    ((rsi < self.params.get('rsi_overbought', 70)) & (rsi > 40) if self.params.get('use_rsi_filter',
                                                                                                   True)
                     else pd.Series(True, index=df.index)).astype(int) +
                    (df['Close'] > ema_200).astype(int)
            )

            # Short features
            short_score = (
                    df['Bearish_Alignment'].astype(int) +
                    df['Bearish_Momentum'].astype(int) +
                    df['Low_Break_Strong'].astype(int) +
                    df['Volume_Surge'].astype(int) +
                    ((adx > self.params.get('adx_threshold', 25)) | (di_minus > di_plus)).astype(int) +
                    atr_filter.astype(int) +
                    ((rsi > self.params.get('rsi_oversold', 30)) & (rsi < 60) if self.params.get('use_rsi_filter', True)
                     else pd.Series(True, index=df.index)).astype(int) +
                    (df['Close'] < ema_200).astype(int)
            )

            # Signal assignment
            df.loc[long_score >= 6, ['Signal', 'Signal_Strength']] = [1, 2]
            df.loc[(long_score >= 4) & (long_score < 6), ['Signal', 'Signal_Strength']] = [1, 1]
            df.loc[short_score >= 6, ['Signal', 'Signal_Strength']] = [-1, 2]
            df.loc[(short_score >= 4) & (short_score < 6), ['Signal', 'Signal_Strength']] = [-1, 1]

            # Entry prices
            signal_mask = df['Signal'] != 0
            df.loc[signal_mask, 'Entry_Price'] = df.loc[signal_mask, 'Close']

            # Stop-loss & Take-profit
            base_atr_mult = self.params.get('atr_mult', 1.2)
            atr_values = df['ATR'].fillna(df['Close'] * 0.01)
            tp_base = self.params.get('take_profit_r', 1.6)

            for idx in df[signal_mask].index:
                entry = df.at[idx, 'Close']
                atr = atr_values.at[idx]
                strength = df.at[idx, 'Signal_Strength']
                direction = df.at[idx, 'Signal']

                atr_mult = base_atr_mult * (0.8 if strength == 2 else 1.2)
                tp_mult = tp_base * (1.2 if strength == 2 else 1.0)

                if direction == 1:
                    df.at[idx, 'Stop_Loss'] = entry - atr * atr_mult
                    df.at[idx, 'Take_Profit'] = entry + atr * atr_mult * tp_mult
                elif direction == -1:
                    df.at[idx, 'Stop_Loss'] = entry + atr * atr_mult
                    df.at[idx, 'Take_Profit'] = entry - atr * atr_mult * tp_mult

            return df

        except Exception as e:
            print(f"Error in generate_signals: {e}")
            import traceback
            traceback.print_exc()
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
        """Get strategy statistics with safe operations"""
        try:
            if df is None or len(df) == 0:
                return {
                    'total_signals': 0,
                    'long_signals': 0,
                    'short_signals': 0,
                    'avg_atr': 0,
                    'bullish_trend_pct': 0,
                    'bearish_trend_pct': 0,
                    'high_adx_pct': 0
                }

            long_signals = df[df['Signal'] == 1] if 'Signal' in df.columns else pd.DataFrame()
            short_signals = df[df['Signal'] == -1] if 'Signal' in df.columns else pd.DataFrame()

            # Safe calculation of statistics
            ema_50_col = 'EMA_50' if 'EMA_50' in df.columns else None
            ema_200_col = 'EMA_200' if 'EMA_200' in df.columns else None

            if ema_50_col and ema_200_col:
                bullish_trend_pct = (df[ema_50_col] > df[ema_200_col]).mean() * 100
                bearish_trend_pct = (df[ema_50_col] < df[ema_200_col]).mean() * 100
            else:
                bullish_trend_pct = 0
                bearish_trend_pct = 0

            stats = {
                'total_signals': len(long_signals) + len(short_signals),
                'long_signals': len(long_signals),
                'short_signals': len(short_signals),
                'avg_atr': df['ATR'].mean() if 'ATR' in df.columns else 0,
                'bullish_trend_pct': bullish_trend_pct,
                'bearish_trend_pct': bearish_trend_pct,
                'high_adx_pct': (df['ADX'] > self.params.get('adx_threshold',
                                                             20)).mean() * 100 if 'ADX' in df.columns else 0
            }

            return stats

        except Exception as e:
            print(f"Error calculating strategy stats: {e}")
            return {
                'total_signals': 0,
                'long_signals': 0,
                'short_signals': 0,
                'avg_atr': 0,
                'bullish_trend_pct': 0,
                'bearish_trend_pct': 0,
                'high_adx_pct': 0
            }