import pandas as pd
import numpy as np
import ta


class TechnicalIndicators:
    """Technical indicator calculations"""

    @staticmethod
    def add_all_indicators(df, lookback_high=15, lookback_low=15):
        df = df.copy()

        # Trend Indicators
        df['EMA_9'] = ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator()
        df['EMA_21'] = ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], window=50).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], window=200).ema_indicator()

        df['Bullish_Alignment'] = (df['EMA_9'] > df['EMA_21']) & (df['EMA_21'] > df['EMA_50'])
        df['Bearish_Alignment'] = (df['EMA_9'] < df['EMA_21']) & (df['EMA_21'] < df['EMA_50'])

        # Momentum Indicators
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['Bullish_Momentum'] = (df['MACD'] > df['MACD_Signal'])
        df['Bearish_Momentum'] = (df['MACD'] < df['MACD_Signal'])

        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

        # Volatility
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=10).average_true_range()

        # Trend Strength
        adx_indicator = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx_indicator.adx()
        df['DI_Plus'] = adx_indicator.adx_pos()
        df['DI_Minus'] = adx_indicator.adx_neg()

        # Volume Indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Surge'] = df['Volume_Ratio'] > 1.3

        # Breakout Setup
        df['Rolling_High'] = df['High'].rolling(window=lookback_high).max()
        df['Rolling_Low'] = df['Low'].rolling(window=lookback_low).min()

        df['Breakout_Strength_High'] = (df['Close'] - df['Rolling_High'].shift(1)) / df['ATR']
        df['Breakout_Strength_Low'] = (df['Rolling_Low'].shift(1) - df['Close']) / df['ATR']

        df['High_Break_Strong'] = df['Breakout_Strength_High'] > 1.5
        df['Low_Break_Strong'] = df['Breakout_Strength_Low'] > 1.5

        return df
