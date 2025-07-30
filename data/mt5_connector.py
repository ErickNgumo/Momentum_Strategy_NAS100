import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os


class MT5Connector:
    """MetaTrader5 data connector"""

    def __init__(self, login=None, password=None, server=None):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False

    def connect(self):
        """Connect to MT5"""
        if not mt5.initialize():
            print(f"initialize() failed, error code = {mt5.last_error()}")
            return False

        if self.login and self.password and self.server:
            if not mt5.login(self.login, password=self.password, server=self.server):
                print(f"Failed to connect to {self.server}, error code = {mt5.last_error()}")
                return False

        self.connected = True
        print("Connected to MT5 successfully")
        return True

    def disconnect(self):
        """Disconnect from MT5"""
        mt5.shutdown()
        self.connected = False
        print("Disconnected from MT5")

    def get_symbol_data(self, symbol, timeframe, start_date, end_date):
        """Download historical data from MT5"""
        if not self.connected:
            if not self.connect():
                return None

        # Convert timeframe string to MT5 constant
        tf_map = {
            "1M": mt5.TIMEFRAME_M1,
            "5M": mt5.TIMEFRAME_M5,
            "15M": mt5.TIMEFRAME_M15,
            "30M": mt5.TIMEFRAME_M30,
            "1H": mt5.TIMEFRAME_H1,
            "4H": mt5.TIMEFRAME_H4,
            "1D": mt5.TIMEFRAME_D1
        }

        mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M15)

        # Get data
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)

        if rates is None:
            print(f"No data retrieved, error code = {mt5.last_error()}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)

        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.set_index('timestamp', inplace=True)

        return df

    def save_data(self, symbol, timeframe, days_back=365, filename=None):
        """Download and save data to CSV"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        df = self.get_symbol_data(symbol, timeframe, start_date, end_date)

        if df is not None:
            if filename is None:
                filename = f"data/{symbol.lower()}_{timeframe.lower()}.csv"

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df.to_csv(filename)
            print(f"Data saved to {filename}")
            return df

        return None
