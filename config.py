class Config:
    """Configuration for NAS100 Momentum Strategy"""

    # MT5 Connection
    MT5_LOGIN = None  # Your MT5 login
    MT5_PASSWORD = None  # Your MT5 password
    MT5_SERVER = None  # Your broker server

    # Strategy Parameters
    STRATEGY_PARAMS = {
        # Breakout parameters
        "lookback_high": 20,
        "lookback_low": 20,

        # Risk management (improved risk-reward)
        "stop_loss_r": 1.0,
        "take_profit_r": 2.5,
        "atr_mult": 2.0,

        # Trend filters
        "ema_fast": 21,
        "ema_slow": 50,
        "use_trend_filter": True,

        # Momentum confirmation
        "adx_threshold": 25,
        "min_atr": 2.0,

        # Volume confirmation
        "volume_multiplier": 1.3,
        "min_volume_ratio": 1.0,

        # Risk per trade and trade management
        "risk_per_trade": 0.02,
        "max_positions": 1,
        "trailing_stop": True,

        # Filters
        "use_rsi_filter": False,

        # Exit management
        "max_trade_duration": 24  # For example: 6 hours on 15M chart
    }
    # Data Parameters
    DATA_PARAMS = {
        "symbol": "USTECz",
        "timeframe": "15M",
        "lookback_days": 365,
        "data_path": "data/nas100_15m.csv"
    }

    # Backtest Parameters
    BACKTEST_PARAMS = {
        "initial_capital": 10000,
        "commission": 0.0003,  # 0.03% per trade
        "slippage": 0.0001,  # 0.01% slippage
        "max_positions": 1  # Only one position at a time
    }