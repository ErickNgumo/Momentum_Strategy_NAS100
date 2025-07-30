#!/usr/bin/env python3
"""
NAS100 Momentum Strategy - Main Application
===========================================

This is the main entry point for the NAS100 momentum trading strategy.
Includes backtesting, optimization, and data management functionality.
"""

import os
import sys
import traceback
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to system path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Import project modules
    from config import Config
    from data.mt5_connector import MT5Connector
    from strategies.improved_momentum import ImprovedMomentumStrategy
    from backtest.backtest_engine import BacktestEngine
    from analysis.metrics import PerformanceAnalyzer

    print("✓ All core modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please ensure all required modules are in the correct directories:")
    print("  - config.py")
    print("  - data/mt5_connector.py")
    print("  - strategies/momentum.py")
    print("  - backtest/backtest_engine.py")
    print("  - analysis/metrics.py")
    sys.exit(1)

# Check for optimization module
try:
    from optimization.optimizer import EnhancedOptimizer

    OPTIMIZER_AVAILABLE = True
    print("✓ Optimization module available")
except ImportError:
    OPTIMIZER_AVAILABLE = False
    print("⚠️ Optimization module not available")


class NAS100MomentumBacktest:
    """Main backtesting system with error handling and validation"""

    def __init__(self):
        """Initialize the backtesting system"""
        try:
            self.config = Config()
            self.mt5_connector = MT5Connector(
                self.config.MT5_LOGIN,
                self.config.MT5_PASSWORD,
                self.config.MT5_SERVER
            )
            print("✓ Backtesting system initialized")
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            raise

    def create_sample_data(self, days=365):
        """Create sample data for testing when MT5 is not available"""
        print(f"Creating sample NAS100 data for {days} days...")

        # Generate timestamps
        dates = pd.date_range(
            start=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - pd.Timedelta(days=days),
            end=datetime.now(),
            freq='15min'
        )

        # Filter for market hours (approximate)
        dates = dates[(dates.hour >= 9) & (dates.hour <= 16)]

        n_bars = len(dates)

        # Generate realistic price data using geometric Brownian motion
        np.random.seed(42)  # For reproducible results

        # Starting price (approximate NAS100 level)
        initial_price = 15000

        # Parameters for price simulation
        drift = 0.0001  # Small upward drift per bar
        volatility = 0.005  # Volatility per bar

        # Generate price changes
        price_changes = np.random.normal(drift, volatility, n_bars)

        # Calculate cumulative prices
        log_prices = np.cumsum(price_changes)
        prices = initial_price * np.exp(log_prices)

        # Generate OHLC data
        close_prices = prices

        # Generate realistic OHLC relationships
        high_low_range = np.random.gamma(2, 0.002) * close_prices  # Range as % of close

        opens = np.roll(close_prices, 1)
        opens[0] = initial_price

        # Add some noise to opens
        opens += np.random.normal(0, close_prices * 0.001)

        highs = np.maximum(opens, close_prices) + np.random.exponential(0.3) * high_low_range
        lows = np.minimum(opens, close_prices) - np.random.exponential(0.3) * high_low_range

        # Generate volume data
        base_volume = 1000
        volume = np.random.lognormal(np.log(base_volume), 0.5, n_bars)

        # Create DataFrame
        df = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': volume.astype(int)
        }, index=dates)

        # Ensure data integrity
        df['High'] = np.maximum.reduce([df['Open'], df['High'], df['Close']])
        df['Low'] = np.minimum.reduce([df['Open'], df['Low'], df['Close']])

        print(f"✓ Generated {len(df)} bars of sample data")
        print(f"  Price range: ${df['Low'].min():.0f} - ${df['High'].max():.0f}")
        print(f"  Average volume: {df['Volume'].mean():.0f}")

        return df

    def download_data(self):
        """Download data from MT5 with fallback to sample data"""
        symbol = self.config.DATA_PARAMS['symbol']
        timeframe = self.config.DATA_PARAMS['timeframe']
        days_back = self.config.DATA_PARAMS['lookback_days']
        data_path = self.config.DATA_PARAMS['data_path']

        print(f"Attempting to download {symbol} {timeframe} data...")

        try:
            df = self.mt5_connector.save_data(symbol, timeframe, days_back, data_path)

            if df is not None and len(df) > 100:
                print(f"✓ Downloaded {len(df)} bars from MT5")
                return df
            else:
                print("⚠️ MT5 download failed or insufficient data")

        except Exception as e:
            print(f"⚠️ MT5 connection error: {e}")

        # Fallback to sample data
        print("Creating sample data for testing...")
        df = self.create_sample_data(days_back)

        # Save sample data
        try:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path)
            print(f"✓ Sample data saved to {data_path}")
        except Exception as e:
            print(f"⚠️ Could not save sample data: {e}")

        return df

    def load_data(self):
        """Load data from CSV or create sample data"""
        data_path = self.config.DATA_PARAMS['data_path']

        if os.path.exists(data_path):
            try:
                print(f"Loading data from {data_path}")
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)

                # Validate data
                if len(df) < 100:
                    print(f"⚠️ Insufficient data ({len(df)} bars), creating new sample data")
                    return self.create_sample_data()

                # Check for required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    print(f"⚠️ Missing columns: {missing_cols}, creating new sample data")
                    return self.create_sample_data()

                print(f"✓ Loaded {len(df)} bars from file")
                print(f"  Date range: {df.index[0]} to {df.index[-1]}")
                return df

            except Exception as e:
                print(f"⚠️ Error loading data: {e}")
                print("Creating sample data instead...")
                return self.create_sample_data()
        else:
            print("Data file not found, downloading...")
            return self.download_data()

    def validate_data(self, df):
        """Validate data quality"""
        if df is None or len(df) == 0:
            return False, "No data provided"

        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        # Check for sufficient data
        if len(df) < 200:
            return False, f"Insufficient data: {len(df)} bars (need at least 200)"

        # Check for data quality issues
        nan_counts = df[required_cols].isnull().sum()
        if nan_counts.sum() > 0:
            return False, f"Data contains NaN values: {nan_counts.to_dict()}"

        # Check OHLC logic
        bad_data = (df['High'] < df['Low']).sum()
        if bad_data > 0:
            return False, f"Invalid OHLC data: {bad_data} bars where High < Low"

        return True, "Data validation passed"

    def run_backtest(self, df=None, show_plots=True, save_results=True):
        """Execute full backtest with error handling"""
        print("\n" + "=" * 60)
        print("RUNNING BACKTEST")
        print("=" * 60)

        try:
            # Load data if not provided
            if df is None:
                df = self.load_data()

            if df is None:
                print("❌ No data available for backtesting")
                return None, None

            # Validate data
            is_valid, validation_msg = self.validate_data(df)
            if not is_valid:
                print(f"❌ Data validation failed: {validation_msg}")
                return None, None

            print(f"✓ {validation_msg}")
            print(f"✓ Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

            # Generate signals
            print("Generating trading signals...")
            strategy = ImprovedMomentumStrategy(self.config.STRATEGY_PARAMS)
            signals_df = strategy.generate_signals(df)

            # Check if signals were generated
            total_signals = (signals_df['Signal'] != 0).sum()
            if total_signals == 0:
                print("⚠️ No trading signals generated. Consider adjusting parameters.")
                return None, None

            # Get strategy statistics
            stats = strategy.get_strategy_stats(signals_df)
            print(f"✓ Generated {stats['total_signals']} signals "
                  f"({stats['long_signals']} long, {stats['short_signals']} short)")

            # Run backtest
            print("Running backtest engine...")
            backtest_params = {**self.config.STRATEGY_PARAMS, **self.config.BACKTEST_PARAMS}
            engine = BacktestEngine(signals_df, backtest_params)
            results = engine.run()

            if len(results['trades']) == 0:
                print("⚠️ No trades executed. Check strategy parameters.")
                return results, None

            # Analyze results
            analyzer = PerformanceAnalyzer(results)
            analyzer.print_summary()

            # Show plots
            if show_plots:
                try:
                    print("Generating performance plots...")
                    fig = analyzer.plot_results()
                    plt.show()
                except Exception as e:
                    print(f"⚠️ Error generating plots: {e}")

            # Save results
            if save_results:
                try:
                    self.save_backtest_results(results, analyzer)
                except Exception as e:
                    print(f"⚠️ Error saving results: {e}")

            return results, analyzer

        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            traceback.print_exc()
            return None, None

    def save_backtest_results(self, results, analyzer):
        """Save backtest results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "backtest_results"
        os.makedirs(results_dir, exist_ok=True)

        # Save trades
        if len(results['trades']) > 0:
            trades_file = os.path.join(results_dir, f"trades_{timestamp}.csv")
            results['trades'].to_csv(trades_file, index=False)
            print(f"✓ Trades saved to {trades_file}")

        # Save equity curve
        equity_file = os.path.join(results_dir, f"equity_{timestamp}.csv")
        results['equity_curve'].to_csv(equity_file, index=False)
        print(f"✓ Equity curve saved to {equity_file}")

        # Save metrics
        metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.txt")
        with open(metrics_file, 'w') as f:
            metrics = analyzer.calculate_metrics()
            f.write("BACKTEST PERFORMANCE METRICS\n")
            f.write("=" * 40 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key:<25}: {value}\n")
        print(f"✓ Metrics saved to {metrics_file}")


def enhanced_optimization_menu(backtester):
    """Enhanced optimization menu with error handling"""

    if not OPTIMIZER_AVAILABLE:
        print("❌ Optimization module not available")
        return

    print("\n" + "=" * 60)
    print("ENHANCED OPTIMIZATION MENU")
    print("=" * 60)
    print("1. Random Search (Fast, Good Results)")
    print("2. Bayesian Optimization (Smart, Efficient)")
    print("3. Parallel Grid Search (Exhaustive)")
    print("4. Quick Smart Search (Auto-reduced grid)")
    print("5. Back to main menu")

    choice = input("\nSelect optimization method (1-5): ").strip()

    if choice == '5':
        return

    # Load and validate data
    print("Loading data for optimization...")
    df = backtester.load_data()
    if df is None:
        print("❌ No data available for optimization")
        return

    is_valid, validation_msg = backtester.validate_data(df)
    if not is_valid:
        print(f"❌ Data validation failed: {validation_msg}")
        return

    print(f"✓ Data validated: {len(df)} bars")

    try:
        # Initialize optimizer
        optimizer = EnhancedOptimizer(backtester.config, df)

        # Define parameter ranges
        param_ranges = {
            'lookback_high': [8, 10, 12, 15, 20, 25],
            'lookback_low': [8, 10, 12, 15, 20, 25],
            'volume_multiplier': [1.1, 1.2, 1.3, 1.4, 1.5],
            'adx_threshold': [15, 20, 25, 30],
            'ema_fast': [9, 15, 21],
            'ema_slow': [30, 50, 100],
            'take_profit_r': [2.5, 3.0, 4.0, 5.0, 6.0, 7.0],
            'stop_loss_r': [0.5, 0.6, 0.8],
            'atr_mult': [1.0, 1.2, 1.5, 2.0],
            'use_rsi_filter': [True, False],
            'trailing_stop': [True, False]
        }

        best_params = None

        if choice == '1':
            # Random Search
            n_trials = int(input("Number of random trials (default 500): ") or "500")
            n_jobs = int(input("Number of parallel workers (default -1 for all cores): ") or "-1")

            best_params, results = optimizer.random_search(
                param_ranges, n_trials=n_trials, n_jobs=n_jobs
            )

            if results:
                optimizer.save_results(results, f'random_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        elif choice == '2':
            # Bayesian Optimization
            try:
                from optimization.optimizer import OPTUNA_AVAILABLE
                if not OPTUNA_AVAILABLE:
                    print("❌ Optuna not installed. Install with: pip install optuna")
                    return

                n_trials = int(input("Number of Bayesian trials (default 100): ") or "100")
                best_params, study = optimizer.bayesian_optimization(
                    param_ranges, n_trials=n_trials
                )
            except Exception as e:
                print(f"❌ Bayesian optimization failed: {e}")
                return

        elif choice == '3':
            # Parallel Grid Search
            total_combinations = np.prod([len(v) for v in param_ranges.values()])
            print(f"\n⚠️ Full grid has {total_combinations:,} combinations!")
            confirm = input("Continue with reduced grid? (y/n): ").lower()

            if confirm == 'y':
                reduced_ranges = optimizer.smart_grid_reduction(param_ranges, max_combinations=200)
                best_params, results = optimizer.grid_search_parallel(reduced_ranges)
                if results:
                    optimizer.save_results(results, f'grid_search_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            else:
                return

        elif choice == '4':
            # Quick Smart Search
            print("Running quick optimization with smart parameter selection...")

            quick_ranges = {
                'lookback_high': [10, 15, 20],
                'volume_multiplier': [1.2, 1.3, 1.4],
                'take_profit_r': [1.5, 2.0, 2.5],
                'stop_loss_r': [0.7, 0.9],
                'adx_threshold': [20, 25],
                'use_rsi_filter': [True, False]
            }

            best_params, results = optimizer.random_search(
                quick_ranges, n_trials=100, n_jobs=-1
            )

        # Test best parameters
        if best_params:
            print("\n" + "=" * 60)
            print("TESTING BEST PARAMETERS")
            print("=" * 60)

            # Backup original parameters
            original_params = backtester.config.STRATEGY_PARAMS.copy()

            try:
                # Update config with best parameters
                backtester.config.STRATEGY_PARAMS.update(best_params)

                # Run backtest
                results, analyzer = backtester.run_backtest(df, show_plots=True)

                if results and analyzer:
                    # Show parameter differences
                    print("\n" + "=" * 60)
                    print("OPTIMIZED PARAMETERS:")
                    print("=" * 60)
                    for key, value in best_params.items():
                        if key not in original_params or original_params[key] != value:
                            original_val = original_params.get(key, 'N/A')
                            print(f"{key:20}: {original_val} → {value}")

            finally:
                # Restore original parameters
                backtester.config.STRATEGY_PARAMS = original_params

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        traceback.print_exc()


def main():
    """Main application entry point"""
    print("=" * 60)
    print("NAS100 MOMENTUM STRATEGY BACKTESTING SYSTEM")
    print("=" * 60)
    print("Version: 1.0")
    print("Author: Trading System Developer")
    print("=" * 60)

    try:
        backtester = NAS100MomentumBacktest()

        while True:
            print("\n" + "=" * 40)
            print("MAIN MENU")
            print("=" * 40)
            print("1. Download fresh data from MT5")
            print("2. Run backtest with current parameters")
            print("3. Optimize parameters")
            print("4. View current configuration")
            print("5. Create sample data")
            print("6. Exit")

            choice = input("\nEnter choice (1-6): ").strip()

            if choice == '1':
                print("\nDownloading data...")
                try:
                    df = backtester.download_data()
                    if df is not None:
                        print("✓ Data download completed")
                    else:
                        print("⚠️ Data download failed, but sample data was created")
                except Exception as e:
                    print(f"❌ Download error: {e}")

            elif choice == '2':
                print("\nStarting backtest...")
                try:
                    results, analyzer = backtester.run_backtest()
                    if results is None:
                        print("❌ Backtest failed - check data and parameters")
                except Exception as e:
                    print(f"❌ Backtest error: {e}")
                    traceback.print_exc()

            elif choice == '3':
                if OPTIMIZER_AVAILABLE:
                    enhanced_optimization_menu(backtester)
                else:
                    print("❌ Optimization not available - install required packages")

            elif choice == '4':
                print("\nCurrent Configuration:")
                print("-" * 40)
                for key, value in backtester.config.STRATEGY_PARAMS.items():
                    print(f"{key:20}: {value}")

            elif choice == '5':
                print("\nCreating sample data...")
                try:
                    df = backtester.create_sample_data()
                    data_path = backtester.config.DATA_PARAMS['data_path']
                    os.makedirs(os.path.dirname(data_path), exist_ok=True)
                    df.to_csv(data_path)
                    print(f"✓ Sample data created and saved to {data_path}")
                except Exception as e:
                    print(f"❌ Error creating sample data: {e}")

            elif choice == '6':
                print("\nExiting system...")
                print("Thank you for using NAS100 Momentum Strategy!")
                break

            else:
                print("❌ Invalid choice. Please enter 1-6.")

    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
    finally:
        print("\nProgram terminated.")


if __name__ == "__main__":
    main()