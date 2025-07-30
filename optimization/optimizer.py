import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import multiprocessing as mp
import random
import json
import os
from functools import partial
import warnings

from strategies.improved_momentum import ImprovedMomentumStrategy

warnings.filterwarnings('ignore')

# Optional imports for advanced optimization
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Scikit-optimize not available. Install with: pip install scikit-optimize")


class EnhancedOptimizer:
    """Fixed parameter optimizer with proper return calculations and Bayesian optimization"""

    def __init__(self, config, data_df):
        self.config = config
        self.data_df = data_df
        self.results_cache = {}
        self.optimization_history = []

    def _evaluate_parameters(self, params: Dict, data_df: pd.DataFrame = None) -> Dict:
        """Evaluate a single parameter combination with FIXED return calculation"""
        if data_df is None:
            data_df = self.data_df

        try:
            # Import here to avoid circular imports
            from strategies.improved_momentum import ImprovedMomentumStrategy
            from backtest.backtest_engine import BacktestEngine
            from analysis.metrics import PerformanceAnalyzer

            # Generate signals
            strategy = ImprovedMomentumStrategy (params)
            signals_df = strategy.generate_signals(data_df)

            # Quick check for minimum signals
            total_signals = (signals_df['Signal'] != 0).sum()
            if total_signals < 10:
                return {
                    'score': -999,
                    'metrics': {'Total Trades': 0, 'Total Return (%)': -100},
                    'params': params
                }

            # Run backtest with FIXED engine
            backtest_params = {**params, **self.config.BACKTEST_PARAMS}
            engine = BacktestEngine(signals_df, backtest_params)
            results = engine.run()

            # Check for minimum trades
            if len(results['trades']) < 5:
                return {
                    'score': -999,
                    'metrics': {'Total Trades': len(results['trades']), 'Total Return (%)': -100},
                    'params': params
                }

            # Analyze results with FIXED analyzer
            analyzer = PerformanceAnalyzer(results)
            metrics = analyzer.calculate_metrics()

            # Validate that we have proper return calculation
            if 'Total Return (%)' not in metrics or metrics['Total Return (%)'] == 0:
                # Manually calculate return if needed
                initial_capital = results.get('initial_capital', 10000)
                final_capital = results.get('final_capital', initial_capital)
                manual_return = (final_capital - initial_capital) / initial_capital * 100
                metrics['Total Return (%)'] = manual_return

            # Calculate composite score with FIXED logic
            score = self._calculate_score(metrics)

            return {
                'score': score,
                'metrics': metrics,
                'params': params,
                'trades': len(results['trades']),
                'return_pct': metrics.get('Total Return (%)', 0)
            }

        except Exception as e:
            print(f"Error evaluating parameters: {e}")
            return {
                'score': -999,
                'metrics': {'error': str(e), 'Total Return (%)': -100},
                'params': params
            }

    def _calculate_score(self, metrics: Dict) -> float:
        """Calculate composite optimization score with PROPER return weighting"""

        # Extract key metrics with defaults
        win_rate = metrics.get('Win Rate (%)', 0)
        total_return = metrics.get('Total Return (%)', 0)
        sharpe = metrics.get('Sharpe Ratio', 0)
        profit_factor = metrics.get('Profit Factor', 0)
        max_dd = abs(metrics.get('Max Drawdown (%)', -100))
        total_trades = metrics.get('Total Trades', 0)
        expectancy = metrics.get('Expectancy ($)', 0)
        calmar_ratio = metrics.get('Calmar Ratio', 0)

        # FIXED Multi-objective scoring with proper return emphasis
        score = 0

        # 1. Total Return (40% weight) - MOST IMPORTANT
        if total_return > 0:
            # Reward positive returns, with diminishing returns for very high values
            return_score = np.log(1 + total_return) * 4.0  # Log to prevent extreme values
        else:
            # Heavily penalize negative returns
            return_score = total_return * 0.1
        score += return_score * 0.4

        # 2. Risk-adjusted return (25% weight)
        if sharpe > 0:
            sharpe_score = min(sharpe * 2, 6)  # Cap at reasonable value
        else:
            sharpe_score = sharpe * 0.5
        score += sharpe_score * 0.25

        # 3. Profit Factor (20% weight)
        if profit_factor > 1:
            pf_score = min(np.log(profit_factor) * 3, 5)  # Log scaling
        else:
            pf_score = (profit_factor - 1) * 5  # Linear penalty below 1
        score += pf_score * 0.2

        # 4. Win Rate bonus (10% weight)
        # Bonus for win rates above 40%, penalty below 30%
        if win_rate > 40:
            wr_score = (win_rate - 40) * 0.1
        elif win_rate < 30:
            wr_score = (win_rate - 30) * 0.2  # Heavier penalty
        else:
            wr_score = 0
        score += wr_score * 0.1

        # 5. Drawdown penalty (5% weight)
        if max_dd > 0:
            dd_penalty = min(max_dd * 0.1, 10)  # Cap penalty
        else:
            dd_penalty = 0
        score -= dd_penalty * 0.05

        # Bonus factors
        bonuses = 0

        # Trade count bonus (need enough trades for statistical significance)
        if total_trades >= 50:
            bonuses += 0.5
        elif total_trades >= 30:
            bonuses += 0.2

        # Expectancy bonus
        if expectancy > 0:
            bonuses += min(expectancy / 10, 1.0)  # Cap at 1.0

        # Calmar ratio bonus
        if calmar_ratio > 1:
            bonuses += min(calmar_ratio * 0.3, 1.0)

        score += bonuses

        return score

    def bayesian_optimization(self, param_ranges: Dict, n_trials: int = 100) -> Tuple[Dict, Any]:
        """Bayesian optimization using Optuna"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")

        print(f"\n{'=' * 60}")
        print(f"BAYESIAN OPTIMIZATION (OPTUNA)")
        print(f"{'=' * 60}")
        print(f"Running {n_trials} smart trials")
        print(f"Data shape: {self.data_df.shape}")

        def objective(trial):
            """Optuna objective function"""
            params = {}

            for param_name, param_range in param_ranges.items():
                if isinstance(param_range[0], bool):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
                elif isinstance(param_range[0], int):
                    params[param_name] = trial.suggest_int(param_name, min(param_range), max(param_range))
                elif isinstance(param_range[0], float):
                    params[param_name] = trial.suggest_float(param_name, min(param_range), max(param_range))
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_range)

            # Add default params
            full_params = self.config.STRATEGY_PARAMS.copy()
            full_params.update(params)

            # Evaluate parameters
            result = self._evaluate_parameters(full_params)

            # Store additional info for analysis
            trial.set_user_attr('return_pct', result.get('return_pct', 0))
            trial.set_user_attr('trades', result.get('trades', 0))
            trial.set_user_attr('metrics', result.get('metrics', {}))

            return result['score']

        # Create study
        study = optuna.create_study(direction='maximize')

        # Optimize
        start_time = datetime.now()
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Get results
        best_params = study.best_params
        best_score = study.best_value

        # Add default params to best params
        full_best_params = self.config.STRATEGY_PARAMS.copy()
        full_best_params.update(best_params)

        # Print results
        total_time = (datetime.now() - start_time).total_seconds() / 60
        print(f"\nOptimization completed in {total_time:.1f} minutes")
        print(f"Best score: {best_score:.3f}")

        # Show top trials
        print(f"\n{'=' * 60}")
        print("TOP 10 TRIALS:")
        print(f"{'=' * 60}")

        top_trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)[:10]

        for i, trial in enumerate(top_trials):
            if trial.value and trial.value > -999:
                return_pct = trial.user_attrs.get('return_pct', 0)
                trades = trial.user_attrs.get('trades', 0)
                metrics = trial.user_attrs.get('metrics', {})

                win_rate = metrics.get('Win Rate (%)', 0)
                pf = metrics.get('Profit Factor', 0)
                max_dd = metrics.get('Max Drawdown (%)', 0)

                print(f"\nTrial {i + 1} | Score: {trial.value:.3f}")
                print(f"  Trades: {trades:3d} | WinRate: {win_rate:5.1f}% | Return: {return_pct:+7.1f}%")
                print(f"  PF: {pf:5.2f} | MaxDD: {max_dd:5.1f}%")

                # Show key parameters
                key_params = ['take_profit_r', 'stop_loss_r', 'lookback_high', 'volume_multiplier']
                param_str = " | ".join([f"{k}={trial.params.get(k, 'N/A')}" for k in key_params if k in trial.params])
                print(f"  Params: {param_str}")

        return full_best_params, study

    def grid_search_parallel(self, param_ranges: Dict, n_jobs: int = -1) -> Tuple[Dict, List]:
        """Parallel grid search optimization"""
        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        print(f"\n{'=' * 60}")
        print(f"PARALLEL GRID SEARCH OPTIMIZATION")
        print(f"{'=' * 60}")

        # Generate all parameter combinations
        param_combinations = []
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        from itertools import product
        for combination in product(*param_values):
            combo = dict(zip(param_names, combination))
            full_params = self.config.STRATEGY_PARAMS.copy()
            full_params.update(combo)
            param_combinations.append(full_params)

        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations")
        print(f"Using {n_jobs} parallel workers")

        # Parallel evaluation
        results = []
        best_score = -999
        best_params = None

        start_time = datetime.now()

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self._evaluate_parameters, params): params
                for params in param_combinations
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_params):
                completed += 1
                result = future.result()
                results.append(result)

                if result['score'] > best_score:
                    best_score = result['score']
                    best_params = result['params']

                if completed % max(1, total_combinations // 20) == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed / elapsed
                    eta = (total_combinations - completed) / rate / 60

                    print(f"Progress: {completed}/{total_combinations} ({completed / total_combinations * 100:.1f}%) "
                          f"| Best Score: {best_score:.3f} "
                          f"| Rate: {rate:.1f} tests/sec | ETA: {eta:.1f} min")

        # Sort results
        results.sort(key=lambda x: x['score'], reverse=True)

        total_time = (datetime.now() - start_time).total_seconds() / 60
        print(f"\nGrid search completed in {total_time:.1f} minutes")

        return best_params, results

    def smart_grid_reduction(self, param_ranges: Dict, max_combinations: int = 500) -> Dict:
        """Intelligently reduce grid size while keeping important parameter combinations"""
        total_combinations = np.prod([len(v) for v in param_ranges.values()])

        if total_combinations <= max_combinations:
            return param_ranges

        print(f"Reducing grid from {total_combinations:,} to ~{max_combinations} combinations")

        reduced_ranges = {}

        # Priority parameters (keep more values)
        priority_params = ['take_profit_r', 'stop_loss_r', 'lookback_high', 'volume_multiplier']

        # Secondary parameters (reduce more aggressively)
        secondary_params = ['adx_threshold', 'ema_fast', 'ema_slow', 'atr_mult']

        for param_name, param_range in param_ranges.items():
            if isinstance(param_range[0], bool):
                # Keep boolean parameters as is
                reduced_ranges[param_name] = param_range
            elif param_name in priority_params:
                # Keep more values for priority parameters
                if len(param_range) > 4:
                    # Take every other value, but ensure we get start, middle, and end
                    indices = [0, len(param_range) // 2, -1]
                    if len(param_range) > 6:
                        indices.insert(1, len(param_range) // 4)
                        indices.insert(-1, 3 * len(param_range) // 4)
                    reduced_ranges[param_name] = [param_range[i] for i in indices]
                else:
                    reduced_ranges[param_name] = param_range
            elif param_name in secondary_params:
                # Reduce secondary parameters more aggressively
                if len(param_range) > 3:
                    # Keep start, middle, end
                    indices = [0, len(param_range) // 2, -1]
                    reduced_ranges[param_name] = [param_range[i] for i in indices]
                else:
                    reduced_ranges[param_name] = param_range
            else:
                # For other parameters, keep reasonable subset
                if len(param_range) > 3:
                    step = max(1, len(param_range) // 3)
                    reduced_ranges[param_name] = param_range[::step]
                else:
                    reduced_ranges[param_name] = param_range

        new_total = np.prod([len(v) for v in reduced_ranges.values()])
        print(f"Reduced grid size: {new_total:,} combinations")

        return reduced_ranges

    def random_search(self, param_ranges: Dict, n_trials: int = 1000,
                      n_jobs: int = -1, verbose: bool = True) -> Tuple[Dict, List]:
        """Random search optimization with FIXED evaluation"""
        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        print(f"\n{'=' * 60}")
        print(f"RANDOM SEARCH OPTIMIZATION")
        print(f"{'=' * 60}")
        print(f"Testing {n_trials} random combinations")
        print(f"Using {n_jobs} parallel workers")
        print(f"Data shape: {self.data_df.shape}")

        # Generate random parameter combinations
        param_combinations = []
        for trial in range(n_trials):
            combo = {}
            for param_name, param_range in param_ranges.items():
                if isinstance(param_range[0], bool):
                    combo[param_name] = random.choice(param_range)
                elif isinstance(param_range[0], int):
                    combo[param_name] = random.randint(min(param_range), max(param_range))
                elif isinstance(param_range[0], float):
                    combo[param_name] = random.uniform(min(param_range), max(param_range))
                else:
                    combo[param_name] = random.choice(param_range)

            # Add default params
            full_params = self.config.STRATEGY_PARAMS.copy()
            full_params.update(combo)
            param_combinations.append(full_params)

        # Parallel evaluation
        results = []
        best_score = -999
        best_params = None
        best_metrics = None

        start_time = datetime.now()

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self._evaluate_parameters, params): params
                for params in param_combinations
            }

            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_params):
                completed += 1
                result = future.result()
                results.append(result)

                if result['score'] > best_score:
                    best_score = result['score']
                    best_params = result['params']
                    best_metrics = result['metrics']

                if verbose and completed % 50 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    rate = completed / elapsed
                    eta = (n_trials - completed) / rate / 60

                    print(f"Progress: {completed}/{n_trials} ({completed / n_trials * 100:.1f}%) "
                          f"| Best Score: {best_score:.3f} "
                          f"| Best Return: {result.get('return_pct', 0):.1f}% "
                          f"| Rate: {rate:.1f} tests/sec "
                          f"| ETA: {eta:.1f} min")

        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)

        # Print top results
        print(f"\n{'=' * 60}")
        print("TOP 10 RESULTS:")
        print(f"{'=' * 60}")

        for i, result in enumerate(results[:10]):
            if 'error' not in result['metrics']:
                metrics = result['metrics']
                trades = result.get('trades', 0)
                return_pct = metrics.get('Total Return (%)', 0)
                win_rate = metrics.get('Win Rate (%)', 0)
                sharpe = metrics.get('Sharpe Ratio', 0)
                pf = metrics.get('Profit Factor', 0)
                max_dd = metrics.get('Max Drawdown (%)', 0)

                print(f"\nRank {i + 1} | Score: {result['score']:.3f}")
                print(f"  Trades: {trades:3d} | WinRate: {win_rate:5.1f}% | Return: {return_pct:+7.1f}%")
                print(f"  Sharpe: {sharpe:5.2f} | PF: {pf:5.2f} | MaxDD: {max_dd:5.1f}%")

                # Show key optimized parameters
                opt_params = result['params']
                print(f"  Key Params: TP={opt_params.get('take_profit_r', 0):.1f}, "
                      f"SL={opt_params.get('stop_loss_r', 0):.1f}, "
                      f"LB={opt_params.get('lookback_high', 0)}")

        total_time = (datetime.now() - start_time).total_seconds() / 60
        print(f"\nOptimization completed in {total_time:.1f} minutes")
        print(f"Average time per test: {total_time * 60 / n_trials:.2f} seconds")

        if best_params and best_metrics:
            print(f"\n🏆 BEST RESULT:")
            print(f"Score: {best_score:.3f}")
            print(f"Return: {best_metrics.get('Total Return (%)', 0):+.1f}%")
            print(f"Win Rate: {best_metrics.get('Win Rate (%)', 0):.1f}%")
            print(f"Profit Factor: {best_metrics.get('Profit Factor', 0):.2f}")

        return best_params, results

    def save_results(self, results: List, filename: str = 'optimization_results.json'):
        """Save optimization results to file with proper formatting"""
        output_dir = 'optimization_results'
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)

        # Convert results to serializable format
        serializable_results = []
        for r in results[:100]:  # Save top 100
            if 'error' not in r['metrics']:
                # Ensure all metrics are properly formatted
                clean_metrics = {}
                for k, v in r['metrics'].items():
                    if isinstance(v, (np.number, np.ndarray)):
                        clean_metrics[k] = float(v)
                    else:
                        clean_metrics[k] = v

                serializable_results.append({
                    'score': float(r['score']),
                    'params': r['params'],
                    'metrics': clean_metrics,
                    'trades': r.get('trades', 0),
                    'return_pct': clean_metrics.get('Total Return (%)', 0)
                })

        # Save with metadata
        output_data = {
            'optimization_date': datetime.now().isoformat(),
            'total_results': len(results),
            'data_points': len(self.data_df),
            'results': serializable_results
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Results saved to {filepath}")
        print(f"Saved {len(serializable_results)} valid results")