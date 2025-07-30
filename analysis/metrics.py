# analysis/metrics.py - Performance Analysis
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PerformanceAnalyzer:
    """Comprehensive performance analysis for backtest results"""

    def __init__(self, backtest_results):
        """Initialize with backtest results"""
        self.results = backtest_results
        self.trades = backtest_results.get('trades', pd.DataFrame())
        self.equity_curve = backtest_results.get('equity_curve', pd.DataFrame())
        self.initial_capital = backtest_results.get('initial_capital', 10000)
        self.final_capital = backtest_results.get('final_capital', 10000)
        self.total_pnl = backtest_results.get('total_pnl', 0)

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if len(self.trades) == 0:
            return {
                'Total Trades': 0,
                'Total Return (%)': 0,
                'Win Rate (%)': 0,
                'Profit Factor': 0,
                'Max Drawdown (%)': 0,
                'Sharpe Ratio': 0,
                'Calmar Ratio': 0,
                'Expectancy ($)': 0
            }

        trades = self.trades.copy()

        # Basic trade statistics
        total_trades = len(trades)
        winning_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]

        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

        # P&L calculations
        total_pnl = trades['pnl'].sum()
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0

        # Profit factor
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Average trade metrics
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        avg_trade = trades['pnl'].mean()

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)

        # Return calculations
        total_return_pct = (self.final_capital - self.initial_capital) / self.initial_capital * 100

        # Risk metrics
        if len(self.equity_curve) > 0:
            equity_series = self.equity_curve['equity']

            # Calculate returns for Sharpe ratio
            equity_returns = equity_series.pct_change().dropna()

            # Sharpe ratio (assuming daily data, 252 trading days)
            if len(equity_returns) > 1 and equity_returns.std() > 0:
                mean_return = equity_returns.mean()
                std_return = equity_returns.std()
                # Annualize (assuming 15-minute bars)
                periods_per_year = 252 * 24 * 4  # 15-min bars per year
                sharpe_ratio = (mean_return * np.sqrt(periods_per_year)) / std_return
            else:
                sharpe_ratio = 0

            # Maximum drawdown
            peak = equity_series.expanding(min_periods=1).max()
            drawdown = (equity_series - peak) / peak * 100
            max_drawdown = drawdown.min()

            # Calmar ratio
            calmar_ratio = (total_return_pct / abs(max_drawdown)) if max_drawdown != 0 else 0

        else:
            sharpe_ratio = 0
            max_drawdown = 0
            calmar_ratio = 0

        # Trade duration analysis
        if 'duration_bars' in trades.columns:
            avg_duration = trades['duration_bars'].mean()
            max_duration = trades['duration_bars'].max()
            min_duration = trades['duration_bars'].min()
        else:
            avg_duration = max_duration = min_duration = 0

        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_runs(trades['pnl'] > 0)
        consecutive_losses = self._calculate_consecutive_runs(trades['pnl'] < 0)

        # Trade distribution
        long_trades = trades[trades['direction'] == 'Long'] if 'direction' in trades.columns else pd.DataFrame()
        short_trades = trades[trades['direction'] == 'Short'] if 'direction' in trades.columns else pd.DataFrame()

        long_win_rate = (len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100) if len(
            long_trades) > 0 else 0
        short_win_rate = (len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100) if len(
            short_trades) > 0 else 0

        # Monthly/weekly performance
        monthly_returns = self._calculate_periodic_returns('M') if len(self.equity_curve) > 0 else []

        return {
            # Basic metrics
            'Total Trades': total_trades,
            'Winning Trades': win_count,
            'Losing Trades': loss_count,
            'Win Rate (%)': round(win_rate, 2),

            # Return metrics
            'Total Return (%)': round(total_return_pct, 2),
            'Total P&L ($)': round(total_pnl, 2),
            'Gross Profit ($)': round(gross_profit, 2),
            'Gross Loss ($)': round(gross_loss, 2),

            # Performance ratios
            'Profit Factor': round(profit_factor, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Calmar Ratio': round(calmar_ratio, 2),

            # Risk metrics
            'Max Drawdown (%)': round(max_drawdown, 2),

            # Trade metrics
            'Average Trade ($)': round(avg_trade, 2),
            'Average Win ($)': round(avg_win, 2),
            'Average Loss ($)': round(avg_loss, 2),
            'Expectancy ($)': round(expectancy, 2),

            # Duration metrics
            'Avg Duration (bars)': round(avg_duration, 1),
            'Max Duration (bars)': max_duration,
            'Min Duration (bars)': min_duration,

            # Consecutive metrics
            'Max Consecutive Wins': consecutive_wins,
            'Max Consecutive Losses': consecutive_losses,

            # Direction analysis
            'Long Trades': len(long_trades),
            'Short Trades': len(short_trades),
            'Long Win Rate (%)': round(long_win_rate, 2),
            'Short Win Rate (%)': round(short_win_rate, 2),

            # Risk metrics
            'Best Trade ($)': round(trades['pnl'].max(), 2) if len(trades) > 0 else 0,
            'Worst Trade ($)': round(trades['pnl'].min(), 2) if len(trades) > 0 else 0,

            # Capital metrics
            'Initial Capital ($)': self.initial_capital,
            'Final Capital ($)': round(self.final_capital, 2),
        }

    def _calculate_consecutive_runs(self, boolean_series) -> int:
        """Calculate maximum consecutive runs of True values"""
        if len(boolean_series) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for value in boolean_series:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_periodic_returns(self, period='M') -> List[float]:
        """Calculate returns for specified period (M=monthly, W=weekly)"""
        if len(self.equity_curve) == 0:
            return []

        try:
            equity = self.equity_curve.set_index('timestamp')['equity']
            period_equity = equity.resample(period).last()
            period_returns = period_equity.pct_change().dropna() * 100
            return period_returns.tolist()
        except:
            return []

    def print_summary(self):
        """Print formatted performance summary"""
        metrics = self.calculate_metrics()

        print("\n" + "=" * 60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 60)

        # Basic performance
        print(f"\n📊 BASIC PERFORMANCE:")
        print(f"  Initial Capital:     ${metrics['Initial Capital ($)']:>10,.2f}")
        print(f"  Final Capital:       ${metrics['Final Capital ($)']:>10,.2f}")
        print(f"  Total Return:        {metrics['Total Return (%)']:>10.2f}%")
        print(f"  Total P&L:           ${metrics['Total P&L ($)']:>10,.2f}")

        # Trade statistics
        print(f"\n📈 TRADE STATISTICS:")
        print(f"  Total Trades:        {metrics['Total Trades']:>10}")
        print(f"  Winning Trades:      {metrics['Winning Trades']:>10}")
        print(f"  Losing Trades:       {metrics['Losing Trades']:>10}")
        print(f"  Win Rate:            {metrics['Win Rate (%)']:>10.2f}%")

        # Performance ratios
        print(f"\n⚖️ PERFORMANCE RATIOS:")
        print(f"  Profit Factor:       {metrics['Profit Factor']:>10.2f}")
        print(f"  Sharpe Ratio:        {metrics['Sharpe Ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {metrics['Calmar Ratio']:>10.2f}")

        # Risk metrics
        print(f"\n⚠️ RISK METRICS:")
        print(f"  Max Drawdown:        {metrics['Max Drawdown (%)']:>10.2f}%")
        print(f"  Best Trade:          ${metrics['Best Trade ($)']:>10,.2f}")
        print(f"  Worst Trade:         ${metrics['Worst Trade ($)']:>10,.2f}")

        # Trade analysis
        print(f"\n🔍 TRADE ANALYSIS:")
        print(f"  Average Trade:       ${metrics['Average Trade ($)']:>10,.2f}")
        print(f"  Average Win:         ${metrics['Average Win ($)']:>10,.2f}")
        print(f"  Average Loss:        ${metrics['Average Loss ($)']:>10,.2f}")
        print(f"  Expectancy:          ${metrics['Expectancy ($)']:>10,.2f}")

        # Direction analysis
        if metrics['Long Trades'] > 0 or metrics['Short Trades'] > 0:
            print(f"\n📋 DIRECTION ANALYSIS:")
            print(
                f"  Long Trades:         {metrics['Long Trades']:>10} ({metrics['Long Win Rate (%)']:>5.1f}% win rate)")
            print(
                f"  Short Trades:        {metrics['Short Trades']:>10} ({metrics['Short Win Rate (%)']:>5.1f}% win rate)")

        # Duration analysis
        if metrics['Avg Duration (bars)'] > 0:
            print(f"\n⏱️ DURATION ANALYSIS:")
            print(f"  Avg Duration:        {metrics['Avg Duration (bars)']:>10.1f} bars")
            print(f"  Max Duration:        {metrics['Max Duration (bars)']:>10} bars")
            print(f"  Min Duration:        {metrics['Min Duration (bars)']:>10} bars")

        # Consecutive analysis
        print(f"\n🔄 CONSECUTIVE ANALYSIS:")
        print(f"  Max Consecutive Wins: {metrics['Max Consecutive Wins']:>9}")
        print(f"  Max Consecutive Loss: {metrics['Max Consecutive Losses']:>9}")

        print("=" * 60)

    def plot_results(self, figsize=(15, 12)):
        """Create comprehensive performance plots"""
        if len(self.trades) == 0:
            print("No trades to plot")
            return None

        fig = plt.figure(figsize=figsize)

        # 1. Equity Curve
        ax1 = plt.subplot(2, 3, 1)
        if len(self.equity_curve) > 0:
            equity_data = self.equity_curve.copy()
            if 'timestamp' in equity_data.columns:
                equity_data['timestamp'] = pd.to_datetime(equity_data['timestamp'])
                ax1.plot(equity_data['timestamp'], equity_data['equity'], 'b-', linewidth=2, label='Equity')
                ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Equity ($)')
                ax1.set_title('Equity Curve')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Format x-axis
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. Drawdown
        ax2 = plt.subplot(2, 3, 2)
        if len(self.equity_curve) > 0:
            equity_series = self.equity_curve['equity']
            peak = equity_series.expanding(min_periods=1).max()
            drawdown = (equity_series - peak) / peak * 100

            if 'timestamp' in self.equity_curve.columns:
                timestamps = pd.to_datetime(self.equity_curve['timestamp'])
                ax2.fill_between(timestamps, drawdown, 0, color='red', alpha=0.7, label='Drawdown')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Drawdown (%)')
                ax2.set_title('Drawdown')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Format x-axis
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 3. Trade P&L Distribution
        ax3 = plt.subplot(2, 3, 3)
        pnl_data = self.trades['pnl']
        winning_trades = pnl_data[pnl_data > 0]
        losing_trades = pnl_data[pnl_data < 0]

        bins = 30
        ax3.hist(losing_trades, bins=bins, alpha=0.7, color='red', label=f'Losses ({len(losing_trades)})')
        ax3.hist(winning_trades, bins=bins, alpha=0.7, color='green', label=f'Wins ({len(winning_trades)})')
        ax3.set_xlabel('P&L ($)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Trade P&L Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.8)

        # 4. Monthly Returns
        ax4 = plt.subplot(2, 3, 4)
        monthly_returns = self._calculate_periodic_returns('M')
        if len(monthly_returns) > 0:
            months = range(1, len(monthly_returns) + 1)
            colors = ['green' if r > 0 else 'red' for r in monthly_returns]
            bars = ax4.bar(months, monthly_returns, color=colors, alpha=0.7)
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Return (%)')
            ax4.set_title('Monthly Returns')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)

            # Add value labels on bars
            for bar, value in zip(bars, monthly_returns):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height + (0.5 if height > 0 else -1.5),
                         f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

        # 5. Trade Duration Analysis
        ax5 = plt.subplot(2, 3, 5)
        if 'duration_bars' in self.trades.columns:
            duration_data = self.trades['duration_bars']
            ax5.hist(duration_data, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax5.set_xlabel('Duration (bars)')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Trade Duration Distribution')
            ax5.grid(True, alpha=0.3)

            # Add average line
            avg_duration = duration_data.mean()
            ax5.axvline(x=avg_duration, color='red', linestyle='--', label=f'Avg: {avg_duration:.1f}')
            ax5.legend()

        # 6. Win/Loss by Direction
        ax6 = plt.subplot(2, 3, 6)
        if 'direction' in self.trades.columns:
            direction_stats = []
            for direction in ['Long', 'Short']:
                dir_trades = self.trades[self.trades['direction'] == direction]
                if len(dir_trades) > 0:
                    wins = len(dir_trades[dir_trades['pnl'] > 0])
                    losses = len(dir_trades[dir_trades['pnl'] <= 0])
                    direction_stats.append((direction, wins, losses))

            if direction_stats:
                directions = [stat[0] for stat in direction_stats]
                wins = [stat[1] for stat in direction_stats]
                losses = [stat[2] for stat in direction_stats]

                x = np.arange(len(directions))
                width = 0.35

                ax6.bar(x - width / 2, wins, width, label='Wins', color='green', alpha=0.7)
                ax6.bar(x + width / 2, losses, width, label='Losses', color='red', alpha=0.7)

                ax6.set_xlabel('Direction')
                ax6.set_ylabel('Number of Trades')
                ax6.set_title('Wins/Losses by Direction')
                ax6.set_xticks(x)
                ax6.set_xticklabels(directions)
                ax6.legend()
                ax6.grid(True, alpha=0.3)

                # Add percentage labels
                for i, (direction, win_count, loss_count) in enumerate(direction_stats):
                    total = win_count + loss_count
                    win_pct = win_count / total * 100 if total > 0 else 0
                    ax6.text(i, max(win_count, loss_count) + 0.5, f'{win_pct:.1f}%',
                             ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        return fig

    def generate_report(self, filename: str = None) -> str:
        """Generate detailed text report"""
        metrics = self.calculate_metrics()

        if filename is None:
            filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE BACKTEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Performance Summary
        report_lines.append("PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if '$' in key:
                    report_lines.append(f"{key:<25}: ${value:>10,.2f}")
                elif '%' in key:
                    report_lines.append(f"{key:<25}: {value:>10.2f}%")
                else:
                    report_lines.append(f"{key:<25}: {value:>10}")
            else:
                report_lines.append(f"{key:<25}: {value}")

        report_lines.append("")

        # Trade Analysis
        if len(self.trades) > 0:
            report_lines.append("DETAILED TRADE ANALYSIS")
            report_lines.append("-" * 40)

            # Best and worst trades
            best_trade = self.trades.loc[self.trades['pnl'].idxmax()]
            worst_trade = self.trades.loc[self.trades['pnl'].idxmin()]

            report_lines.append(f"Best Trade:")
            report_lines.append(f"  Date: {best_trade.get('entry_time', 'N/A')}")
            report_lines.append(f"  Direction: {best_trade.get('direction', 'N/A')}")
            report_lines.append(f"  P&L: ${best_trade['pnl']:.2f}")
            report_lines.append("")

            report_lines.append(f"Worst Trade:")
            report_lines.append(f"  Date: {worst_trade.get('entry_time', 'N/A')}")
            report_lines.append(f"  Direction: {worst_trade.get('direction', 'N/A')}")
            report_lines.append(f"  P&L: ${worst_trade['pnl']:.2f}")
            report_lines.append("")

        # Monthly performance
        monthly_returns = self._calculate_periodic_returns('M')
        if len(monthly_returns) > 0:
            report_lines.append("MONTHLY PERFORMANCE")
            report_lines.append("-" * 40)
            for i, ret in enumerate(monthly_returns, 1):
                report_lines.append(f"Month {i:2d}: {ret:>6.2f}%")
            report_lines.append("")

        report_text = "\n".join(report_lines)

        # Save to file
        try:
            os.makedirs('reports', exist_ok=True)
            full_path = os.path.join('reports', filename)
            with open(full_path, 'w') as f:
                f.write(report_text)
            print(f"✅ Report saved to {full_path}")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")

        return report_text