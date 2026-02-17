"""
Performance metrics calculation for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from src.utils.logger import LoggerMixin


@dataclass
class PerformanceMetrics:
    """
    Container for performance metrics.
    """
    # Return metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    cagr: float = 0.0
    monthly_return: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_holding_period: float = 0.0

    # Additional metrics
    avg_daily_return: float = 0.0
    median_daily_return: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'cagr': self.cagr,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades
        }


class MetricsCalculator(LoggerMixin):
    """
    Calculate performance metrics from equity curve and trades.
    """

    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        """
        Initialize metrics calculator.

        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate (default: 2%)
        periods_per_year : int
            Number of trading periods per year (default: 252 for daily)
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        trades_df: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity curve
        trades_df : pd.DataFrame, optional
            DataFrame of completed trades
        benchmark : pd.Series, optional
            Benchmark equity curve for comparison

        Returns
        -------
        PerformanceMetrics
            Calculated metrics
        """
        metrics = PerformanceMetrics()

        if len(equity_curve) == 0:
            return metrics

        # Calculate returns
        returns = equity_curve.pct_change().fillna(0)

        # Return metrics
        metrics.total_return = self._calculate_total_return(equity_curve)
        metrics.annual_return = self._calculate_annual_return(equity_curve)
        metrics.cagr = self._calculate_cagr(equity_curve)

        # Risk metrics
        metrics.volatility = self._calculate_volatility(returns)
        (
            metrics.max_drawdown,
            _,
            _,
            metrics.max_drawdown_duration
        ) = self._calculate_max_drawdown(equity_curve)
        metrics.avg_drawdown = self._calculate_avg_drawdown(equity_curve)

        # Risk-adjusted returns
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(returns)
        metrics.sortino_ratio = self._calculate_sortino_ratio(returns)
        metrics.calmar_ratio = self._calculate_calmar_ratio(
            metrics.annual_return, metrics.max_drawdown
        )

        if benchmark is not None:
            metrics.information_ratio = self._calculate_information_ratio(
                equity_curve, benchmark
            )

        # Daily return statistics
        metrics.avg_daily_return = returns.mean()
        metrics.median_daily_return = returns.median()
        metrics.best_day = returns.max()
        metrics.worst_day = returns.min()

        # Trade statistics
        if trades_df is not None and len(trades_df) > 0:
            trade_metrics = self._calculate_trade_metrics(trades_df)
            metrics.total_trades = trade_metrics['total_trades']
            metrics.winning_trades = trade_metrics['winning_trades']
            metrics.losing_trades = trade_metrics['losing_trades']
            metrics.win_rate = trade_metrics['win_rate']
            metrics.profit_factor = trade_metrics['profit_factor']
            metrics.avg_win = trade_metrics['avg_win']
            metrics.avg_loss = trade_metrics['avg_loss']
            metrics.largest_win = trade_metrics['largest_win']
            metrics.largest_loss = trade_metrics['largest_loss']
            metrics.avg_holding_period = trade_metrics['avg_holding_period']

        return metrics

    def _calculate_total_return(self, equity_curve: pd.Series) -> float:
        """Calculate total return."""
        if len(equity_curve) == 0:
            return 0.0

        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]

        if initial_value == 0:
            return 0.0

        return (final_value - initial_value) / initial_value

    def _calculate_annual_return(self, equity_curve: pd.Series) -> float:
        """Calculate annualized return."""
        total_return = self._calculate_total_return(equity_curve)
        days = len(equity_curve)

        if days == 0:
            return 0.0

        years = days / self.periods_per_year

        if years == 0:
            return 0.0

        annual_return = (1 + total_return) ** (1 / years) - 1

        return annual_return

    def _calculate_cagr(self, equity_curve: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        if len(equity_curve) == 0:
            return 0.0

        initial_value = equity_curve.iloc[0]
        final_value = equity_curve.iloc[-1]
        days = len(equity_curve)

        if initial_value == 0 or days == 0:
            return 0.0

        years = days / self.periods_per_year

        if years == 0:
            return 0.0

        cagr = (final_value / initial_value) ** (1 / years) - 1

        return cagr

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0

        return returns.std() * np.sqrt(self.periods_per_year)

    def _calculate_max_drawdown(
        self,
        equity_curve: pd.Series
    ) -> Tuple[float, pd.Timestamp, pd.Timestamp, int]:
        """
        Calculate maximum drawdown.

        Returns
        -------
        tuple
            (max_drawdown, peak_date, trough_date, duration)
        """
        if len(equity_curve) == 0:
            return 0.0, None, None, 0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max

        # Find maximum drawdown
        max_dd = drawdown.min()

        # Find peak and trough dates
        trough_date = drawdown.idxmin()
        peak_date = equity_curve[:trough_date].idxmax()

        # Calculate duration
        duration = (trough_date - peak_date).days if trough_date and peak_date else 0

        return max_dd, peak_date, trough_date, duration

    def _calculate_avg_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate average drawdown."""
        if len(equity_curve) == 0:
            return 0.0

        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        # Only consider drawdown periods
        drawdowns = drawdown[drawdown < 0]

        if len(drawdowns) == 0:
            return 0.0

        return drawdowns.mean()

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.periods_per_year

        if excess_returns.std() == 0:
            return 0.0

        sharpe = np.sqrt(self.periods_per_year) * excess_returns.mean() / excess_returns.std()

        return sharpe

    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.periods_per_year

        # Only consider downside returns
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return 0.0

        downside_std = downside_returns.std() * np.sqrt(self.periods_per_year)

        if downside_std == 0:
            return 0.0

        sortino = (excess_returns.mean() * self.periods_per_year) / downside_std

        return sortino

    def _calculate_calmar_ratio(
        self,
        annual_return: float,
        max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0

        return annual_return / abs(max_drawdown)

    def _calculate_information_ratio(
        self,
        equity_curve: pd.Series,
        benchmark: pd.Series
    ) -> float:
        """Calculate Information ratio."""
        if len(equity_curve) == 0 or len(benchmark) == 0:
            return 0.0

        # Align series
        aligned_equity, aligned_benchmark = equity_curve.align(benchmark, join='inner')

        # Calculate returns
        strategy_returns = aligned_equity.pct_change().fillna(0)
        benchmark_returns = aligned_benchmark.pct_change().fillna(0)

        # Calculate excess returns
        excess_returns = strategy_returns - benchmark_returns

        if excess_returns.std() == 0:
            return 0.0

        information_ratio = (
            np.sqrt(self.periods_per_year) *
            excess_returns.mean() /
            excess_returns.std()
        )

        return information_ratio

    def _calculate_trade_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate trade-based metrics."""
        metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_holding_period': 0.0
        }

        if len(trades_df) == 0:
            return metrics

        # Total trades
        metrics['total_trades'] = len(trades_df)

        # Winning and losing trades
        winning_trades = trades_df[trades_df['profit_loss'] > 0]
        losing_trades = trades_df[trades_df['profit_loss'] < 0]

        metrics['winning_trades'] = len(winning_trades)
        metrics['losing_trades'] = len(losing_trades)

        # Win rate
        metrics['win_rate'] = (
            metrics['winning_trades'] / metrics['total_trades']
            if metrics['total_trades'] > 0 else 0.0
        )

        # Average win/loss
        metrics['avg_win'] = (
            winning_trades['profit_loss'].mean()
            if len(winning_trades) > 0 else 0.0
        )
        metrics['avg_loss'] = (
            losing_trades['profit_loss'].mean()
            if len(losing_trades) > 0 else 0.0
        )

        # Profit factor
        gross_profit = winning_trades['profit_loss'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 else 0

        metrics['profit_factor'] = (
            gross_profit / gross_loss
            if gross_loss > 0 else 0.0
        )

        # Largest win/loss
        metrics['largest_win'] = trades_df['profit_loss'].max()
        metrics['largest_loss'] = trades_df['profit_loss'].min()

        # Average holding period
        if 'holding_period' in trades_df.columns:
            metrics['avg_holding_period'] = trades_df['holding_period'].mean()

        return metrics

    def calculate_rolling_metrics(
        self,
        equity_curve: pd.Series,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.

        Parameters
        ----------
        equity_curve : pd.Series
            Portfolio equity curve
        window : int
            Rolling window size

        Returns
        -------
        pd.DataFrame
            Rolling metrics
        """
        returns = equity_curve.pct_change().fillna(0)

        rolling_metrics = pd.DataFrame(index=equity_curve.index)

        # Rolling return
        rolling_metrics['return'] = (
            equity_curve.pct_change(periods=window)
        )

        # Rolling volatility
        rolling_metrics['volatility'] = (
            returns.rolling(window=window).std() * np.sqrt(self.periods_per_year)
        )

        # Rolling Sharpe ratio
        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        rolling_metrics['sharpe_ratio'] = (
            np.sqrt(self.periods_per_year) *
            excess_returns.rolling(window=window).mean() /
            excess_returns.rolling(window=window).std()
        )

        return rolling_metrics
