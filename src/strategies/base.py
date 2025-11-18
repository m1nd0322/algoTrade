"""
Base class for all trading strategies.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.utils.logger import LoggerMixin


@dataclass
class TradeSignal:
    """
    Represents a trading signal.
    """
    date: datetime
    ticker: str
    signal: int  # 1 = buy, -1 = sell, 0 = hold
    strength: float = 1.0  # Signal strength (0-1)
    price: float = 0.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """
    Represents a trading position.
    """
    ticker: str
    shares: float
    entry_price: float
    entry_date: datetime
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return (self.current_price - self.entry_price) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


class BaseStrategy(ABC, LoggerMixin):
    """
    Abstract base class for all trading strategies.

    All strategy implementations must inherit from this class and
    implement the required abstract methods.
    """

    # Strategy metadata
    strategy_name: str = "BaseStrategy"
    strategy_type: str = "base"  # traditional, ml, dl
    version: str = "1.0.0"

    def __init__(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize base strategy.

        Parameters
        ----------
        data : pd.DataFrame
            Historical price data with OHLCV columns
        params : dict, optional
            Strategy-specific parameters
        config : dict, optional
            General configuration
        """
        self.data = data.copy()
        self.params = params or {}
        self.config = config or {}

        # Trading state
        self.signals = pd.Series(index=data.index, dtype=float)
        self.positions = pd.Series(index=data.index, dtype=float)
        self.equity_curve = pd.Series(index=data.index, dtype=float)

        # Position tracking
        self.current_position: Optional[Position] = None
        self.trades = []

        # Performance metrics
        self.metrics = {}

        self.logger.info(
            f"Initialized {self.strategy_name} with {len(data)} data points"
        )

    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals based on strategy logic.

        Returns
        -------
        pd.Series
            Series of signals (1 = buy, -1 = sell, 0 = hold)

        Notes
        -----
        This method must be implemented by all strategy subclasses.
        """
        pass

    @abstractmethod
    def calculate_positions(self, signals: pd.Series) -> pd.Series:
        """
        Calculate positions from signals.

        Parameters
        ----------
        signals : pd.Series
            Trading signals

        Returns
        -------
        pd.Series
            Position sizes

        Notes
        -----
        This method must be implemented by all strategy subclasses.
        """
        pass

    def backtest(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001
    ) -> Dict[str, Any]:
        """
        Run backtest of the strategy.

        Parameters
        ----------
        initial_capital : float
            Initial capital
        commission : float
            Commission rate (as decimal)

        Returns
        -------
        dict
            Backtest results including metrics and equity curve
        """
        self.logger.info(f"Running backtest for {self.strategy_name}...")

        # Generate signals
        self.signals = self.generate_signals()

        # Calculate positions
        self.positions = self.calculate_positions(self.signals)

        # Calculate returns
        returns = self.calculate_returns(self.positions, commission)

        # Calculate equity curve
        self.equity_curve = initial_capital * (1 + returns).cumprod()

        # Calculate metrics
        self.metrics = self.calculate_metrics(returns, initial_capital)

        self.logger.info(
            f"Backtest complete. Total Return: {self.metrics['total_return']:.2%}"
        )

        return {
            'signals': self.signals,
            'positions': self.positions,
            'equity_curve': self.equity_curve,
            'metrics': self.metrics,
            'trades': self.trades
        }

    def calculate_returns(
        self,
        positions: pd.Series,
        commission: float = 0.001
    ) -> pd.Series:
        """
        Calculate strategy returns.

        Parameters
        ----------
        positions : pd.Series
            Position sizes
        commission : float
            Commission rate

        Returns
        -------
        pd.Series
            Strategy returns
        """
        # Calculate market returns
        market_returns = self.data['Close'].pct_change()

        # Calculate strategy returns (positions shifted by 1 for next-day execution)
        strategy_returns = positions.shift(1) * market_returns

        # Subtract trading costs
        position_changes = positions.diff().abs()
        trading_costs = position_changes * commission

        # Net returns
        net_returns = strategy_returns - trading_costs

        return net_returns.fillna(0)

    def calculate_metrics(
        self,
        returns: pd.Series,
        initial_capital: float
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        initial_capital : float
            Initial capital

        Returns
        -------
        dict
            Performance metrics
        """
        # Total return
        total_return = (1 + returns).prod() - 1

        # Annual return
        days = len(returns)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        risk_free_rate = self.config.get('risk_free_rate', 0.02)
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = (
            np.sqrt(252) * excess_returns.mean() / excess_returns.std()
            if excess_returns.std() > 0 else 0
        )

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (
            (annual_return - risk_free_rate) / downside_std
            if downside_std > 0 else 0
        )

        # Maximum drawdown
        equity = initial_capital * (1 + returns).cumprod()
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_days = (returns > 0).sum()
        total_trading_days = (returns != 0).sum()
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0

        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Total trades
        position_changes = self.positions.diff().fillna(0)
        total_trades = (position_changes != 0).sum()

        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'avg_daily_return': returns.mean(),
            'median_daily_return': returns.median(),
            'best_day': returns.max(),
            'worst_day': returns.min()
        }

        return metrics

    def get_parameter_grid(self) -> Dict[str, list]:
        """
        Get parameter grid for optimization.

        Returns
        -------
        dict
            Parameter grid for grid search

        Notes
        -----
        Override this method in subclasses to provide strategy-specific
        parameter ranges for optimization.
        """
        return {}

    def optimize_parameters(
        self,
        parameter_grid: Optional[Dict[str, list]] = None,
        metric: str = 'sharpe_ratio'
    ) -> Tuple[Dict, float]:
        """
        Optimize strategy parameters.

        Parameters
        ----------
        parameter_grid : dict, optional
            Parameter grid to search
        metric : str
            Metric to optimize

        Returns
        -------
        tuple
            (best_params, best_score)
        """
        if parameter_grid is None:
            parameter_grid = self.get_parameter_grid()

        if not parameter_grid:
            self.logger.warning("No parameter grid provided for optimization")
            return self.params, 0.0

        from itertools import product

        # Generate all parameter combinations
        keys = parameter_grid.keys()
        values = parameter_grid.values()
        combinations = [dict(zip(keys, v)) for v in product(*values)]

        best_score = -np.inf
        best_params = self.params.copy()

        self.logger.info(
            f"Optimizing {len(combinations)} parameter combinations..."
        )

        for params in combinations:
            # Update parameters
            self.params = params

            # Run backtest
            try:
                results = self.backtest()
                score = results['metrics'].get(metric, -np.inf)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                self.logger.warning(f"Error with params {params}: {e}")
                continue

        # Restore best parameters
        self.params = best_params

        self.logger.info(
            f"Optimization complete. Best {metric}: {best_score:.4f}"
        )

        return best_params, best_score

    def plot_results(self, show: bool = True, save_path: Optional[str] = None):
        """
        Plot strategy results.

        Parameters
        ----------
        show : bool
            Whether to display the plot
        save_path : str, optional
            Path to save the plot

        Notes
        -----
        Requires matplotlib. Override in subclasses for custom plots.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("matplotlib not installed")
            return

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Plot 1: Price and signals
        axes[0].plot(self.data.index, self.data['Close'], label='Price', alpha=0.7)
        buy_signals = self.signals[self.signals > 0]
        sell_signals = self.signals[self.signals < 0]
        axes[0].scatter(buy_signals.index, self.data.loc[buy_signals.index, 'Close'],
                       marker='^', color='g', label='Buy', s=100, alpha=0.7)
        axes[0].scatter(sell_signals.index, self.data.loc[sell_signals.index, 'Close'],
                       marker='v', color='r', label='Sell', s=100, alpha=0.7)
        axes[0].set_title(f'{self.strategy_name} - Signals')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Equity curve
        axes[1].plot(self.equity_curve.index, self.equity_curve, label='Strategy', linewidth=2)
        axes[1].set_title('Equity Curve')
        axes[1].set_ylabel('Portfolio Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Drawdown
        running_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - running_max) / running_max
        axes[2].fill_between(drawdown.index, 0, drawdown, alpha=0.3, color='red')
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def __repr__(self) -> str:
        """String representation of strategy."""
        return (
            f"{self.strategy_name}(type={self.strategy_type}, "
            f"version={self.version}, params={self.params})"
        )
