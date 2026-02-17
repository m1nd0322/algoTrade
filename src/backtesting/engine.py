"""
Backtesting engine for trading strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from src.backtesting.portfolio import Portfolio
from src.backtesting.metrics import MetricsCalculator, PerformanceMetrics
from src.strategies.base import BaseStrategy
from src.utils.logger import LoggerMixin


@dataclass
class BacktestConfig:
    """
    Configuration for backtesting.
    """
    # Capital settings
    initial_capital: float = 100000.0
    currency: str = "USD"

    # Position sizing
    position_sizing_method: str = "fixed_fraction"  # fixed_fraction, fixed_amount, risk_based
    fixed_fraction: float = 1.0  # 100% of capital
    fixed_amount: float = 10000.0
    max_position_size: float = 1.0

    # Trading costs
    commission_type: str = "percentage"
    commission_value: float = 0.001  # 0.1%
    min_commission: float = 1.0
    slippage_type: str = "percentage"
    slippage_value: float = 0.0005  # 0.05%

    # Risk management
    stop_loss_enabled: bool = False
    stop_loss_pct: float = 0.05
    take_profit_enabled: bool = False
    take_profit_pct: float = 0.10
    max_drawdown_limit: float = 0.20

    # Performance calculation
    risk_free_rate: float = 0.02
    periods_per_year: int = 252

    # Benchmark
    benchmark_ticker: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BacktestResult:
    """
    Results from backtesting.
    """
    strategy_name: str
    ticker: str
    start_date: datetime
    end_date: datetime

    # Performance metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0

    # Additional data
    metrics: Optional[PerformanceMetrics] = None
    equity_curve: Optional[pd.Series] = None
    trades_log: Optional[pd.DataFrame] = None
    signals: Optional[pd.Series] = None
    positions: Optional[pd.Series] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding large data)."""
        return {
            'strategy_name': self.strategy_name,
            'ticker': self.ticker,
            'start_date': self.start_date.strftime('%Y-%m-%d') if self.start_date else None,
            'end_date': self.end_date.strftime('%Y-%m-%d') if self.end_date else None,
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades
        }


class BacktestEngine(LoggerMixin):
    """
    Engine for backtesting trading strategies.
    """

    def __init__(self, config: Optional[Union[Dict, BacktestConfig]] = None):
        """
        Initialize backtest engine.

        Parameters
        ----------
        config : dict or BacktestConfig, optional
            Backtesting configuration
        """
        if config is None:
            self.config = BacktestConfig()
        elif isinstance(config, dict):
            self.config = BacktestConfig(**config)
        else:
            self.config = config

        self.logger.info("Initialized BacktestEngine")

    def run(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        ticker: str = "UNKNOWN"
    ) -> BacktestResult:
        """
        Run backtest for a strategy.

        Parameters
        ----------
        strategy : BaseStrategy
            Trading strategy to backtest
        data : pd.DataFrame
            Historical price data
        ticker : str
            Ticker symbol

        Returns
        -------
        BacktestResult
            Backtesting results
        """
        self.logger.info(
            f"Starting backtest: {strategy.strategy_name} on {ticker}"
        )

        # Initialize portfolio
        portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission_value,
            slippage_rate=self.config.slippage_value
        )

        # Generate signals
        self.logger.info("Generating trading signals...")
        signals = strategy.generate_signals()

        # Execute trades based on signals
        self.logger.info("Executing trades...")
        self._execute_trades(portfolio, data, signals, ticker)

        # Calculate metrics
        self.logger.info("Calculating performance metrics...")
        equity_curve = portfolio.get_equity_curve()
        trades_df = portfolio.get_trades_df()

        metrics_calculator = MetricsCalculator(
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=self.config.periods_per_year
        )

        metrics = metrics_calculator.calculate_all_metrics(
            equity_curve=equity_curve,
            trades_df=trades_df
        )

        # Create result
        result = BacktestResult(
            strategy_name=strategy.strategy_name,
            ticker=ticker,
            start_date=data.index[0],
            end_date=data.index[-1],
            total_return=metrics.total_return,
            annual_return=metrics.annual_return,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown=metrics.max_drawdown,
            volatility=metrics.volatility,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            total_trades=metrics.total_trades,
            metrics=metrics,
            equity_curve=equity_curve,
            trades_log=trades_df,
            signals=signals,
            positions=strategy.positions
        )

        self.logger.info(
            f"Backtest complete: Total Return={metrics.total_return:.2%}, "
            f"Sharpe={metrics.sharpe_ratio:.2f}, Trades={metrics.total_trades}"
        )

        return result

    def _execute_trades(
        self,
        portfolio: Portfolio,
        data: pd.DataFrame,
        signals: pd.Series,
        ticker: str
    ) -> None:
        """
        Execute trades based on signals.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio manager
        data : pd.DataFrame
            Price data
        signals : pd.Series
            Trading signals
        ticker : str
            Ticker symbol
        """
        current_position = 0  # 0 = no position, 1 = long

        for i, (date, signal) in enumerate(signals.items()):
            if date not in data.index:
                continue

            # Get current price (use Close for execution)
            current_price = data.loc[date, 'Close']

            # Update portfolio history
            portfolio.update_history(date, {ticker: current_price})

            # Process signals
            if signal == 1 and current_position == 0:
                # Buy signal
                shares = self._calculate_position_size(
                    portfolio.cash,
                    current_price
                )

                if shares > 0:
                    success = portfolio.buy(
                        ticker=ticker,
                        shares=shares,
                        price=current_price,
                        date=date
                    )

                    if success:
                        current_position = 1

            elif signal == -1 and current_position == 1:
                # Sell signal
                if ticker in portfolio.positions:
                    shares = portfolio.positions[ticker].shares

                    success = portfolio.sell(
                        ticker=ticker,
                        shares=shares,
                        price=current_price,
                        date=date
                    )

                    if success:
                        current_position = 0

            # Check stop loss and take profit
            if current_position == 1 and ticker in portfolio.positions:
                position = portfolio.positions[ticker]
                price_change = (current_price - position.entry_price) / position.entry_price

                # Stop loss
                if self.config.stop_loss_enabled:
                    if price_change <= -self.config.stop_loss_pct:
                        self.logger.debug(
                            f"Stop loss triggered at {date}: {price_change:.2%}"
                        )
                        portfolio.sell(
                            ticker=ticker,
                            shares=position.shares,
                            price=current_price,
                            date=date
                        )
                        current_position = 0

                # Take profit
                if self.config.take_profit_enabled:
                    if price_change >= self.config.take_profit_pct:
                        self.logger.debug(
                            f"Take profit triggered at {date}: {price_change:.2%}"
                        )
                        portfolio.sell(
                            ticker=ticker,
                            shares=position.shares,
                            price=current_price,
                            date=date
                        )
                        current_position = 0

        # Close any remaining positions at the end
        if current_position == 1 and ticker in portfolio.positions:
            final_date = data.index[-1]
            final_price = data.loc[final_date, 'Close']
            position = portfolio.positions[ticker]

            portfolio.sell(
                ticker=ticker,
                shares=position.shares,
                price=final_price,
                date=final_date
            )

    def _calculate_position_size(
        self,
        available_cash: float,
        price: float
    ) -> float:
        """
        Calculate position size based on configuration.

        Parameters
        ----------
        available_cash : float
            Available cash
        price : float
            Current price

        Returns
        -------
        float
            Number of shares to buy
        """
        if self.config.position_sizing_method == "fixed_fraction":
            # Invest fixed fraction of capital
            amount = available_cash * self.config.fixed_fraction
            shares = amount / price

        elif self.config.position_sizing_method == "fixed_amount":
            # Invest fixed dollar amount
            amount = min(self.config.fixed_amount, available_cash)
            shares = amount / price

        else:
            # Default to full investment
            shares = available_cash / price

        # Apply max position size limit
        max_shares = (available_cash * self.config.max_position_size) / price
        shares = min(shares, max_shares)

        # Round down to whole shares
        shares = int(shares)

        return shares

    def save_results(
        self,
        results: Union[BacktestResult, list],
        output_dir: Union[str, Path]
    ) -> None:
        """
        Save backtest results to disk.

        Parameters
        ----------
        results : BacktestResult or list
            Single result or list of results
        output_dir : str or Path
            Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert single result to list
        if isinstance(results, BacktestResult):
            results = [results]

        for result in results:
            # Create subdirectory for this result
            result_dir = output_path / f"{result.ticker}_{result.strategy_name.replace(' ', '_')}"
            result_dir.mkdir(parents=True, exist_ok=True)

            # Save summary as JSON
            summary_file = result_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

            # Save equity curve
            if result.equity_curve is not None:
                equity_file = result_dir / "equity_curve.csv"
                result.equity_curve.to_csv(equity_file)

            # Save trades log
            if result.trades_log is not None and len(result.trades_log) > 0:
                trades_file = result_dir / "trades.csv"
                result.trades_log.to_csv(trades_file, index=False)

            # Save signals
            if result.signals is not None:
                signals_file = result_dir / "signals.csv"
                result.signals.to_csv(signals_file)

            # Save full metrics
            if result.metrics is not None:
                metrics_file = result_dir / "metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(result.metrics.to_dict(), f, indent=2)

            self.logger.info(f"Saved results to {result_dir}")

    def load_results(
        self,
        input_dir: Union[str, Path]
    ) -> list:
        """
        Load backtest results from disk.

        Parameters
        ----------
        input_dir : str or Path
            Input directory

        Returns
        -------
        list
            List of BacktestResult objects
        """
        input_path = Path(input_dir)
        results = []

        # Find all summary.json files
        for summary_file in input_path.rglob("summary.json"):
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)

                # Load additional data
                result_dir = summary_file.parent

                equity_file = result_dir / "equity_curve.csv"
                equity_curve = None
                if equity_file.exists():
                    equity_curve = pd.read_csv(
                        equity_file, index_col=0, parse_dates=True
                    ).squeeze()

                trades_file = result_dir / "trades.csv"
                trades_log = None
                if trades_file.exists():
                    trades_log = pd.read_csv(trades_file, parse_dates=['entry_date', 'exit_date'])

                # Create result object
                result = BacktestResult(
                    strategy_name=summary['strategy_name'],
                    ticker=summary['ticker'],
                    start_date=pd.to_datetime(summary['start_date']),
                    end_date=pd.to_datetime(summary['end_date']),
                    total_return=summary['total_return'],
                    annual_return=summary['annual_return'],
                    sharpe_ratio=summary['sharpe_ratio'],
                    sortino_ratio=summary['sortino_ratio'],
                    max_drawdown=summary['max_drawdown'],
                    volatility=summary['volatility'],
                    win_rate=summary['win_rate'],
                    profit_factor=summary['profit_factor'],
                    total_trades=summary['total_trades'],
                    equity_curve=equity_curve,
                    trades_log=trades_log
                )

                results.append(result)

            except Exception as e:
                self.logger.error(f"Error loading {summary_file}: {e}")
                continue

        self.logger.info(f"Loaded {len(results)} results from {input_path}")

        return results
