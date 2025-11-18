"""
Performance analysis module for backtesting results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from src.backtesting.engine import BacktestResult
from src.backtesting.metrics import PerformanceMetrics
from src.utils.logger import LoggerMixin


class PerformanceAnalyzer(LoggerMixin):
    """
    Analyze and compare backtesting results across multiple strategies and tickers.
    """

    def __init__(self, results_dir: Optional[Union[str, Path]] = None):
        """
        Initialize performance analyzer.

        Parameters
        ----------
        results_dir : str or Path, optional
            Directory containing backtest results
        """
        self.results_dir = Path(results_dir) if results_dir else None
        self.results: List[BacktestResult] = []

        if self.results_dir and self.results_dir.exists():
            self.load_results()

    def load_results(self, results_dir: Optional[Union[str, Path]] = None) -> List[BacktestResult]:
        """
        Load all backtest results from directory.

        Parameters
        ----------
        results_dir : str or Path, optional
            Directory to load from (uses self.results_dir if not provided)

        Returns
        -------
        list
            List of BacktestResult objects
        """
        if results_dir:
            self.results_dir = Path(results_dir)

        if not self.results_dir or not self.results_dir.exists():
            self.logger.warning(f"Results directory not found: {self.results_dir}")
            return []

        self.results = []

        # Find all summary.json files
        for summary_file in self.results_dir.rglob("summary.json"):
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)

                result_dir = summary_file.parent

                # Load equity curve
                equity_file = result_dir / "equity_curve.csv"
                equity_curve = None
                if equity_file.exists():
                    equity_curve = pd.read_csv(
                        equity_file, index_col=0, parse_dates=True
                    ).squeeze()

                # Load trades
                trades_file = result_dir / "trades.csv"
                trades_log = None
                if trades_file.exists():
                    trades_log = pd.read_csv(
                        trades_file,
                        parse_dates=['entry_date', 'exit_date']
                    )

                # Load full metrics
                metrics_file = result_dir / "metrics.json"
                metrics = None
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics_dict = json.load(f)
                        metrics = PerformanceMetrics(**metrics_dict)

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
                    metrics=metrics,
                    equity_curve=equity_curve,
                    trades_log=trades_log
                )

                self.results.append(result)

            except Exception as e:
                self.logger.error(f"Error loading {summary_file}: {e}")
                continue

        self.logger.info(f"Loaded {len(self.results)} backtest results")

        return self.results

    def calculate_all_metrics(self) -> pd.DataFrame:
        """
        Calculate metrics for all results.

        Returns
        -------
        pd.DataFrame
            DataFrame with all metrics for all strategy-ticker combinations
        """
        if not self.results:
            return pd.DataFrame()

        metrics_list = []

        for result in self.results:
            metric_dict = {
                'ticker': result.ticker,
                'strategy': result.strategy_name,
                'start_date': result.start_date,
                'end_date': result.end_date,
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': result.max_drawdown,
                'volatility': result.volatility,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_trades': result.total_trades
            }

            # Add additional metrics if available
            if result.metrics:
                metric_dict.update({
                    'cagr': result.metrics.cagr,
                    'calmar_ratio': result.metrics.calmar_ratio,
                    'avg_drawdown': result.metrics.avg_drawdown,
                    'winning_trades': result.metrics.winning_trades,
                    'losing_trades': result.metrics.losing_trades,
                    'avg_win': result.metrics.avg_win,
                    'avg_loss': result.metrics.avg_loss,
                    'largest_win': result.metrics.largest_win,
                    'largest_loss': result.metrics.largest_loss,
                    'avg_holding_period': result.metrics.avg_holding_period
                })

            metrics_list.append(metric_dict)

        df = pd.DataFrame(metrics_list)

        return df

    def find_best_strategy_per_ticker(
        self,
        metric: str = 'sharpe_ratio'
    ) -> Dict[str, BacktestResult]:
        """
        Find the best strategy for each ticker.

        Parameters
        ----------
        metric : str
            Metric to optimize ('sharpe_ratio', 'total_return', 'calmar_ratio', etc.)

        Returns
        -------
        dict
            Dictionary mapping ticker to best BacktestResult
        """
        if not self.results:
            return {}

        best_per_ticker = {}

        # Group by ticker
        tickers = set(r.ticker for r in self.results)

        for ticker in tickers:
            ticker_results = [r for r in self.results if r.ticker == ticker]

            # Find best by metric
            best_result = max(
                ticker_results,
                key=lambda r: getattr(r, metric, float('-inf'))
            )

            best_per_ticker[ticker] = best_result

        self.logger.info(
            f"Found best strategies for {len(best_per_ticker)} tickers "
            f"(optimizing {metric})"
        )

        return best_per_ticker

    def get_strategy_rankings(
        self,
        metric: str = 'sharpe_ratio',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Rank all strategy-ticker combinations by a metric.

        Parameters
        ----------
        metric : str
            Metric to rank by
        ascending : bool
            Sort order (False = descending, best first)

        Returns
        -------
        pd.DataFrame
            Ranked strategies
        """
        df = self.calculate_all_metrics()

        if df.empty:
            return df

        # Sort by metric
        df_sorted = df.sort_values(by=metric, ascending=ascending)

        # Add rank
        df_sorted['rank'] = range(1, len(df_sorted) + 1)

        return df_sorted

    def get_strategy_summary_by_type(self) -> pd.DataFrame:
        """
        Summarize performance by strategy type.

        Returns
        -------
        pd.DataFrame
            Summary statistics grouped by strategy
        """
        df = self.calculate_all_metrics()

        if df.empty:
            return df

        # Group by strategy
        summary = df.groupby('strategy').agg({
            'total_return': ['mean', 'std', 'min', 'max'],
            'sharpe_ratio': ['mean', 'std', 'min', 'max'],
            'max_drawdown': ['mean', 'std', 'min', 'max'],
            'win_rate': ['mean', 'std', 'min', 'max'],
            'total_trades': ['mean', 'sum']
        }).round(4)

        return summary

    def get_ticker_summary(self) -> pd.DataFrame:
        """
        Summarize performance by ticker.

        Returns
        -------
        pd.DataFrame
            Summary statistics grouped by ticker
        """
        df = self.calculate_all_metrics()

        if df.empty:
            return df

        # Group by ticker
        summary = df.groupby('ticker').agg({
            'total_return': ['mean', 'std', 'min', 'max'],
            'sharpe_ratio': ['mean', 'std', 'min', 'max'],
            'max_drawdown': ['mean', 'std', 'min', 'max'],
            'win_rate': ['mean', 'std', 'min', 'max']
        }).round(4)

        return summary

    def get_correlation_matrix(
        self,
        method: str = 'returns'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between strategies.

        Parameters
        ----------
        method : str
            'returns' or 'equity' - what to correlate

        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        if not self.results:
            return pd.DataFrame()

        # Create a DataFrame with equity curves or returns
        data = {}

        for result in self.results:
            if result.equity_curve is None:
                continue

            key = f"{result.ticker}_{result.strategy_name}"

            if method == 'returns':
                data[key] = result.equity_curve.pct_change().fillna(0)
            else:
                data[key] = result.equity_curve

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Calculate correlation
        corr_matrix = df.corr()

        return corr_matrix

    def identify_best_worst_performers(
        self,
        n: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Identify best and worst performing strategies.

        Parameters
        ----------
        n : int
            Number of top/bottom strategies to return

        Returns
        -------
        dict
            Dictionary with 'best' and 'worst' DataFrames
        """
        df = self.calculate_all_metrics()

        if df.empty:
            return {'best': pd.DataFrame(), 'worst': pd.DataFrame()}

        # Sort by Sharpe ratio
        df_sorted = df.sort_values(by='sharpe_ratio', ascending=False)

        results = {
            'best': df_sorted.head(n),
            'worst': df_sorted.tail(n)
        }

        return results

    def calculate_portfolio_metrics(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Calculate metrics for a portfolio of strategies.

        Parameters
        ----------
        weights : dict, optional
            Weights for each strategy (equal weight if None)

        Returns
        -------
        dict
            Portfolio metrics
        """
        if not self.results:
            return {}

        # Get all equity curves
        equity_curves = {}
        for result in self.results:
            if result.equity_curve is None:
                continue
            key = f"{result.ticker}_{result.strategy_name}"
            equity_curves[key] = result.equity_curve

        if not equity_curves:
            return {}

        # Create DataFrame
        df = pd.DataFrame(equity_curves)

        # Apply weights
        if weights is None:
            # Equal weight
            weights = {k: 1.0 / len(equity_curves) for k in equity_curves.keys()}

        # Calculate portfolio equity
        portfolio_equity = sum(
            df[k] * weights.get(k, 0) for k in equity_curves.keys()
        )

        # Calculate metrics
        from src.backtesting.metrics import MetricsCalculator

        calculator = MetricsCalculator()
        metrics = calculator.calculate_all_metrics(portfolio_equity)

        return metrics.to_dict()

    def export_summary(
        self,
        output_file: Union[str, Path],
        format: str = 'csv'
    ) -> None:
        """
        Export summary to file.

        Parameters
        ----------
        output_file : str or Path
            Output file path
        format : str
            Format ('csv', 'excel', 'json')
        """
        df = self.calculate_all_metrics()

        if df.empty:
            self.logger.warning("No data to export")
            return

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        self.logger.info(f"Exported summary to {output_path}")
