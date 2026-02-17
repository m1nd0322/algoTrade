"""
Strategy comparison and statistical analysis module.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import stats

from src.backtesting.engine import BacktestResult
from src.utils.logger import LoggerMixin


class StrategyComparator(LoggerMixin):
    """
    Compare and statistically analyze multiple trading strategies.
    """

    def __init__(self, results: List[BacktestResult]):
        """
        Initialize strategy comparator.

        Parameters
        ----------
        results : list
            List of BacktestResult objects to compare
        """
        self.results = results
        self.logger.info(f"Initialized comparator with {len(results)} results")

    def compare_metrics(
        self,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare key metrics across all strategies.

        Parameters
        ----------
        metrics : list, optional
            List of metrics to compare (all if None)

        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        if metrics is None:
            metrics = [
                'total_return', 'annual_return', 'sharpe_ratio',
                'sortino_ratio', 'max_drawdown', 'volatility',
                'win_rate', 'profit_factor'
            ]

        data = []

        for result in self.results:
            row = {
                'ticker': result.ticker,
                'strategy': result.strategy_name
            }

            for metric in metrics:
                row[metric] = getattr(result, metric, np.nan)

            data.append(row)

        df = pd.DataFrame(data)

        return df

    def rank_strategies(
        self,
        by: str = 'sharpe_ratio',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Rank strategies by a specific metric.

        Parameters
        ----------
        by : str
            Metric to rank by
        ascending : bool
            Sort order

        Returns
        -------
        pd.DataFrame
            Ranked strategies
        """
        df = self.compare_metrics()

        if df.empty:
            return df

        df_sorted = df.sort_values(by=by, ascending=ascending)
        df_sorted['rank'] = range(1, len(df_sorted) + 1)

        return df_sorted

    def calculate_risk_return_profile(self) -> pd.DataFrame:
        """
        Calculate risk-return profile for all strategies.

        Returns
        -------
        pd.DataFrame
            Risk-return metrics
        """
        data = []

        for result in self.results:
            data.append({
                'ticker': result.ticker,
                'strategy': result.strategy_name,
                'return': result.annual_return,
                'risk': result.volatility,
                'sharpe': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'return_to_risk': result.annual_return / result.volatility if result.volatility > 0 else 0,
                'return_to_mdd': result.annual_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0
            })

        df = pd.DataFrame(data)

        return df

    def perform_t_test(
        self,
        strategy1_name: str,
        strategy2_name: str,
        ticker: Optional[str] = None
    ) -> Dict:
        """
        Perform t-test to compare returns between two strategies.

        Parameters
        ----------
        strategy1_name : str
            First strategy name
        strategy2_name : str
            Second strategy name
        ticker : str, optional
            Specific ticker to compare (all if None)

        Returns
        -------
        dict
            T-test results
        """
        # Get results for both strategies
        results1 = [
            r for r in self.results
            if r.strategy_name == strategy1_name and (ticker is None or r.ticker == ticker)
        ]

        results2 = [
            r for r in self.results
            if r.strategy_name == strategy2_name and (ticker is None or r.ticker == ticker)
        ]

        if not results1 or not results2:
            return {'error': 'Strategy not found'}

        # Get returns series
        returns1 = []
        returns2 = []

        for r in results1:
            if r.equity_curve is not None:
                returns1.extend(r.equity_curve.pct_change().dropna().values)

        for r in results2:
            if r.equity_curve is not None:
                returns2.extend(r.equity_curve.pct_change().dropna().values)

        if not returns1 or not returns2:
            return {'error': 'No return data available'}

        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(returns1, returns2)

        result = {
            'strategy1': strategy1_name,
            'strategy2': strategy2_name,
            'ticker': ticker or 'all',
            't_statistic': t_statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_return1': np.mean(returns1),
            'mean_return2': np.mean(returns2),
            'std_return1': np.std(returns1),
            'std_return2': np.std(returns2)
        }

        return result

    def calculate_drawdown_comparison(self) -> pd.DataFrame:
        """
        Compare drawdown characteristics across strategies.

        Returns
        -------
        pd.DataFrame
            Drawdown comparison
        """
        data = []

        for result in self.results:
            if result.equity_curve is None:
                continue

            # Calculate drawdown
            running_max = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - running_max) / running_max

            # Find all drawdown periods
            is_drawdown = drawdown < 0
            drawdown_starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
            drawdown_ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)

            num_drawdowns = drawdown_starts.sum()
            avg_drawdown = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0

            data.append({
                'ticker': result.ticker,
                'strategy': result.strategy_name,
                'max_drawdown': result.max_drawdown,
                'avg_drawdown': avg_drawdown,
                'num_drawdowns': num_drawdowns,
                'time_underwater_pct': (is_drawdown.sum() / len(is_drawdown)) * 100
            })

        df = pd.DataFrame(data)

        return df

    def calculate_consistency_metrics(self) -> pd.DataFrame:
        """
        Calculate consistency metrics (rolling performance).

        Returns
        -------
        pd.DataFrame
            Consistency metrics
        """
        data = []

        for result in self.results:
            if result.equity_curve is None:
                continue

            returns = result.equity_curve.pct_change().fillna(0)

            # Monthly returns
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

            # Calculate consistency metrics
            positive_months_pct = (monthly_returns > 0).sum() / len(monthly_returns) * 100
            avg_positive_month = monthly_returns[monthly_returns > 0].mean() if (monthly_returns > 0).any() else 0
            avg_negative_month = monthly_returns[monthly_returns < 0].mean() if (monthly_returns < 0).any() else 0

            data.append({
                'ticker': result.ticker,
                'strategy': result.strategy_name,
                'positive_months_pct': positive_months_pct,
                'avg_positive_month': avg_positive_month,
                'avg_negative_month': avg_negative_month,
                'best_month': monthly_returns.max(),
                'worst_month': monthly_returns.min(),
                'monthly_std': monthly_returns.std()
            })

        df = pd.DataFrame(data)

        return df

    def identify_dominant_strategy(
        self,
        metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        Identify the strategy that dominates across multiple metrics.

        Parameters
        ----------
        metrics : list, optional
            Metrics to consider

        Returns
        -------
        dict
            Dominant strategy analysis
        """
        if metrics is None:
            metrics = ['sharpe_ratio', 'calmar_ratio', 'win_rate']

        # Count wins per strategy
        wins = {}

        for metric in metrics:
            # Higher is better for these metrics
            if metric in ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                         'win_rate', 'profit_factor', 'total_return']:
                best = max(self.results, key=lambda r: getattr(r, metric, float('-inf')))
            # Lower is better for these
            elif metric in ['max_drawdown', 'volatility']:
                best = min(self.results, key=lambda r: abs(getattr(r, metric, float('inf'))))
            else:
                continue

            key = f"{best.ticker}_{best.strategy_name}"
            wins[key] = wins.get(key, 0) + 1

        # Find dominant strategy
        if wins:
            dominant_key = max(wins, key=wins.get)
            dominant_wins = wins[dominant_key]

            result = {
                'dominant_strategy': dominant_key,
                'wins': dominant_wins,
                'total_metrics': len(metrics),
                'win_percentage': (dominant_wins / len(metrics)) * 100,
                'all_wins': wins
            }
        else:
            result = {'error': 'Could not determine dominant strategy'}

        return result

    def calculate_correlation_with_benchmark(
        self,
        benchmark: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate correlation of each strategy with a benchmark.

        Parameters
        ----------
        benchmark : pd.Series
            Benchmark equity curve or returns

        Returns
        -------
        pd.DataFrame
            Correlation metrics
        """
        data = []

        for result in self.results:
            if result.equity_curve is None:
                continue

            # Align with benchmark
            equity_aligned, benchmark_aligned = result.equity_curve.align(
                benchmark, join='inner'
            )

            if len(equity_aligned) == 0:
                continue

            # Calculate returns
            strategy_returns = equity_aligned.pct_change().fillna(0)
            benchmark_returns = benchmark_aligned.pct_change().fillna(0)

            # Calculate correlation
            correlation = strategy_returns.corr(benchmark_returns)

            # Calculate beta
            covariance = strategy_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            # Calculate alpha (Jensen's alpha)
            avg_strategy_return = strategy_returns.mean()
            avg_benchmark_return = benchmark_returns.mean()
            alpha = avg_strategy_return - beta * avg_benchmark_return

            data.append({
                'ticker': result.ticker,
                'strategy': result.strategy_name,
                'correlation': correlation,
                'beta': beta,
                'alpha': alpha * 252  # Annualized
            })

        df = pd.DataFrame(data)

        return df

    def perform_pairwise_comparison(
        self,
        metric: str = 'sharpe_ratio'
    ) -> pd.DataFrame:
        """
        Perform pairwise comparison between all strategies.

        Parameters
        ----------
        metric : str
            Metric to compare

        Returns
        -------
        pd.DataFrame
            Pairwise comparison matrix
        """
        # Get unique strategy names
        strategies = list(set(r.strategy_name for r in self.results))

        # Create comparison matrix
        matrix = pd.DataFrame(index=strategies, columns=strategies)

        for i, strat1 in enumerate(strategies):
            for j, strat2 in enumerate(strategies):
                if i == j:
                    matrix.loc[strat1, strat2] = 0
                    continue

                # Get metric values
                values1 = [
                    getattr(r, metric, 0) for r in self.results
                    if r.strategy_name == strat1
                ]
                values2 = [
                    getattr(r, metric, 0) for r in self.results
                    if r.strategy_name == strat2
                ]

                # Calculate difference
                diff = np.mean(values1) - np.mean(values2)
                matrix.loc[strat1, strat2] = diff

        return matrix.astype(float)

    def generate_comparison_summary(self) -> Dict:
        """
        Generate comprehensive comparison summary.

        Returns
        -------
        dict
            Summary of all comparisons
        """
        summary = {
            'total_strategies': len(set(r.strategy_name for r in self.results)),
            'total_tickers': len(set(r.ticker for r in self.results)),
            'total_results': len(self.results)
        }

        # Best by each metric
        metrics = ['sharpe_ratio', 'total_return', 'win_rate']
        summary['best_by_metric'] = {}

        for metric in metrics:
            best = max(self.results, key=lambda r: getattr(r, metric, float('-inf')))
            summary['best_by_metric'][metric] = {
                'strategy': best.strategy_name,
                'ticker': best.ticker,
                'value': getattr(best, metric, 0)
            }

        # Lowest risk
        lowest_risk = min(self.results, key=lambda r: abs(r.max_drawdown))
        summary['lowest_risk'] = {
            'strategy': lowest_risk.strategy_name,
            'ticker': lowest_risk.ticker,
            'max_drawdown': lowest_risk.max_drawdown
        }

        return summary
