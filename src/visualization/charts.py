"""
Chart generation for backtesting results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Union, Dict

from src.backtesting.engine import BacktestResult
from src.visualization.plots import PlotUtilities
from src.utils.logger import LoggerMixin


class ChartGenerator(LoggerMixin):
    """
    Generate comprehensive charts from backtesting results.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize chart generator.

        Parameters
        ----------
        config : dict, optional
            Visualization configuration
        """
        self.config = config or {}

        # Get settings from config
        general = self.config.get('general', {})
        self.style = general.get('style', 'seaborn')
        self.color_palette = general.get('color_palette', 'Set2')
        self.figsize = tuple(general.get('figure_size', [15, 10]))
        self.dpi = general.get('dpi', 100)
        self.save_format = general.get('save_format', 'png')
        self.save_dir = Path(general.get('save_dir', 'visualizations'))

        # Initialize plot utilities
        self.plotter = PlotUtilities(self.style, self.color_palette)

        self.logger.info("Initialized ChartGenerator")

    def generate_all_charts(
        self,
        results: List[BacktestResult],
        ticker: str,
        output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Generate all charts for a ticker's results.

        Parameters
        ----------
        results : list
            List of BacktestResult objects
        ticker : str
            Ticker symbol
        output_dir : str or Path, optional
            Output directory
        """
        if output_dir is None:
            output_dir = self.save_dir

        output_path = Path(output_dir) / ticker
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating charts for {ticker}...")

        # 1. Equity curves comparison
        self.plot_equity_curves_comparison(
            results,
            save_path=output_path / f"equity_curves_comparison.{self.save_format}"
        )

        # 2. Drawdown comparison
        self.plot_drawdown_comparison(
            results,
            save_path=output_path / f"drawdown_comparison.{self.save_format}"
        )

        # 3. Returns distribution
        self.plot_returns_distributions(
            results,
            save_path=output_path / f"returns_distributions.{self.save_format}"
        )

        # 4. Performance metrics comparison
        self.plot_metrics_comparison(
            results,
            save_path=output_path / f"metrics_comparison.{self.save_format}"
        )

        # 5. Individual strategy dashboards
        for result in results:
            self.create_strategy_dashboard(
                result,
                save_path=output_path / f"{result.strategy_name.replace(' ', '_')}_dashboard.{self.save_format}"
            )

        self.logger.info(f"Saved all charts to {output_path}")

    def plot_equity_curves_comparison(
        self,
        results: List[BacktestResult],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot equity curves for multiple strategies.

        Parameters
        ----------
        results : list
            List of BacktestResult objects
        save_path : Path, optional
            Path to save figure

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = self.plotter.setup_figure(self.figsize, self.dpi)

        for result in results:
            if result.equity_curve is None:
                continue

            ax.plot(
                result.equity_curve.index,
                result.equity_curve.values,
                label=result.strategy_name,
                linewidth=2
            )

        self.plotter.format_axis(
            ax,
            f"Equity Curves Comparison - {results[0].ticker if results else 'N/A'}",
            "Date",
            "Portfolio Value ($)",
            grid=True,
            legend=True
        )

        if save_path:
            self.plotter.save_figure(fig, save_path, dpi=self.dpi)
        else:
            plt.show()

        return fig

    def plot_drawdown_comparison(
        self,
        results: List[BacktestResult],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot drawdown comparison for multiple strategies.

        Parameters
        ----------
        results : list
            List of BacktestResult objects
        save_path : Path, optional
            Path to save figure

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = self.plotter.setup_figure(self.figsize, self.dpi)

        for result in results:
            if result.equity_curve is None:
                continue

            # Calculate drawdown
            running_max = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - running_max) / running_max

            ax.plot(
                drawdown.index,
                drawdown.values * 100,  # Convert to percentage
                label=result.strategy_name,
                linewidth=2
            )

        self.plotter.format_axis(
            ax,
            f"Drawdown Comparison - {results[0].ticker if results else 'N/A'}",
            "Date",
            "Drawdown (%)",
            grid=True,
            legend=True
        )

        if save_path:
            self.plotter.save_figure(fig, save_path, dpi=self.dpi)
        else:
            plt.show()

        return fig

    def plot_returns_distributions(
        self,
        results: List[BacktestResult],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot returns distributions for multiple strategies.

        Parameters
        ----------
        results : list
            List of BacktestResult objects
        save_path : Path, optional
            Path to save figure

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        n_strategies = len(results)
        n_cols = 2
        n_rows = (n_strategies + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), dpi=self.dpi)
        axes = axes.flatten() if n_strategies > 1 else [axes]

        for i, result in enumerate(results):
            if result.equity_curve is None:
                continue

            returns = result.equity_curve.pct_change().dropna()

            self.plotter.plot_returns_distribution(
                returns,
                title=f"{result.strategy_name} Returns Distribution",
                ax=axes[i]
            )

        # Hide unused subplots
        for i in range(n_strategies, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            self.plotter.save_figure(fig, save_path, dpi=self.dpi)
        else:
            plt.show()

        return fig

    def plot_metrics_comparison(
        self,
        results: List[BacktestResult],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot key metrics comparison as bar charts.

        Parameters
        ----------
        results : list
            List of BacktestResult objects
        save_path : Path, optional
            Path to save figure

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        axes = axes.flatten()

        metrics = [
            ('total_return', 'Total Return (%)', 100),
            ('sharpe_ratio', 'Sharpe Ratio', 1),
            ('max_drawdown', 'Max Drawdown (%)', 100),
            ('win_rate', 'Win Rate (%)', 100),
            ('profit_factor', 'Profit Factor', 1),
            ('volatility', 'Volatility (%)', 100)
        ]

        for idx, (metric, label, scale) in enumerate(metrics):
            strategies = [r.strategy_name for r in results]
            values = [getattr(r, metric, 0) * scale for r in results]

            axes[idx].bar(strategies, values)
            axes[idx].set_title(label, fontsize=14, fontweight='bold')
            axes[idx].set_ylabel(label)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)

        plt.suptitle(
            f"Performance Metrics Comparison - {results[0].ticker if results else 'N/A'}",
            fontsize=16,
            fontweight='bold'
        )
        plt.tight_layout()

        if save_path:
            self.plotter.save_figure(fig, save_path, dpi=self.dpi)
        else:
            plt.show()

        return fig

    def create_strategy_dashboard(
        self,
        result: BacktestResult,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive dashboard for a single strategy.

        Parameters
        ----------
        result : BacktestResult
            Backtest result
        save_path : Path, optional
            Path to save figure

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 12), dpi=self.dpi)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Equity curve (top, full width)
        ax1 = fig.add_subplot(gs[0, :])
        if result.equity_curve is not None:
            self.plotter.plot_equity_curve(
                result.equity_curve,
                title=f"{result.strategy_name} - {result.ticker}",
                ax=ax1,
                show_drawdown=True
            )

        # 2. Returns distribution
        ax2 = fig.add_subplot(gs[1, 0])
        if result.equity_curve is not None:
            returns = result.equity_curve.pct_change().dropna()
            self.plotter.plot_returns_distribution(
                returns,
                title="Returns Distribution",
                ax=ax2
            )

        # 3. Monthly returns heatmap
        ax3 = fig.add_subplot(gs[1, 1:])
        if result.equity_curve is not None:
            returns = result.equity_curve.pct_change().dropna()
            self.plotter.plot_monthly_returns_heatmap(
                returns,
                title="Monthly Returns",
                ax=ax3
            )

        # 4. Rolling Sharpe
        ax4 = fig.add_subplot(gs[2, 0])
        if result.equity_curve is not None:
            self.plotter.plot_rolling_metric(
                result.equity_curve,
                window=60,
                metric='sharpe',
                title="Rolling Sharpe Ratio (60d)",
                ax=ax4
            )

        # 5. Metrics table
        ax5 = fig.add_subplot(gs[2, 1:])
        ax5.axis('off')

        metrics_data = [
            ['Metric', 'Value'],
            ['Total Return', f"{result.total_return:.2%}"],
            ['Annual Return', f"{result.annual_return:.2%}"],
            ['Sharpe Ratio', f"{result.sharpe_ratio:.2f}"],
            ['Sortino Ratio', f"{result.sortino_ratio:.2f}"],
            ['Max Drawdown', f"{result.max_drawdown:.2%}"],
            ['Volatility', f"{result.volatility:.2%}"],
            ['Win Rate', f"{result.win_rate:.2%}"],
            ['Profit Factor', f"{result.profit_factor:.2f}"],
            ['Total Trades', f"{result.total_trades}"]
        ]

        table = ax5.table(
            cellText=metrics_data,
            cellLoc='left',
            loc='center',
            colWidths=[0.4, 0.4]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.suptitle(
            f"{result.strategy_name} - {result.ticker} Dashboard",
            fontsize=18,
            fontweight='bold'
        )

        if save_path:
            self.plotter.save_figure(fig, save_path, dpi=self.dpi)
        else:
            plt.show()

        return fig

    def generate_comparison_charts(
        self,
        all_results: List[BacktestResult],
        output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Generate overall comparison charts across all tickers and strategies.

        Parameters
        ----------
        all_results : list
            All backtest results
        output_dir : str or Path, optional
            Output directory
        """
        if output_dir is None:
            output_dir = self.save_dir

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Generating comparison charts...")

        # 1. Risk-Return scatter
        self.plot_risk_return_scatter(
            all_results,
            save_path=output_path / f"risk_return_scatter.{self.save_format}"
        )

        # 2. Strategy performance comparison
        self.plot_strategy_performance_overview(
            all_results,
            save_path=output_path / f"strategy_performance_overview.{self.save_format}"
        )

        # 3. Correlation heatmap
        self.plot_correlation_heatmap(
            all_results,
            save_path=output_path / f"correlation_heatmap.{self.save_format}"
        )

        self.logger.info(f"Saved comparison charts to {output_path}")

    def plot_risk_return_scatter(
        self,
        results: List[BacktestResult],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot risk-return scatter for all strategies.

        Parameters
        ----------
        results : list
            List of BacktestResult objects
        save_path : Path, optional
            Path to save figure

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        fig, ax = self.plotter.setup_figure(self.figsize, self.dpi)

        returns = [r.annual_return * 100 for r in results]
        risks = [r.volatility * 100 for r in results]
        labels = [f"{r.ticker}_{r.strategy_name}" for r in results]
        sizes = [abs(r.sharpe_ratio) * 100 for r in results]

        self.plotter.plot_scatter_risk_return(
            returns,
            risks,
            labels,
            sizes=sizes,
            title="Risk-Return Profile (bubble size = Sharpe Ratio)",
            ax=ax
        )

        if save_path:
            self.plotter.save_figure(fig, save_path, dpi=self.dpi)
        else:
            plt.show()

        return fig

    def plot_strategy_performance_overview(
        self,
        results: List[BacktestResult],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot overall strategy performance comparison.

        Parameters
        ----------
        results : list
            List of BacktestResult objects
        save_path : Path, optional
            Path to save figure

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        # Group by strategy and calculate average metrics
        strategy_metrics = {}

        for result in results:
            if result.strategy_name not in strategy_metrics:
                strategy_metrics[result.strategy_name] = {
                    'returns': [],
                    'sharpe': [],
                    'mdd': []
                }

            strategy_metrics[result.strategy_name]['returns'].append(result.total_return)
            strategy_metrics[result.strategy_name]['sharpe'].append(result.sharpe_ratio)
            strategy_metrics[result.strategy_name]['mdd'].append(result.max_drawdown)

        # Calculate averages
        strategies = []
        avg_returns = []
        avg_sharpe = []
        avg_mdd = []

        for strategy, metrics in strategy_metrics.items():
            strategies.append(strategy)
            avg_returns.append(np.mean(metrics['returns']) * 100)
            avg_sharpe.append(np.mean(metrics['sharpe']))
            avg_mdd.append(np.mean(metrics['mdd']) * 100)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=self.dpi)

        # Plot 1: Average Returns
        axes[0].barh(strategies, avg_returns)
        axes[0].set_title('Average Total Return (%)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Return (%)')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Average Sharpe
        axes[1].barh(strategies, avg_sharpe)
        axes[1].set_title('Average Sharpe Ratio', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Sharpe Ratio')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Average Max Drawdown
        axes[2].barh(strategies, avg_mdd)
        axes[2].set_title('Average Max Drawdown (%)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Max Drawdown (%)')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('Strategy Performance Overview (Averaged Across Tickers)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            self.plotter.save_figure(fig, save_path, dpi=self.dpi)
        else:
            plt.show()

        return fig

    def plot_correlation_heatmap(
        self,
        results: List[BacktestResult],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot correlation heatmap of strategy returns.

        Parameters
        ----------
        results : list
            List of BacktestResult objects
        save_path : Path, optional
            Path to save figure

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        # Create DataFrame of returns
        returns_data = {}

        for result in results:
            if result.equity_curve is None:
                continue

            key = f"{result.ticker}_{result.strategy_name}"
            returns_data[key] = result.equity_curve.pct_change().fillna(0)

        if not returns_data:
            self.logger.warning("No data available for correlation heatmap")
            return None

        df = pd.DataFrame(returns_data)
        corr = df.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 12), dpi=self.dpi)

        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax
        )

        ax.set_title('Strategy Returns Correlation Matrix',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()

        if save_path:
            self.plotter.save_figure(fig, save_path, dpi=self.dpi)
        else:
            plt.show()

        return fig
