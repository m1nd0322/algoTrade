"""
Plot utilities and helper functions for visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Union
from pathlib import Path

from src.utils.logger import LoggerMixin


class PlotUtilities(LoggerMixin):
    """
    Utility functions for creating plots and charts.
    """

    def __init__(self, style: str = 'seaborn', color_palette: str = 'Set2'):
        """
        Initialize plot utilities.

        Parameters
        ----------
        style : str
            Matplotlib style
        color_palette : str
            Seaborn color palette
        """
        self.style = style
        self.color_palette = color_palette

        # Set style
        try:
            plt.style.use(style)
        except:
            self.logger.warning(f"Style '{style}' not found, using default")

        # Set color palette
        try:
            sns.set_palette(color_palette)
        except:
            self.logger.warning(f"Color palette '{color_palette}' not found, using default")

    def setup_figure(
        self,
        figsize: Tuple[int, int] = (15, 10),
        dpi: int = 100
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Setup figure and axes.

        Parameters
        ----------
        figsize : tuple
            Figure size
        dpi : int
            DPI

        Returns
        -------
        tuple
            (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        return fig, ax

    def format_axis(
        self,
        ax: plt.Axes,
        title: str,
        xlabel: str,
        ylabel: str,
        grid: bool = True,
        legend: bool = True
    ) -> None:
        """
        Format axis with labels and styling.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes
        title : str
            Plot title
        xlabel : str
            X-axis label
        ylabel : str
            Y-axis label
        grid : bool
            Show grid
        legend : bool
            Show legend
        """
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)

        if grid:
            ax.grid(True, alpha=0.3, linestyle='--')

        if legend:
            ax.legend(loc='best', framealpha=0.9)

    def save_figure(
        self,
        fig: plt.Figure,
        filename: Union[str, Path],
        dpi: int = 300,
        bbox_inches: str = 'tight'
    ) -> None:
        """
        Save figure to file.

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure
        filename : str or Path
            Output filename
        dpi : int
            DPI for saved image
        bbox_inches : str
            Bounding box setting
        """
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        self.logger.info(f"Saved figure to {filepath}")

    def plot_equity_curve(
        self,
        equity: pd.Series,
        title: str = "Equity Curve",
        ax: Optional[plt.Axes] = None,
        show_drawdown: bool = True
    ) -> plt.Axes:
        """
        Plot equity curve with optional drawdown.

        Parameters
        ----------
        equity : pd.Series
            Equity curve
        title : str
            Plot title
        ax : plt.Axes, optional
            Matplotlib axes
        show_drawdown : bool
            Show drawdown on secondary axis

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        if ax is None:
            fig, ax = self.setup_figure()

        # Plot equity
        ax.plot(equity.index, equity.values, linewidth=2, label='Portfolio Value')

        # Format
        self.format_axis(ax, title, 'Date', 'Portfolio Value ($)', grid=True, legend=True)

        # Drawdown on secondary axis
        if show_drawdown:
            ax2 = ax.twinx()
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max

            ax2.fill_between(
                drawdown.index,
                0,
                drawdown.values,
                alpha=0.3,
                color='red',
                label='Drawdown'
            )
            ax2.set_ylabel('Drawdown (%)', fontsize=14)
            ax2.legend(loc='lower right')

        return ax

    def plot_returns_distribution(
        self,
        returns: pd.Series,
        bins: int = 50,
        title: str = "Returns Distribution",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot returns distribution histogram.

        Parameters
        ----------
        returns : pd.Series
            Returns series
        bins : int
            Number of bins
        title : str
            Plot title
        ax : plt.Axes, optional
            Matplotlib axes

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        if ax is None:
            fig, ax = self.setup_figure()

        # Histogram
        ax.hist(returns.values, bins=bins, alpha=0.7, edgecolor='black')

        # Add normal distribution overlay
        mu = returns.mean()
        sigma = returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
             np.exp(-0.5 * (1 / sigma * (x - mu))**2))

        # Scale y to histogram
        y_scaled = y * len(returns) * (returns.max() - returns.min()) / bins
        ax.plot(x, y_scaled, 'r-', linewidth=2, label='Normal Distribution')

        # Add statistics
        textstr = f'Mean: {mu:.4f}\nStd: {sigma:.4f}\nSkew: {returns.skew():.4f}\nKurt: {returns.kurtosis():.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        self.format_axis(ax, title, 'Returns', 'Frequency', grid=True, legend=True)

        return ax

    def plot_rolling_metric(
        self,
        equity: pd.Series,
        window: int = 60,
        metric: str = 'sharpe',
        title: str = "Rolling Metric",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot rolling performance metric.

        Parameters
        ----------
        equity : pd.Series
            Equity curve
        window : int
            Rolling window
        metric : str
            Metric to plot ('sharpe', 'volatility', 'return')
        title : str
            Plot title
        ax : plt.Axes, optional
            Matplotlib axes

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        if ax is None:
            fig, ax = self.setup_figure()

        returns = equity.pct_change().fillna(0)

        if metric == 'sharpe':
            values = (
                np.sqrt(252) *
                returns.rolling(window=window).mean() /
                returns.rolling(window=window).std()
            )
            ylabel = 'Sharpe Ratio'
        elif metric == 'volatility':
            values = returns.rolling(window=window).std() * np.sqrt(252)
            ylabel = 'Annualized Volatility'
        elif metric == 'return':
            values = returns.rolling(window=window).mean() * 252
            ylabel = 'Annualized Return'
        else:
            raise ValueError(f"Unknown metric: {metric}")

        ax.plot(values.index, values.values, linewidth=2)

        self.format_axis(ax, title, 'Date', ylabel, grid=True, legend=False)

        return ax

    def plot_monthly_returns_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns Heatmap",
        ax: Optional[plt.Axes] = None,
        cmap: str = 'RdYlGn'
    ) -> plt.Axes:
        """
        Plot monthly returns as heatmap.

        Parameters
        ----------
        returns : pd.Series
            Daily returns
        title : str
            Plot title
        ax : plt.Axes, optional
            Matplotlib axes
        cmap : str
            Colormap

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        if ax is None:
            fig, ax = self.setup_figure()

        # Resample to monthly
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # Create pivot table (years x months)
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values * 100  # Convert to percentage
        })

        pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')

        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap=cmap,
            center=0,
            cbar_kws={'label': 'Return (%)'},
            ax=ax,
            linewidths=0.5
        )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=14)
        ax.set_ylabel('Year', fontsize=14)

        # Set month labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names)

        return ax

    def plot_comparison_bars(
        self,
        data: pd.DataFrame,
        metric_col: str,
        label_col: str,
        title: str = "Comparison",
        ax: Optional[plt.Axes] = None,
        horizontal: bool = False
    ) -> plt.Axes:
        """
        Plot comparison bar chart.

        Parameters
        ----------
        data : pd.DataFrame
            Data to plot
        metric_col : str
            Column with metric values
        label_col : str
            Column with labels
        title : str
            Plot title
        ax : plt.Axes, optional
            Matplotlib axes
        horizontal : bool
            Horizontal bars

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        if ax is None:
            fig, ax = self.setup_figure()

        if horizontal:
            ax.barh(data[label_col], data[metric_col])
            xlabel, ylabel = metric_col, label_col
        else:
            ax.bar(data[label_col], data[metric_col])
            xlabel, ylabel = label_col, metric_col
            plt.xticks(rotation=45, ha='right')

        self.format_axis(ax, title, xlabel, ylabel, grid=True, legend=False)

        return ax

    def plot_scatter_risk_return(
        self,
        returns: List[float],
        risks: List[float],
        labels: List[str],
        sizes: Optional[List[float]] = None,
        title: str = "Risk-Return Profile",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Plot risk-return scatter plot.

        Parameters
        ----------
        returns : list
            Return values
        risks : list
            Risk values
        labels : list
            Strategy labels
        sizes : list, optional
            Marker sizes
        title : str
            Plot title
        ax : plt.Axes, optional
            Matplotlib axes

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        if ax is None:
            fig, ax = self.setup_figure()

        if sizes is None:
            sizes = [100] * len(returns)

        scatter = ax.scatter(risks, returns, s=sizes, alpha=0.6, edgecolors='black')

        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (risks[i], returns[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )

        self.format_axis(
            ax,
            title,
            'Risk (Volatility)',
            'Return (Annual)',
            grid=True,
            legend=False
        )

        return ax

    def close_all(self) -> None:
        """Close all matplotlib figures."""
        plt.close('all')
