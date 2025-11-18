"""
Relative Momentum Strategy

A strategy that ranks multiple assets by their momentum and invests in the top performers.
Also known as Cross-Sectional Momentum.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from src.strategies.base import BaseStrategy


class RelativeMomentumStrategy(BaseStrategy):
    """
    Relative Momentum Strategy (Cross-Sectional Momentum).

    This strategy compares the momentum of multiple assets and invests in those
    with the highest relative momentum. Unlike absolute momentum which compares
    an asset to its own past, relative momentum ranks assets against each other.

    Strategy Logic:
    - Calculate momentum for all assets in the universe
    - Rank assets by momentum
    - Invest equally in top N assets
    - Rebalance monthly or at specified frequency

    Parameters
    ----------
    data : pd.DataFrame or dict
        Historical price data (single asset or multiple assets)
    params : dict
        Strategy parameters:
        - lookback_period (int): Period to calculate momentum (default: 126 = 6 months)
        - top_n (int): Number of top assets to hold (default: 5)
        - rebalance_frequency (str): 'daily', 'weekly', 'monthly' (default: 'monthly')
    config : dict, optional
        General configuration

    Notes
    -----
    This strategy is designed for multi-asset portfolios. When used with single
    asset data, it behaves similarly to absolute momentum.

    References
    ----------
    Jegadeesh, N., & Titman, S. (1993). "Returns to buying winners and selling losers:
    Implications for stock market efficiency"

    Examples
    --------
    >>> # Multi-asset usage
    >>> data_dict = {'AAPL': aapl_data, 'MSFT': msft_data, 'GOOGL': googl_data}
    >>> params = {'lookback_period': 126, 'top_n': 2}
    >>> strategy = RelativeMomentumStrategy(data_dict, params)
    >>> results = strategy.backtest()
    """

    strategy_name = "Relative Momentum"
    strategy_type = "traditional"
    version = "1.0.0"

    def __init__(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict] = None
    ):
        """Initialize Relative Momentum strategy."""
        default_params = {
            'lookback_period': 126,  # 6 months
            'top_n': 5,
            'rebalance_frequency': 'monthly'
        }

        if params:
            default_params.update(params)

        super().__init__(data, default_params, config)

        # For single asset, store as is
        # For multi-asset, would need to handle differently
        self.is_multi_asset = isinstance(data, dict)

        self.logger.info(
            f"Initialized {self.strategy_name} with lookback={self.params['lookback_period']}, "
            f"top_n={self.params['top_n']}"
        )

    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals based on relative momentum.

        For single asset implementation, we'll use a simplified approach
        that compares current momentum to rolling average momentum.

        Returns
        -------
        pd.Series
            Series of signals (1 = buy, -1 = sell, 0 = hold)
        """
        lookback = self.params['lookback_period']
        rebalance_freq = self.params['rebalance_frequency']

        # Calculate momentum
        momentum = self.data['Close'].pct_change(periods=lookback)

        # Calculate rolling median momentum for comparison
        momentum_median = momentum.rolling(window=lookback * 2, min_periods=lookback).median()

        # Initialize signals
        signals = pd.Series(0, index=self.data.index)

        # Determine rebalance frequency
        if rebalance_freq == 'monthly':
            rebalance_days = 21  # Approximate trading days per month
        elif rebalance_freq == 'weekly':
            rebalance_days = 5
        else:  # daily
            rebalance_days = 1

        previous_position = 0
        days_since_rebalance = 0

        for i in range(lookback * 2, len(self.data)):
            days_since_rebalance += 1

            # Rebalance at specified frequency
            if days_since_rebalance >= rebalance_days:
                current_momentum = momentum.iloc[i]
                median_momentum = momentum_median.iloc[i]

                # Invest if current momentum is above median (relative strength)
                if pd.notna(current_momentum) and pd.notna(median_momentum):
                    if current_momentum > median_momentum:
                        new_position = 1
                    else:
                        new_position = 0

                    # Generate signal on position change
                    if new_position != previous_position:
                        if new_position == 1:
                            signals.iloc[i] = 1
                            self.logger.debug(
                                f"Buy signal at {self.data.index[i]}: "
                                f"momentum={current_momentum:.2%} > "
                                f"median={median_momentum:.2%}"
                            )
                        else:
                            signals.iloc[i] = -1
                            self.logger.debug(
                                f"Sell signal at {self.data.index[i]}: "
                                f"momentum={current_momentum:.2%} < "
                                f"median={median_momentum:.2%}"
                            )

                    previous_position = new_position
                    days_since_rebalance = 0

        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()

        self.logger.info(
            f"Generated {buy_signals} buy and {sell_signals} sell signals "
            f"(rebalancing {rebalance_freq})"
        )

        return signals

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
            Position sizes (1 = invested, 0 = cash)
        """
        positions = pd.Series(0, index=signals.index)
        current_position = 0

        for i in range(len(signals)):
            if signals.iloc[i] == 1:  # Buy signal
                current_position = 1
            elif signals.iloc[i] == -1:  # Sell signal
                current_position = 0

            positions.iloc[i] = current_position

        trades = (positions.diff().fillna(0) != 0).sum()
        self.logger.info(f"Calculated positions: {trades} rebalances")

        return positions

    def get_parameter_grid(self) -> Dict[str, list]:
        """
        Get parameter grid for optimization.

        Returns
        -------
        dict
            Parameter ranges for grid search
        """
        return {
            'lookback_period': [63, 126, 189, 252],  # 3, 6, 9, 12 months
            'rebalance_frequency': ['weekly', 'monthly'],
            'top_n': [3, 5, 7, 10]
        }
