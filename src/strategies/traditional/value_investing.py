"""
Value Investing Strategy

A strategy that invests in undervalued stocks based on fundamental ratios.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.strategies.base import BaseStrategy


class ValueInvestingStrategy(BaseStrategy):
    """
    Value Investing Strategy.

    This strategy identifies undervalued stocks using fundamental metrics
    such as P/E ratio, P/B ratio, and other value indicators. It buys
    stocks trading below their intrinsic value and holds them until
    they reach fair value or rebalancing occurs.

    Strategy Logic:
    - Calculate or use fundamental ratios (P/E, P/B, etc.)
    - Buy when ratios indicate undervaluation
    - Rebalance quarterly or at specified frequency
    - For price-only data, use price-based value proxies

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data (and fundamentals if available)
    params : dict
        Strategy parameters:
        - pe_threshold (float): Maximum P/E ratio (default: 15)
        - pb_threshold (float): Maximum P/B ratio (default: 1.5)
        - rebalance_frequency (str): 'monthly', 'quarterly', 'annually' (default: 'quarterly')
        - min_market_cap (float): Minimum market cap filter (default: 1e9)
    config : dict, optional
        General configuration

    Notes
    -----
    When fundamental data is not available, this implementation uses
    price-based value proxies such as price relative to long-term moving
    average and cyclically adjusted returns.

    Examples
    --------
    >>> params = {'pe_threshold': 15, 'rebalance_frequency': 'quarterly'}
    >>> strategy = ValueInvestingStrategy(data, params)
    >>> results = strategy.backtest()
    """

    strategy_name = "Value Investing"
    strategy_type = "traditional"
    version = "1.0.0"

    def __init__(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict] = None
    ):
        """Initialize Value Investing strategy."""
        default_params = {
            'pe_threshold': 15,
            'pb_threshold': 1.5,
            'rebalance_frequency': 'quarterly',
            'min_market_cap': 1e9,
            'lookback_period': 252  # For price-based value proxy
        }

        if params:
            default_params.update(params)

        super().__init__(data, default_params, config)

        # Check if fundamental data is available
        self.has_fundamentals = 'PE_Ratio' in data.columns or 'PB_Ratio' in data.columns

        self.logger.info(
            f"Initialized {self.strategy_name} strategy "
            f"(fundamentals available: {self.has_fundamentals})"
        )

    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals based on value metrics.

        When fundamental data is not available, uses price-based value proxy:
        - Long-term price mean reversion
        - Cyclically adjusted price ratios
        - Price relative to historical average

        Returns
        -------
        pd.Series
            Series of signals (1 = buy, -1 = sell, 0 = hold)
        """
        rebalance_freq = self.params['rebalance_frequency']
        lookback = self.params['lookback_period']

        # Initialize signals
        signals = pd.Series(0, index=self.data.index)

        if self.has_fundamentals:
            # Use actual fundamental ratios
            signals = self._generate_fundamental_signals()
        else:
            # Use price-based value proxy
            signals = self._generate_price_based_signals(lookback, rebalance_freq)

        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()

        self.logger.info(
            f"Generated {buy_signals} buy and {sell_signals} sell signals"
        )

        return signals

    def _generate_fundamental_signals(self) -> pd.Series:
        """Generate signals using fundamental ratios."""
        pe_threshold = self.params['pe_threshold']
        pb_threshold = self.params['pb_threshold']

        signals = pd.Series(0, index=self.data.index)
        previous_position = 0

        for i in range(len(self.data)):
            # Check if stock is undervalued
            is_undervalued = False

            if 'PE_Ratio' in self.data.columns:
                pe = self.data['PE_Ratio'].iloc[i]
                if pd.notna(pe) and 0 < pe < pe_threshold:
                    is_undervalued = True

            if 'PB_Ratio' in self.data.columns:
                pb = self.data['PB_Ratio'].iloc[i]
                if pd.notna(pb) and 0 < pb < pb_threshold:
                    is_undervalued = True

            # Generate position
            new_position = 1 if is_undervalued else 0

            # Signal on position change
            if new_position != previous_position:
                signals.iloc[i] = 1 if new_position == 1 else -1

            previous_position = new_position

        return signals

    def _generate_price_based_signals(
        self,
        lookback: int,
        rebalance_freq: str
    ) -> pd.Series:
        """
        Generate signals using price-based value proxy.

        Uses the concept that undervalued stocks trade below their long-term
        average price (adjusted for trend).
        """
        signals = pd.Series(0, index=self.data.index)

        # Calculate long-term moving average (proxy for "fair value")
        long_ma = self.data['Close'].rolling(window=lookback).mean()

        # Calculate price-to-moving-average ratio (proxy for P/E)
        price_to_ma = self.data['Close'] / long_ma

        # Calculate rolling percentile of price-to-MA ratio
        # Lower percentiles indicate better value
        value_percentile = price_to_ma.rolling(
            window=lookback * 2, min_periods=lookback
        ).apply(
            lambda x: (x[-1] <= x).sum() / len(x) * 100
        )

        # Determine rebalance frequency
        if rebalance_freq == 'quarterly':
            rebalance_days = 63  # ~3 months
        elif rebalance_freq == 'monthly':
            rebalance_days = 21
        elif rebalance_freq == 'annually':
            rebalance_days = 252
        else:
            rebalance_days = 21

        previous_position = 0
        days_since_rebalance = 0

        for i in range(lookback * 2, len(self.data)):
            days_since_rebalance += 1

            # Rebalance at specified frequency
            if days_since_rebalance >= rebalance_days:
                percentile = value_percentile.iloc[i]

                # Buy if in bottom 30% of historical valuations (undervalued)
                # Sell if in top 70% (fairly valued or overvalued)
                if pd.notna(percentile):
                    if percentile <= 30:  # Undervalued
                        new_position = 1
                    elif percentile >= 70:  # Overvalued
                        new_position = 0
                    else:  # Hold current position
                        new_position = previous_position

                    # Generate signal on position change
                    if new_position != previous_position:
                        if new_position == 1:
                            signals.iloc[i] = 1
                            self.logger.debug(
                                f"Buy signal at {self.data.index[i]}: "
                                f"value percentile={percentile:.1f}%"
                            )
                        else:
                            signals.iloc[i] = -1
                            self.logger.debug(
                                f"Sell signal at {self.data.index[i]}: "
                                f"value percentile={percentile:.1f}%"
                            )

                    previous_position = new_position
                    days_since_rebalance = 0

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

        rebalances = (positions.diff().fillna(0) != 0).sum()
        self.logger.info(f"Calculated positions: {rebalances} rebalances")

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
            'lookback_period': [126, 252, 378, 504],  # 6mo, 1yr, 1.5yr, 2yr
            'rebalance_frequency': ['monthly', 'quarterly'],
            'pe_threshold': [12, 15, 18, 20],
            'pb_threshold': [1.0, 1.5, 2.0]
        }
