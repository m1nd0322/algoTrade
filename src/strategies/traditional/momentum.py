"""
Momentum Strategy

A strategy that buys assets with strong recent performance and sells those with weak performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy.

    This strategy identifies and trades assets based on their recent price momentum.
    The core principle is that assets that have performed well recently will
    continue to perform well in the near future.

    Strategy Logic:
    - Calculate momentum over a lookback period
    - Enter long position when momentum exceeds threshold
    - Hold for a specified holding period
    - Exit after holding period or when momentum turns negative

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data
    params : dict
        Strategy parameters:
        - lookback_period (int): Period to calculate momentum (default: 60)
        - holding_period (int): Days to hold position (default: 20)
        - threshold (float): Minimum momentum threshold (default: 0.05)
    config : dict, optional
        General configuration

    Examples
    --------
    >>> params = {'lookback_period': 60, 'holding_period': 20}
    >>> strategy = MomentumStrategy(data, params)
    >>> results = strategy.backtest()
    """

    strategy_name = "Momentum"
    strategy_type = "traditional"
    version = "1.0.0"

    def __init__(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict] = None
    ):
        """Initialize Momentum strategy."""
        default_params = {
            'lookback_period': 60,
            'holding_period': 20,
            'threshold': 0.05
        }

        if params:
            default_params.update(params)

        super().__init__(data, default_params, config)

        self.logger.info(
            f"Initialized {self.strategy_name} with lookback={self.params['lookback_period']}, "
            f"holding={self.params['holding_period']}"
        )

    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals based on momentum.

        Returns
        -------
        pd.Series
            Series of signals (1 = buy, -1 = sell, 0 = hold)
        """
        lookback = self.params['lookback_period']
        holding = self.params['holding_period']
        threshold = self.params['threshold']

        # Calculate momentum (rate of change over lookback period)
        momentum = self.data['Close'].pct_change(periods=lookback)

        # Initialize signals
        signals = pd.Series(0, index=self.data.index)

        # Track position
        position_days = 0
        entry_index = -1

        for i in range(lookback, len(self.data)):
            current_momentum = momentum.iloc[i]

            if position_days == 0:
                # Enter position if momentum exceeds threshold
                if current_momentum > threshold:
                    signals.iloc[i] = 1
                    position_days = 1
                    entry_index = i
                    self.logger.debug(
                        f"Buy signal at {self.data.index[i]}: "
                        f"momentum={current_momentum:.2%}"
                    )

            else:
                position_days += 1

                # Exit conditions
                # 1. Holding period expired
                if position_days >= holding:
                    signals.iloc[i] = -1
                    position_days = 0
                    self.logger.debug(
                        f"Sell signal (holding period) at {self.data.index[i]}"
                    )

                # 2. Momentum turned negative
                elif current_momentum < 0:
                    signals.iloc[i] = -1
                    position_days = 0
                    self.logger.debug(
                        f"Sell signal (negative momentum) at {self.data.index[i]}: "
                        f"momentum={current_momentum:.2%}"
                    )

        buy_signals = (signals == 1).sum()
        sell_signals = (signals == -1).sum()

        self.logger.info(
            f"Generated {buy_signals} buy and {sell_signals} sell signals"
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
            Position sizes (1 = long, 0 = no position)
        """
        positions = pd.Series(0, index=signals.index)
        current_position = 0

        for i in range(len(signals)):
            if signals.iloc[i] == 1:  # Buy
                current_position = 1
            elif signals.iloc[i] == -1:  # Sell
                current_position = 0

            positions.iloc[i] = current_position

        trades = (positions.diff().fillna(0) != 0).sum()
        self.logger.info(f"Calculated positions: {trades} trades")

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
            'lookback_period': [30, 60, 90, 120],
            'holding_period': [10, 20, 30, 40],
            'threshold': [0.02, 0.05, 0.10, 0.15]
        }
