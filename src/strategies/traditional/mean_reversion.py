"""
Mean Reversion Strategy

A strategy based on the principle that prices tend to revert to their mean over time.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy.

    This strategy identifies when prices deviate significantly from their
    moving average and takes positions expecting prices to revert to the mean.

    Strategy Logic:
    - Calculate moving average and standard deviation
    - Generate buy signal when price is N std deviations below mean
    - Generate sell signal when price is N std deviations above mean
    - Exit when price reverts to mean

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data
    params : dict
        Strategy parameters:
        - window (int): Lookback window for mean calculation (default: 20)
        - std_dev (float): Number of standard deviations for threshold (default: 2.0)
        - entry_threshold (float): Entry threshold in std deviations (default: -2.0)
        - exit_threshold (float): Exit threshold in std deviations (default: 0.0)
        - stop_loss (float): Stop loss percentage (default: -0.05)
    config : dict, optional
        General configuration

    Examples
    --------
    >>> params = {'window': 20, 'std_dev': 2.0}
    >>> strategy = MeanReversionStrategy(data, params)
    >>> results = strategy.backtest()
    """

    strategy_name = "Mean Reversion"
    strategy_type = "traditional"
    version = "1.0.0"

    def __init__(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict] = None
    ):
        """Initialize Mean Reversion strategy."""
        # Default parameters
        default_params = {
            'window': 20,
            'std_dev': 2.0,
            'entry_threshold': -2.0,
            'exit_threshold': 0.0,
            'stop_loss': -0.05
        }

        if params:
            default_params.update(params)

        super().__init__(data, default_params, config)

        self.logger.info(
            f"Initialized {self.strategy_name} with window={self.params['window']}, "
            f"std_dev={self.params['std_dev']}"
        )

    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals based on mean reversion.

        Returns
        -------
        pd.Series
            Series of signals (1 = buy, -1 = sell, 0 = hold)
        """
        window = self.params['window']
        std_dev = self.params['std_dev']
        entry_threshold = self.params['entry_threshold']
        exit_threshold = self.params['exit_threshold']

        # Calculate moving average and standard deviation
        ma = self.data['Close'].rolling(window=window).mean()
        std = self.data['Close'].rolling(window=window).std()

        # Calculate z-score (number of std deviations from mean)
        z_score = (self.data['Close'] - ma) / std

        # Initialize signals
        signals = pd.Series(0, index=self.data.index)

        # Track position state
        in_position = False
        entry_price = 0

        for i in range(window, len(self.data)):
            current_z = z_score.iloc[i]
            current_price = self.data['Close'].iloc[i]

            if not in_position:
                # Enter long position when price is significantly below mean
                if current_z <= entry_threshold:
                    signals.iloc[i] = 1
                    in_position = True
                    entry_price = current_price
                    self.logger.debug(
                        f"Buy signal at {self.data.index[i]}: z-score={current_z:.2f}"
                    )

            else:
                # Exit conditions
                # 1. Price reverted to mean
                if current_z >= exit_threshold:
                    signals.iloc[i] = -1
                    in_position = False
                    self.logger.debug(
                        f"Sell signal (mean reversion) at {self.data.index[i]}: "
                        f"z-score={current_z:.2f}"
                    )

                # 2. Stop loss triggered
                elif entry_price > 0:
                    loss = (current_price - entry_price) / entry_price
                    if loss <= self.params['stop_loss']:
                        signals.iloc[i] = -1
                        in_position = False
                        self.logger.debug(
                            f"Sell signal (stop loss) at {self.data.index[i]}: "
                            f"loss={loss:.2%}"
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

        # Track current position
        current_position = 0

        for i in range(len(signals)):
            if signals.iloc[i] == 1:  # Buy signal
                current_position = 1
            elif signals.iloc[i] == -1:  # Sell signal
                current_position = 0

            positions.iloc[i] = current_position

        position_changes = positions.diff().fillna(0)
        trades = (position_changes != 0).sum()

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
            'window': [10, 20, 30, 50],
            'std_dev': [1.5, 2.0, 2.5, 3.0],
            'entry_threshold': [-2.5, -2.0, -1.5],
            'exit_threshold': [-0.5, 0.0, 0.5]
        }
