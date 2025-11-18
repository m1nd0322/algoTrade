"""
Buy and Hold Strategy

The simplest investment strategy: buy at the beginning and hold until the end.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.strategies.base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy and Hold Strategy.

    This strategy buys the asset at the beginning of the period
    and holds it until the end, representing a passive investment approach.

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data
    params : dict, optional
        Strategy parameters (not used for buy and hold)
    config : dict, optional
        General configuration

    Examples
    --------
    >>> strategy = BuyAndHoldStrategy(data)
    >>> results = strategy.backtest()
    >>> print(f"Total Return: {results['metrics']['total_return']:.2%}")
    """

    strategy_name = "Buy and Hold"
    strategy_type = "traditional"
    version = "1.0.0"

    def __init__(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict] = None
    ):
        """Initialize Buy and Hold strategy."""
        super().__init__(data, params, config)

        # No specific parameters needed for buy and hold
        self.logger.info(f"Initialized {self.strategy_name} strategy")

    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals.

        For buy and hold, we generate a buy signal at the start
        and hold thereafter.

        Returns
        -------
        pd.Series
            Series of signals (1 = buy and hold, 0 = no action)
        """
        signals = pd.Series(0, index=self.data.index)

        # Buy signal at the first available date
        signals.iloc[0] = 1

        self.logger.info("Generated buy and hold signals")

        return signals

    def calculate_positions(self, signals: pd.Series) -> pd.Series:
        """
        Calculate positions from signals.

        For buy and hold, position is always 1 (fully invested) after initial buy.

        Parameters
        ----------
        signals : pd.Series
            Trading signals

        Returns
        -------
        pd.Series
            Position sizes (1 = fully invested)
        """
        # Start with position 0, then buy and hold
        positions = pd.Series(0, index=signals.index)

        # Set position to 1 (fully invested) from first day onwards
        positions[:] = 1

        self.logger.info("Calculated positions: fully invested throughout")

        return positions

    def get_parameter_grid(self) -> Dict[str, list]:
        """
        Get parameter grid for optimization.

        Returns
        -------
        dict
            Empty dict (no parameters to optimize for buy and hold)
        """
        # Buy and hold has no parameters to optimize
        return {}

    def __repr__(self) -> str:
        """String representation."""
        return f"BuyAndHoldStrategy(data_points={len(self.data)})"
