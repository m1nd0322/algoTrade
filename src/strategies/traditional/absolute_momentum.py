"""
Absolute Momentum Strategy

A strategy that invests in an asset only when its momentum is positive (trending upward).
Also known as Time Series Momentum.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.strategies.base import BaseStrategy


class AbsoluteMomentumStrategy(BaseStrategy):
    """
    Absolute Momentum Strategy (Time Series Momentum).

    This strategy compares an asset's recent performance to its own past performance
    and a risk-free rate. It invests only when the asset shows positive absolute momentum.

    Strategy Logic:
    - Calculate return over lookback period
    - Buy if return > risk-free rate (positive momentum)
    - Sell if return < risk-free rate (negative momentum)
    - Rebalance monthly or at specified frequency

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data
    params : dict
        Strategy parameters:
        - lookback_period (int): Period to calculate momentum (default: 252 = 1 year)
        - risk_free_rate (float): Annual risk-free rate (default: 0.02 = 2%)
        - threshold (float): Additional threshold above risk-free rate (default: 0.0)
    config : dict, optional
        General configuration

    References
    ----------
    Antonacci, G. (2014). "Dual Momentum Investing: An Innovative Strategy for
    Higher Returns with Lower Risk"

    Examples
    --------
    >>> params = {'lookback_period': 252, 'risk_free_rate': 0.02}
    >>> strategy = AbsoluteMomentumStrategy(data, params)
    >>> results = strategy.backtest()
    """

    strategy_name = "Absolute Momentum"
    strategy_type = "traditional"
    version = "1.0.0"

    def __init__(
        self,
        data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict] = None
    ):
        """Initialize Absolute Momentum strategy."""
        default_params = {
            'lookback_period': 252,  # 1 year
            'risk_free_rate': 0.02,  # 2% annual
            'threshold': 0.0
        }

        if params:
            default_params.update(params)

        super().__init__(data, default_params, config)

        self.logger.info(
            f"Initialized {self.strategy_name} with lookback={self.params['lookback_period']} days"
        )

    def generate_signals(self) -> pd.Series:
        """
        Generate trading signals based on absolute momentum.

        Returns
        -------
        pd.Series
            Series of signals (1 = buy, -1 = sell, 0 = hold)
        """
        lookback = self.params['lookback_period']
        risk_free_rate = self.params['risk_free_rate']
        threshold = self.params['threshold']

        # Calculate rolling returns over lookback period
        returns = self.data['Close'].pct_change(periods=lookback)

        # Annualize the lookback period return for comparison
        # (returns are already for the full lookback period)
        annualization_factor = 252 / lookback
        annualized_returns = (1 + returns) ** annualization_factor - 1

        # Calculate threshold (risk-free rate + additional threshold)
        momentum_threshold = risk_free_rate + threshold

        # Initialize signals
        signals = pd.Series(0, index=self.data.index)
        previous_position = 0

        for i in range(lookback, len(self.data)):
            current_return = annualized_returns.iloc[i]

            # Determine position based on momentum vs threshold
            if pd.notna(current_return):
                if current_return > momentum_threshold:
                    # Positive absolute momentum - be in the market
                    new_position = 1
                else:
                    # Negative absolute momentum - exit the market
                    new_position = 0

                # Generate signal only on position change
                if new_position != previous_position:
                    if new_position == 1:
                        signals.iloc[i] = 1  # Buy
                        self.logger.debug(
                            f"Buy signal at {self.data.index[i]}: "
                            f"return={current_return:.2%} > threshold={momentum_threshold:.2%}"
                        )
                    else:
                        signals.iloc[i] = -1  # Sell
                        self.logger.debug(
                            f"Sell signal at {self.data.index[i]}: "
                            f"return={current_return:.2%} < threshold={momentum_threshold:.2%}"
                        )

                previous_position = new_position

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

        # Calculate time in market
        time_in_market = (positions > 0).sum() / len(positions)
        trades = (positions.diff().fillna(0) != 0).sum()

        self.logger.info(
            f"Calculated positions: {trades} trades, "
            f"{time_in_market:.1%} time in market"
        )

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
            'risk_free_rate': [0.01, 0.02, 0.03],
            'threshold': [0.0, 0.01, 0.02]
        }
