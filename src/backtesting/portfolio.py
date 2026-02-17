"""
Portfolio management for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.logger import LoggerMixin


@dataclass
class Trade:
    """
    Represents a completed trade.
    """
    ticker: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: float
    direction: str  # 'long' or 'short'
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def profit_loss(self) -> float:
        """Calculate profit/loss."""
        if self.direction == 'long':
            pnl = (self.exit_price - self.entry_price) * self.shares
        else:
            pnl = (self.entry_price - self.exit_price) * self.shares

        # Subtract costs
        pnl -= (self.commission + self.slippage)

        return pnl

    @property
    def profit_loss_pct(self) -> float:
        """Calculate profit/loss percentage."""
        if self.entry_price == 0:
            return 0.0

        if self.direction == 'long':
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

    @property
    def holding_period(self) -> int:
        """Calculate holding period in days."""
        return (self.exit_date - self.entry_date).days

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'direction': self.direction,
            'commission': self.commission,
            'slippage': self.slippage,
            'profit_loss': self.profit_loss,
            'profit_loss_pct': self.profit_loss_pct,
            'holding_period': self.holding_period
        }


@dataclass
class Position:
    """
    Represents an open position.
    """
    ticker: str
    shares: float
    entry_price: float
    entry_date: datetime
    direction: str = 'long'
    commission_paid: float = 0.0
    slippage_paid: float = 0.0

    def update_price(self, current_price: float) -> None:
        """Update current price for position valuation."""
        self.current_price = current_price

    def market_value(self, current_price: float) -> float:
        """Calculate current market value."""
        return abs(self.shares) * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.direction == 'long':
            return (current_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - current_price) * abs(self.shares)


class Portfolio(LoggerMixin):
    """
    Portfolio manager for backtesting.

    Tracks cash, positions, trades, and portfolio value over time.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005
    ):
        """
        Initialize portfolio.

        Parameters
        ----------
        initial_capital : float
            Starting cash balance
        commission_rate : float
            Commission rate (as decimal)
        slippage_rate : float
            Slippage rate (as decimal)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # Positions and trades
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # Portfolio history
        self.equity_history: List[Tuple[datetime, float]] = []
        self.cash_history: List[Tuple[datetime, float]] = []

        self.logger.info(
            f"Initialized portfolio with ${initial_capital:,.2f}"
        )

    def buy(
        self,
        ticker: str,
        shares: float,
        price: float,
        date: datetime
    ) -> bool:
        """
        Execute buy order.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        shares : float
            Number of shares
        price : float
            Execution price
        date : datetime
            Execution date

        Returns
        -------
        bool
            True if order executed successfully
        """
        # Calculate costs
        gross_cost = shares * price
        commission = gross_cost * self.commission_rate
        slippage = gross_cost * self.slippage_rate
        total_cost = gross_cost + commission + slippage

        # Check if enough cash
        if total_cost > self.cash:
            self.logger.warning(
                f"Insufficient cash for buy order: "
                f"need ${total_cost:,.2f}, have ${self.cash:,.2f}"
            )
            return False

        # Update cash
        self.cash -= total_cost

        # Update or create position
        if ticker in self.positions:
            # Average up existing position
            old_pos = self.positions[ticker]
            total_shares = old_pos.shares + shares
            avg_price = (
                (old_pos.shares * old_pos.entry_price + shares * price) /
                total_shares
            )

            self.positions[ticker] = Position(
                ticker=ticker,
                shares=total_shares,
                entry_price=avg_price,
                entry_date=old_pos.entry_date,
                direction='long',
                commission_paid=old_pos.commission_paid + commission,
                slippage_paid=old_pos.slippage_paid + slippage
            )
        else:
            # New position
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                entry_price=price,
                entry_date=date,
                direction='long',
                commission_paid=commission,
                slippage_paid=slippage
            )

        self.logger.debug(
            f"Buy {shares} shares of {ticker} @ ${price:.2f} "
            f"(cost: ${total_cost:,.2f})"
        )

        return True

    def sell(
        self,
        ticker: str,
        shares: float,
        price: float,
        date: datetime
    ) -> bool:
        """
        Execute sell order.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        shares : float
            Number of shares
        price : float
            Execution price
        date : datetime
            Execution date

        Returns
        -------
        bool
            True if order executed successfully
        """
        # Check if position exists
        if ticker not in self.positions:
            self.logger.warning(
                f"Cannot sell {ticker}: no position exists"
            )
            return False

        position = self.positions[ticker]

        # Check if enough shares
        if shares > position.shares:
            self.logger.warning(
                f"Cannot sell {shares} shares of {ticker}: "
                f"only have {position.shares}"
            )
            return False

        # Calculate proceeds
        gross_proceeds = shares * price
        commission = gross_proceeds * self.commission_rate
        slippage = gross_proceeds * self.slippage_rate
        net_proceeds = gross_proceeds - commission - slippage

        # Update cash
        self.cash += net_proceeds

        # Record trade
        trade = Trade(
            ticker=ticker,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=price,
            shares=shares,
            direction='long',
            commission=position.commission_paid + commission,
            slippage=position.slippage_paid + slippage
        )
        self.trades.append(trade)

        # Update or close position
        if shares == position.shares:
            # Close entire position
            del self.positions[ticker]
            self.logger.debug(
                f"Closed position in {ticker}: P&L ${trade.profit_loss:,.2f}"
            )
        else:
            # Partial close
            position.shares -= shares
            self.logger.debug(
                f"Partial close {shares} shares of {ticker} @ ${price:.2f}"
            )

        return True

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.

        Parameters
        ----------
        prices : dict
            Current prices for all tickers

        Returns
        -------
        float
            Total portfolio value (cash + positions)
        """
        positions_value = sum(
            pos.market_value(prices.get(ticker, pos.entry_price))
            for ticker, pos in self.positions.items()
        )

        total_value = self.cash + positions_value

        return total_value

    def update_history(self, date: datetime, prices: Dict[str, float]) -> None:
        """
        Update portfolio history.

        Parameters
        ----------
        date : datetime
            Current date
        prices : dict
            Current prices
        """
        portfolio_value = self.get_portfolio_value(prices)

        self.equity_history.append((date, portfolio_value))
        self.cash_history.append((date, self.cash))

    def get_equity_curve(self) -> pd.Series:
        """
        Get equity curve as pandas Series.

        Returns
        -------
        pd.Series
            Equity curve indexed by date
        """
        if not self.equity_history:
            return pd.Series(dtype=float)

        dates, values = zip(*self.equity_history)
        return pd.Series(values, index=dates)

    def get_trades_df(self) -> pd.DataFrame:
        """
        Get trades as DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame of all trades
        """
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([trade.to_dict() for trade in self.trades])

    def get_summary(self) -> Dict:
        """
        Get portfolio summary statistics.

        Returns
        -------
        dict
            Summary statistics
        """
        equity_curve = self.get_equity_curve()

        if len(equity_curve) == 0:
            return {}

        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital

        # Trade statistics
        trades_df = self.get_trades_df()

        if len(trades_df) > 0:
            winning_trades = trades_df[trades_df['profit_loss'] > 0]
            losing_trades = trades_df[trades_df['profit_loss'] < 0]

            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0

            summary = {
                'initial_capital': self.initial_capital,
                'final_value': equity_curve.iloc[-1],
                'total_return': total_return,
                'total_trades': len(trades_df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': trades_df['profit_loss'].max(),
                'largest_loss': trades_df['profit_loss'].min(),
                'avg_holding_period': trades_df['holding_period'].mean()
            }
        else:
            summary = {
                'initial_capital': self.initial_capital,
                'final_value': equity_curve.iloc[-1],
                'total_return': total_return,
                'total_trades': 0
            }

        return summary

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        self.cash_history = []

        self.logger.info("Portfolio reset")
