"""
Risk management system for live trading.

This module implements risk controls, position limits, loss limits,
and portfolio exposure management to protect trading capital.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import LoggerMixin


class RiskViolation(Enum):
    """Risk violation types."""
    POSITION_LIMIT = "position_limit"           # 종목별 포지션 한도 초과
    PORTFOLIO_LIMIT = "portfolio_limit"         # 전체 포트폴리오 한도 초과
    DAILY_LOSS_LIMIT = "daily_loss_limit"       # 일일 손실 한도 초과
    MAX_DRAWDOWN = "max_drawdown"               # 최대 낙폭 초과
    CONCENTRATION = "concentration"             # 집중도 초과
    LEVERAGE = "leverage"                       # 레버리지 초과
    ORDER_SIZE = "order_size"                   # 주문 크기 초과
    VOLATILITY = "volatility"                   # 변동성 초과


@dataclass
class RiskLimits:
    """
    Risk limit configuration.
    """
    # Position limits
    max_position_size: float = 100000.0         # Maximum position value per ticker
    max_position_pct: float = 0.20              # Maximum % of portfolio per position
    max_positions: int = 10                     # Maximum number of positions

    # Portfolio limits
    max_portfolio_value: float = 1000000.0      # Maximum total portfolio value
    max_leverage: float = 1.0                   # Maximum leverage (1.0 = no leverage)

    # Loss limits
    daily_loss_limit: float = -5000.0           # Maximum daily loss ($)
    daily_loss_limit_pct: float = -0.05         # Maximum daily loss (%)
    max_drawdown: float = -0.10                 # Maximum drawdown from peak

    # Order limits
    max_order_size: int = 1000                  # Maximum shares per order
    max_order_value: float = 50000.0            # Maximum order value

    # Concentration limits
    max_sector_exposure: float = 0.30           # Maximum exposure per sector
    min_diversification: int = 3                # Minimum number of positions

    # Volatility limits
    max_portfolio_volatility: float = 0.30      # Maximum annualized volatility

    # Circuit breaker
    enable_circuit_breaker: bool = True         # Enable emergency stop
    circuit_breaker_loss_pct: float = -0.10     # Emergency stop loss %


@dataclass
class Position:
    """Current position information."""
    ticker: str
    quantity: int
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class RiskManager(LoggerMixin):
    """
    Manage trading risk and enforce risk limits.

    Features:
    - Position size validation
    - Loss limit monitoring
    - Drawdown tracking
    - Portfolio exposure control
    - Circuit breaker (emergency stop)

    Example:
    --------
    >>> limits = RiskLimits(
    ...     max_position_size=100000,
    ...     daily_loss_limit=-5000,
    ...     max_drawdown=-0.10
    ... )
    >>> risk_manager = RiskManager(limits, initial_capital=100000)
    >>>
    >>> # Check if order is allowed
    >>> can_trade, reason = risk_manager.check_order(
    ...     ticker="005930",
    ...     quantity=10,
    ...     price=70000,
    ...     side="buy"
    ... )
    """

    def __init__(
        self,
        limits: RiskLimits,
        initial_capital: float = 100000.0
    ):
        """
        Initialize risk manager.

        Parameters
        ----------
        limits : RiskLimits
            Risk limit configuration
        initial_capital : float
            Initial trading capital
        """
        super().__init__()
        self.limits = limits
        self.initial_capital = initial_capital

        # Current state
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.positions: Dict[str, Position] = {}

        # Daily tracking
        self.daily_start_capital = initial_capital
        self.daily_pnl = 0.0
        self.daily_pnl_pct = 0.0
        self.last_reset_date = datetime.now().date()

        # Circuit breaker
        self.circuit_breaker_triggered = False

        # Risk violations log
        self.violations: List[Dict] = []

        self.logger.info(
            f"RiskManager initialized with capital: {initial_capital:,.0f}"
        )

    def reset_daily_limits(self) -> None:
        """Reset daily limits (call at start of each trading day)."""
        today = datetime.now().date()

        if today > self.last_reset_date:
            self.daily_start_capital = self.current_capital
            self.daily_pnl = 0.0
            self.daily_pnl_pct = 0.0
            self.last_reset_date = today
            self.circuit_breaker_triggered = False

            self.logger.info(f"Daily limits reset. Starting capital: {self.daily_start_capital:,.0f}")

    def update_capital(self, new_capital: float) -> None:
        """
        Update current capital.

        Parameters
        ----------
        new_capital : float
            New capital value
        """
        self.current_capital = new_capital

        # Update peak
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

        # Update daily P&L
        self.daily_pnl = new_capital - self.daily_start_capital
        if self.daily_start_capital > 0:
            self.daily_pnl_pct = self.daily_pnl / self.daily_start_capital

        # Check circuit breaker
        if self.limits.enable_circuit_breaker:
            if self.daily_pnl_pct <= self.limits.circuit_breaker_loss_pct:
                self.trigger_circuit_breaker()

    def update_positions(self, positions: Dict[str, Position]) -> None:
        """
        Update current positions.

        Parameters
        ----------
        positions : dict
            Dictionary of current positions
        """
        self.positions = positions

    def trigger_circuit_breaker(self) -> None:
        """Trigger emergency circuit breaker."""
        if not self.circuit_breaker_triggered:
            self.circuit_breaker_triggered = True
            self.logger.critical(
                f"CIRCUIT BREAKER TRIGGERED! Daily loss: {self.daily_pnl_pct:.2%}"
            )
            self._log_violation(RiskViolation.DAILY_LOSS_LIMIT, {
                'daily_pnl': self.daily_pnl,
                'daily_pnl_pct': self.daily_pnl_pct,
                'circuit_breaker': True
            })

    def check_order(
        self,
        ticker: str,
        quantity: int,
        price: float,
        side: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if order is allowed under risk limits.

        Parameters
        ----------
        ticker : str
            Ticker code
        quantity : int
            Order quantity
        price : float
            Order price
        side : str
            'buy' or 'sell'

        Returns
        -------
        tuple
            (allowed: bool, reason: str or None)
        """
        # Check circuit breaker
        if self.circuit_breaker_triggered:
            return False, "Circuit breaker triggered - trading halted"

        order_value = quantity * price

        # Check order size limits
        if quantity > self.limits.max_order_size:
            self._log_violation(RiskViolation.ORDER_SIZE, {
                'ticker': ticker,
                'quantity': quantity,
                'limit': self.limits.max_order_size
            })
            return False, f"Order size exceeds limit ({self.limits.max_order_size})"

        if order_value > self.limits.max_order_value:
            self._log_violation(RiskViolation.ORDER_SIZE, {
                'ticker': ticker,
                'order_value': order_value,
                'limit': self.limits.max_order_value
            })
            return False, f"Order value exceeds limit ({self.limits.max_order_value:,.0f})"

        # Check position limits (for buy orders)
        if side.lower() == 'buy':
            # Check maximum positions
            if ticker not in self.positions:
                if len(self.positions) >= self.limits.max_positions:
                    self._log_violation(RiskViolation.POSITION_LIMIT, {
                        'current_positions': len(self.positions),
                        'limit': self.limits.max_positions
                    })
                    return False, f"Maximum positions reached ({self.limits.max_positions})"

            # Calculate new position size
            current_position_value = 0.0
            if ticker in self.positions:
                current_position_value = self.positions[ticker].market_value

            new_position_value = current_position_value + order_value

            # Check position size limit
            if new_position_value > self.limits.max_position_size:
                self._log_violation(RiskViolation.POSITION_LIMIT, {
                    'ticker': ticker,
                    'new_position_value': new_position_value,
                    'limit': self.limits.max_position_size
                })
                return False, f"Position size exceeds limit ({self.limits.max_position_size:,.0f})"

            # Check position percentage limit
            if self.current_capital > 0:
                position_pct = new_position_value / self.current_capital
                if position_pct > self.limits.max_position_pct:
                    self._log_violation(RiskViolation.POSITION_LIMIT, {
                        'ticker': ticker,
                        'position_pct': position_pct,
                        'limit': self.limits.max_position_pct
                    })
                    return False, f"Position % exceeds limit ({self.limits.max_position_pct:.1%})"

            # Check portfolio limit
            total_portfolio_value = sum(p.market_value for p in self.positions.values())
            new_total_value = total_portfolio_value + order_value

            if new_total_value > self.limits.max_portfolio_value:
                self._log_violation(RiskViolation.PORTFOLIO_LIMIT, {
                    'new_total_value': new_total_value,
                    'limit': self.limits.max_portfolio_value
                })
                return False, f"Portfolio value exceeds limit ({self.limits.max_portfolio_value:,.0f})"

            # Check leverage
            if self.current_capital > 0:
                leverage = new_total_value / self.current_capital
                if leverage > self.limits.max_leverage:
                    self._log_violation(RiskViolation.LEVERAGE, {
                        'leverage': leverage,
                        'limit': self.limits.max_leverage
                    })
                    return False, f"Leverage exceeds limit ({self.limits.max_leverage:.2f})"

        # Check daily loss limit
        if self.daily_pnl <= self.limits.daily_loss_limit:
            self._log_violation(RiskViolation.DAILY_LOSS_LIMIT, {
                'daily_pnl': self.daily_pnl,
                'limit': self.limits.daily_loss_limit
            })
            return False, f"Daily loss limit reached ({self.limits.daily_loss_limit:,.0f})"

        if self.daily_pnl_pct <= self.limits.daily_loss_limit_pct:
            self._log_violation(RiskViolation.DAILY_LOSS_LIMIT, {
                'daily_pnl_pct': self.daily_pnl_pct,
                'limit': self.limits.daily_loss_limit_pct
            })
            return False, f"Daily loss % limit reached ({self.limits.daily_loss_limit_pct:.1%})"

        # Check drawdown
        if self.peak_capital > 0:
            current_drawdown = (self.current_capital - self.peak_capital) / self.peak_capital
            if current_drawdown <= self.limits.max_drawdown:
                self._log_violation(RiskViolation.MAX_DRAWDOWN, {
                    'current_drawdown': current_drawdown,
                    'limit': self.limits.max_drawdown
                })
                return False, f"Maximum drawdown reached ({self.limits.max_drawdown:.1%})"

        # Check minimum diversification (for sell orders)
        if side.lower() == 'sell':
            if ticker in self.positions:
                # Check if selling entire position
                position = self.positions[ticker]
                if quantity >= position.quantity:
                    # Would reduce number of positions
                    remaining_positions = len(self.positions) - 1
                    if remaining_positions > 0 and remaining_positions < self.limits.min_diversification:
                        self._log_violation(RiskViolation.CONCENTRATION, {
                            'remaining_positions': remaining_positions,
                            'limit': self.limits.min_diversification
                        })
                        return False, f"Would violate minimum diversification ({self.limits.min_diversification})"

        # All checks passed
        return True, None

    def get_position_size_recommendation(
        self,
        ticker: str,
        price: float,
        volatility: Optional[float] = None
    ) -> int:
        """
        Get recommended position size based on risk limits.

        Parameters
        ----------
        ticker : str
            Ticker code
        price : float
            Current price
        volatility : float, optional
            Asset volatility (for volatility-based sizing)

        Returns
        -------
        int
            Recommended number of shares
        """
        if price <= 0:
            return 0

        # Calculate maximum shares based on position limits
        max_shares_by_value = int(self.limits.max_position_size / price)
        max_shares_by_pct = int((self.limits.max_position_pct * self.current_capital) / price)

        # Take minimum
        max_shares = min(max_shares_by_value, max_shares_by_pct, self.limits.max_order_size)

        # Adjust for existing position
        if ticker in self.positions:
            current_shares = self.positions[ticker].quantity
            max_shares = max(0, max_shares - current_shares)

        # Volatility-based adjustment (optional)
        if volatility is not None and volatility > 0:
            # Reduce size for high volatility assets
            target_volatility = 0.20  # 20% target volatility
            vol_adjustment = min(1.0, target_volatility / volatility)
            max_shares = int(max_shares * vol_adjustment)

        return max_shares

    def _log_violation(self, violation_type: RiskViolation, details: Dict) -> None:
        """
        Log risk violation.

        Parameters
        ----------
        violation_type : RiskViolation
            Type of violation
        details : dict
            Violation details
        """
        violation = {
            'timestamp': datetime.now(),
            'type': violation_type.value,
            'details': details
        }

        self.violations.append(violation)
        self.logger.warning(f"Risk violation: {violation_type.value} - {details}")

    def get_risk_metrics(self) -> Dict[str, Any]:
        """
        Get current risk metrics.

        Returns
        -------
        dict
            Risk metrics
        """
        # Calculate portfolio metrics
        total_position_value = sum(p.market_value for p in self.positions.values())
        total_unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        leverage = total_position_value / self.current_capital if self.current_capital > 0 else 0
        drawdown = (self.current_capital - self.peak_capital) / self.peak_capital if self.peak_capital > 0 else 0

        # Position concentration
        max_position_pct = 0.0
        if self.current_capital > 0 and self.positions:
            max_position_value = max(p.market_value for p in self.positions.values())
            max_position_pct = max_position_value / self.current_capital

        metrics = {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl_pct,
            'total_position_value': total_position_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'leverage': leverage,
            'current_drawdown': drawdown,
            'num_positions': len(self.positions),
            'max_position_pct': max_position_pct,
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
            'violations_today': len([v for v in self.violations if v['timestamp'].date() == datetime.now().date()])
        }

        return metrics

    def get_violations_dataframe(
        self,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get violations as DataFrame.

        Parameters
        ----------
        start_date : datetime, optional
            Filter violations after this date

        Returns
        -------
        pd.DataFrame
            Violations DataFrame
        """
        violations = self.violations

        if start_date:
            violations = [v for v in violations if v['timestamp'] >= start_date]

        if not violations:
            return pd.DataFrame()

        data = []
        for v in violations:
            row = {
                'timestamp': v['timestamp'],
                'type': v['type']
            }
            row.update(v['details'])
            data.append(row)

        df = pd.DataFrame(data)

        return df

    def export_risk_report(self, output_file: str) -> None:
        """
        Export risk report to file.

        Parameters
        ----------
        output_file : str
            Output file path
        """
        import json

        report = {
            'timestamp': datetime.now().isoformat(),
            'risk_limits': {
                'max_position_size': self.limits.max_position_size,
                'max_position_pct': self.limits.max_position_pct,
                'max_positions': self.limits.max_positions,
                'daily_loss_limit': self.limits.daily_loss_limit,
                'daily_loss_limit_pct': self.limits.daily_loss_limit_pct,
                'max_drawdown': self.limits.max_drawdown,
                'max_order_size': self.limits.max_order_size,
                'max_order_value': self.limits.max_order_value
            },
            'current_metrics': self.get_risk_metrics(),
            'violations': [
                {
                    'timestamp': v['timestamp'].isoformat(),
                    'type': v['type'],
                    'details': v['details']
                }
                for v in self.violations
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Risk report exported to {output_file}")

    def is_trading_allowed(self) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is currently allowed.

        Returns
        -------
        tuple
            (allowed: bool, reason: str or None)
        """
        if self.circuit_breaker_triggered:
            return False, "Circuit breaker triggered"

        if self.daily_pnl_pct <= self.limits.daily_loss_limit_pct:
            return False, "Daily loss limit reached"

        if self.peak_capital > 0:
            drawdown = (self.current_capital - self.peak_capital) / self.peak_capital
            if drawdown <= self.limits.max_drawdown:
                return False, "Maximum drawdown reached"

        return True, None
