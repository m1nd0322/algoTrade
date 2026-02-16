"""
Live trading orchestrator for Kiwoom Securities.

This module coordinates real-time trading by connecting strategies,
data reception, order execution, and risk management.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, time
import time as time_module
from pathlib import Path

from src.trading.kiwoom.api import KiwoomAPI
from src.trading.kiwoom.data_receiver import DataReceiver, MarketData
from src.trading.order_manager import OrderManager, OrderSide, OrderType, OrderStatus
from src.trading.risk_manager import RiskManager, RiskLimits, Position
from src.strategies.base import BaseStrategy
from src.utils.logger import LoggerMixin


class LiveTrader(LoggerMixin):
    """
    Live trading orchestrator for Korean stock markets.

    Coordinates:
    - Strategy signal generation
    - Real-time data reception
    - Order execution
    - Risk management
    - Portfolio tracking
    - Performance monitoring

    Features:
    - Paper trading mode (no real orders)
    - Automatic position tracking
    - Risk limit enforcement
    - Market hours checking
    - Emergency shutdown

    Example:
    --------
    >>> from src.trading.kiwoom import KiwoomAPI, LiveTrader
    >>> from src.trading.risk_manager import RiskLimits
    >>> from src.strategies.traditional.momentum import MomentumStrategy
    >>>
    >>> # Initialize
    >>> api = KiwoomAPI()
    >>> api.comm_connect()
    >>>
    >>> limits = RiskLimits(
    ...     max_position_size=100000,
    ...     daily_loss_limit=-5000
    ... )
    >>>
    >>> trader = LiveTrader(
    ...     api=api,
    ...     risk_limits=limits,
    ...     initial_capital=1000000,
    ...     paper_trading=True
    ... )
    >>>
    >>> # Add strategy
    >>> strategy = MomentumStrategy(lookback_period=60)
    >>> trader.add_strategy(strategy, tickers=['005930', '000660'])
    >>>
    >>> # Start trading
    >>> trader.start()
    """

    # Market hours (Korean stock market: 09:00 - 15:30)
    MARKET_OPEN = time(9, 0)
    MARKET_CLOSE = time(15, 30)

    def __init__(
        self,
        api: KiwoomAPI,
        risk_limits: RiskLimits,
        initial_capital: float = 1000000.0,
        paper_trading: bool = True,
        update_interval: int = 1,
        log_dir: Optional[str] = None
    ):
        """
        Initialize live trader.

        Parameters
        ----------
        api : KiwoomAPI
            Connected Kiwoom API instance
        risk_limits : RiskLimits
            Risk limit configuration
        initial_capital : float
            Starting capital
        paper_trading : bool
            If True, simulate orders without real execution
        update_interval : int
            Strategy update interval in seconds
        log_dir : str, optional
            Directory for trading logs
        """
        super().__init__()
        self.api = api
        self.paper_trading = paper_trading
        self.update_interval = update_interval

        # Initialize components
        self.data_receiver = DataReceiver(api)
        self.order_manager = OrderManager(api)
        self.risk_manager = RiskManager(risk_limits, initial_capital)

        # Strategy management
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_tickers: Dict[str, List[str]] = {}

        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.portfolio_value = initial_capital

        # Trading state
        self.is_running = False
        self.last_update_time = None

        # Performance tracking
        self.equity_curve: List[Dict] = []
        self.trades_log: List[Dict] = []

        # Logging
        self.log_dir = Path(log_dir) if log_dir else Path("logs/live_trading")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        mode = "PAPER TRADING" if paper_trading else "LIVE TRADING"
        self.logger.info(f"LiveTrader initialized in {mode} mode with capital: {initial_capital:,.0f}")

    def add_strategy(
        self,
        strategy: BaseStrategy,
        tickers: List[str],
        name: Optional[str] = None
    ) -> None:
        """
        Add trading strategy.

        Parameters
        ----------
        strategy : BaseStrategy
            Strategy instance
        tickers : list
            List of tickers to trade with this strategy
        name : str, optional
            Strategy name (uses class name if None)
        """
        strategy_name = name or strategy.__class__.__name__

        self.strategies[strategy_name] = strategy
        self.strategy_tickers[strategy_name] = tickers

        # Subscribe to real-time data
        self.data_receiver.subscribe_realtime(tickers)

        # Register callbacks
        for ticker in tickers:
            self.data_receiver.register_callback(
                ticker,
                lambda t, d, s=strategy_name: self._on_market_data(t, d, s)
            )

        self.logger.info(f"Added strategy '{strategy_name}' for tickers: {tickers}")

    def remove_strategy(self, strategy_name: str) -> None:
        """
        Remove trading strategy.

        Parameters
        ----------
        strategy_name : str
            Name of strategy to remove
        """
        if strategy_name in self.strategies:
            # Unsubscribe from tickers
            tickers = self.strategy_tickers.get(strategy_name, [])
            for ticker in tickers:
                self.data_receiver.unsubscribe(ticker)

            del self.strategies[strategy_name]
            del self.strategy_tickers[strategy_name]

            self.logger.info(f"Removed strategy '{strategy_name}'")

    def _on_market_data(
        self,
        ticker: str,
        market_data: MarketData,
        strategy_name: str
    ) -> None:
        """
        Handle market data updates.

        Parameters
        ----------
        ticker : str
            Ticker code
        market_data : MarketData
            Market data update
        strategy_name : str
            Strategy name
        """
        try:
            # Update position with current price
            if ticker in self.positions:
                self.positions[ticker].current_price = market_data.current_price
                self._update_position_pnl(ticker)

            # Get strategy
            strategy = self.strategies.get(strategy_name)
            if not strategy:
                return

            # Check if it's time to update
            current_time = datetime.now()
            if self.last_update_time:
                elapsed = (current_time - self.last_update_time).total_seconds()
                if elapsed < self.update_interval:
                    return

            # Generate signals
            self._process_strategy_signals(strategy_name, ticker)

            self.last_update_time = current_time

        except Exception as e:
            self.logger.error(f"Error processing market data for {ticker}: {e}")

    def _process_strategy_signals(self, strategy_name: str, ticker: str) -> None:
        """
        Process strategy signals and execute orders.

        Parameters
        ----------
        strategy_name : str
            Strategy name
        ticker : str
            Ticker code
        """
        strategy = self.strategies[strategy_name]

        # Get historical data
        historical_df = self.data_receiver.convert_to_dataframe(ticker, n=500)

        if historical_df.empty or len(historical_df) < 50:
            return

        # Update strategy data
        strategy.data = historical_df

        # Generate signals
        try:
            signals = strategy.generate_signals()

            if signals is None or signals.empty:
                return

            # Get latest signal
            latest_signal = signals.iloc[-1]

            # Execute signal
            self._execute_signal(ticker, latest_signal, strategy_name)

        except Exception as e:
            self.logger.error(f"Error generating signals for {ticker} ({strategy_name}): {e}")

    def _execute_signal(
        self,
        ticker: str,
        signal: float,
        strategy_name: str
    ) -> None:
        """
        Execute trading signal.

        Parameters
        ----------
        ticker : str
            Ticker code
        signal : float
            Signal value (1=buy, -1=sell, 0=hold)
        strategy_name : str
            Strategy name
        """
        # Check if trading is allowed
        allowed, reason = self.risk_manager.is_trading_allowed()
        if not allowed:
            self.logger.warning(f"Trading not allowed: {reason}")
            return

        # Get current price
        market_data = self.data_receiver.get_latest_data(ticker)
        if not market_data:
            return

        current_price = market_data.current_price

        # Check current position
        has_position = ticker in self.positions
        current_quantity = self.positions[ticker].quantity if has_position else 0

        # Buy signal
        if signal > 0 and not has_position:
            # Calculate position size
            quantity = self.risk_manager.get_position_size_recommendation(
                ticker, current_price
            )

            if quantity > 0:
                # Check risk limits
                can_trade, risk_reason = self.risk_manager.check_order(
                    ticker, quantity, current_price, 'buy'
                )

                if can_trade:
                    self._place_order(
                        ticker=ticker,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        price=current_price,
                        order_type=OrderType.MARKET,
                        strategy_name=strategy_name
                    )
                else:
                    self.logger.warning(f"Order rejected by risk manager: {risk_reason}")

        # Sell signal
        elif signal < 0 and has_position:
            # Sell entire position
            quantity = current_quantity

            # Check risk limits
            can_trade, risk_reason = self.risk_manager.check_order(
                ticker, quantity, current_price, 'sell'
            )

            if can_trade:
                self._place_order(
                    ticker=ticker,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    price=current_price,
                    order_type=OrderType.MARKET,
                    strategy_name=strategy_name
                )
            else:
                self.logger.warning(f"Order rejected by risk manager: {risk_reason}")

    def _place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        price: float,
        order_type: OrderType,
        strategy_name: str
    ) -> None:
        """
        Place order.

        Parameters
        ----------
        ticker : str
            Ticker code
        side : OrderSide
            Buy or sell
        quantity : int
            Order quantity
        price : float
            Order price
        order_type : OrderType
            Order type
        strategy_name : str
            Strategy name
        """
        # Create order
        order = self.order_manager.create_order(
            ticker=ticker,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price if order_type != OrderType.MARKET else None,
            strategy_name=strategy_name
        )

        # Submit order (or simulate in paper trading)
        if self.paper_trading:
            # Simulate order fill
            self._simulate_order_fill(order, price)
        else:
            # Submit real order
            success = self.order_manager.submit_order(order)

            if success:
                self.logger.info(f"Order submitted: {order.order_id}")
            else:
                self.logger.error(f"Failed to submit order: {order.order_id}")

    def _simulate_order_fill(self, order, fill_price: float) -> None:
        """
        Simulate order fill for paper trading.

        Parameters
        ----------
        order : Order
            Order to fill
        fill_price : float
            Fill price
        """
        # Update order status
        self.order_manager.update_order_status(
            order.order_id,
            OrderStatus.FILLED,
            filled_quantity=order.quantity,
            fill_price=fill_price
        )

        # Update positions
        if order.side == OrderSide.BUY:
            self._add_position(order.ticker, order.quantity, fill_price)
        else:
            self._remove_position(order.ticker, order.quantity, fill_price)

        # Log trade
        self._log_trade(order, fill_price)

        self.logger.info(
            f"SIMULATED FILL: {order.side.value.upper()} {order.quantity} "
            f"{order.ticker} @ {fill_price:,.0f}"
        )

    def _add_position(self, ticker: str, quantity: int, price: float) -> None:
        """Add or update position."""
        if ticker in self.positions:
            # Update existing position
            pos = self.positions[ticker]
            total_cost = (pos.quantity * pos.avg_entry_price) + (quantity * price)
            pos.quantity += quantity
            pos.avg_entry_price = total_cost / pos.quantity
        else:
            # Create new position
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0
            )

        # Update cash
        self.cash -= quantity * price

        # Update portfolio
        self._update_portfolio_value()

    def _remove_position(self, ticker: str, quantity: int, price: float) -> None:
        """Remove or reduce position."""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]

        if quantity >= pos.quantity:
            # Close entire position
            self.cash += pos.quantity * price
            del self.positions[ticker]
        else:
            # Reduce position
            pos.quantity -= quantity
            self.cash += quantity * price

        # Update portfolio
        self._update_portfolio_value()

    def _update_position_pnl(self, ticker: str) -> None:
        """Update position P&L."""
        if ticker not in self.positions:
            return

        pos = self.positions[ticker]
        pos.market_value = pos.quantity * pos.current_price
        pos.unrealized_pnl = (pos.current_price - pos.avg_entry_price) * pos.quantity
        pos.unrealized_pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price

    def _update_portfolio_value(self) -> None:
        """Update total portfolio value."""
        position_value = sum(p.market_value for p in self.positions.values())
        self.portfolio_value = self.cash + position_value

        # Update risk manager
        self.risk_manager.update_capital(self.portfolio_value)
        self.risk_manager.update_positions(self.positions)

        # Log equity
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position_value': position_value,
            'num_positions': len(self.positions)
        })

    def _log_trade(self, order, fill_price: float) -> None:
        """Log executed trade."""
        self.trades_log.append({
            'timestamp': datetime.now(),
            'order_id': order.order_id,
            'ticker': order.ticker,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': fill_price,
            'value': order.quantity * fill_price,
            'strategy': order.strategy_name
        })

    def start(self) -> None:
        """Start live trading."""
        if self.is_running:
            self.logger.warning("Trader is already running")
            return

        self.is_running = True
        self.logger.info("Live trading started")

        try:
            # Main trading loop
            while self.is_running:
                # Check market hours
                if not self._is_market_open():
                    self.logger.info("Market is closed, waiting...")
                    time_module.sleep(60)
                    continue

                # Reset daily limits if new day
                self.risk_manager.reset_daily_limits()

                # Update portfolio value
                self._update_portfolio_value()

                # Sleep
                time_module.sleep(self.update_interval)

        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop live trading."""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel all active orders
        cancelled = self.order_manager.cancel_all_orders()
        self.logger.info(f"Cancelled {cancelled} active orders")

        # Save state
        self._save_state()

        self.logger.info("Live trading stopped")

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now().time()
        return self.MARKET_OPEN <= now <= self.MARKET_CLOSE

    def _save_state(self) -> None:
        """Save trading state to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save equity curve
        if self.equity_curve:
            df_equity = pd.DataFrame(self.equity_curve)
            df_equity.to_csv(self.log_dir / f"equity_curve_{timestamp}.csv", index=False)

        # Save trades log
        if self.trades_log:
            df_trades = pd.DataFrame(self.trades_log)
            df_trades.to_csv(self.log_dir / f"trades_{timestamp}.csv", index=False)

        # Save orders
        df_orders = self.order_manager.get_orders_dataframe()
        if not df_orders.empty:
            df_orders.to_csv(self.log_dir / f"orders_{timestamp}.csv", index=False)

        # Save risk report
        self.risk_manager.export_risk_report(
            str(self.log_dir / f"risk_report_{timestamp}.json")
        )

        self.logger.info(f"State saved to {self.log_dir}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current trading status.

        Returns
        -------
        dict
            Trading status
        """
        status = {
            'is_running': self.is_running,
            'paper_trading': self.paper_trading,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'num_positions': len(self.positions),
            'num_strategies': len(self.strategies),
            'active_orders': len(self.order_manager.get_active_orders()),
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'last_update': self.last_update_time
        }

        return status

    def print_status(self) -> None:
        """Print current status to console."""
        status = self.get_status()

        print("\n" + "="*60)
        print("LIVE TRADER STATUS")
        print("="*60)
        print(f"Mode: {'PAPER TRADING' if self.paper_trading else 'LIVE TRADING'}")
        print(f"Running: {status['is_running']}")
        print(f"\nPortfolio Value: {status['portfolio_value']:,.0f}")
        print(f"Cash: {status['cash']:,.0f}")
        print(f"Positions: {status['num_positions']}")
        print(f"Active Orders: {status['active_orders']}")
        print(f"\nStrategies: {status['num_strategies']}")

        risk = status['risk_metrics']
        print(f"\nDaily P&L: {risk['daily_pnl']:,.0f} ({risk['daily_pnl_pct']:.2%})")
        print(f"Drawdown: {risk['current_drawdown']:.2%}")
        print(f"Leverage: {risk['leverage']:.2f}x")
        print("="*60 + "\n")
