"""
Order management system for live trading.

This module handles order lifecycle, tracking, and execution management
for real-time trading operations.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

from src.utils.logger import LoggerMixin


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"           # 주문 대기
    SUBMITTED = "submitted"       # 주문 접수
    PARTIAL_FILLED = "partial"    # 부분 체결
    FILLED = "filled"             # 전체 체결
    CANCELLED = "cancelled"       # 주문 취소
    REJECTED = "rejected"         # 주문 거부
    FAILED = "failed"             # 주문 실패


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"             # 시장가
    LIMIT = "limit"               # 지정가
    STOP = "stop"                 # 정지가
    STOP_LIMIT = "stop_limit"     # 정지-지정가


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"                   # 매수
    SELL = "sell"                 # 매도


@dataclass
class Order:
    """
    Order object representing a trading order.
    """
    order_id: str                              # Unique order ID
    ticker: str                                # Ticker code
    side: OrderSide                            # Buy or sell
    order_type: OrderType                      # Market, limit, etc.
    quantity: int                              # Number of shares
    price: Optional[float] = None              # Limit price (None for market orders)
    stop_price: Optional[float] = None         # Stop price
    status: OrderStatus = OrderStatus.PENDING  # Current status

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # Execution details
    filled_quantity: int = 0                   # Filled shares
    remaining_quantity: int = 0                # Remaining shares
    avg_fill_price: float = 0.0                # Average fill price
    commission: float = 0.0                    # Commission paid

    # Broker details
    broker_order_no: Optional[str] = None      # Broker's order number
    account_no: Optional[str] = None           # Account number

    # Additional info
    strategy_name: Optional[str] = None        # Strategy that generated order
    notes: Optional[str] = None                # Additional notes

    def __post_init__(self):
        """Initialize remaining quantity."""
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]

    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    def is_cancelled(self) -> bool:
        """Check if order is cancelled."""
        return self.status == OrderStatus.CANCELLED

    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100


class OrderManager(LoggerMixin):
    """
    Manage order lifecycle and execution tracking.

    Features:
    - Order creation and validation
    - Order status tracking
    - Execution monitoring
    - Order history
    - Cancel/modify orders

    Example:
    --------
    >>> manager = OrderManager(api)
    >>> order = manager.create_order(
    ...     ticker="005930",
    ...     side=OrderSide.BUY,
    ...     quantity=10,
    ...     order_type=OrderType.LIMIT,
    ...     price=70000
    ... )
    >>> manager.submit_order(order)
    >>> manager.get_order_status(order.order_id)
    """

    def __init__(self, api, account_no: Optional[str] = None):
        """
        Initialize order manager.

        Parameters
        ----------
        api : KiwoomAPI
            Connected Kiwoom API instance
        account_no : str, optional
            Trading account number
        """
        super().__init__()
        self.api = api
        self.account_no = account_no or self._get_default_account()

        # Order storage
        self.orders: Dict[str, Order] = {}           # All orders
        self.active_orders: Dict[str, Order] = {}    # Active orders only
        self.order_history: List[Order] = []         # Completed orders

        # Mapping: broker order number -> our order ID
        self.broker_order_map: Dict[str, str] = {}

        self.logger.info(f"OrderManager initialized with account: {self.account_no}")

    def _get_default_account(self) -> str:
        """Get default trading account."""
        accounts = self.api.get_account_list()
        if accounts:
            return accounts[0]
        else:
            self.logger.warning("No trading accounts found")
            return ""

    def create_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        strategy_name: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Order:
        """
        Create a new order.

        Parameters
        ----------
        ticker : str
            Ticker code
        side : OrderSide
            BUY or SELL
        quantity : int
            Number of shares
        order_type : OrderType
            Market, limit, etc.
        price : float, optional
            Limit price (required for limit orders)
        stop_price : float, optional
            Stop price (for stop orders)
        strategy_name : str, optional
            Name of strategy generating order
        notes : str, optional
            Additional notes

        Returns
        -------
        Order
            Created order object
        """
        # Validate inputs
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")

        if order_type == OrderType.LIMIT and price is None:
            raise ValueError("Limit orders require a price")

        # Generate unique order ID
        order_id = str(uuid.uuid4())

        # Create order
        order = Order(
            order_id=order_id,
            ticker=ticker,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            account_no=self.account_no,
            strategy_name=strategy_name,
            notes=notes
        )

        # Store order
        self.orders[order_id] = order
        self.active_orders[order_id] = order

        self.logger.info(
            f"Created order {order_id}: {side.value.upper()} {quantity} {ticker} "
            f"@ {price if price else 'MARKET'}"
        )

        return order

    def submit_order(self, order: Order) -> bool:
        """
        Submit order to broker.

        Parameters
        ----------
        order : Order
            Order to submit

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if order.status != OrderStatus.PENDING:
            self.logger.warning(f"Cannot submit order {order.order_id} with status {order.status}")
            return False

        try:
            # Convert to Kiwoom API parameters
            order_type_code = 1 if order.side == OrderSide.BUY else 2  # 1=매수, 2=매도

            # Determine 호가구분
            if order.order_type == OrderType.MARKET:
                hoga_gb = "03"  # 시장가
                order_price = 0
            else:
                hoga_gb = "00"  # 지정가
                order_price = int(order.price) if order.price else 0

            # Submit order
            ret = self.api.send_order(
                rqname=f"order_{order.order_id[:8]}",
                screen_no="0101",
                acc_no=order.account_no,
                order_type=order_type_code,
                code=order.ticker,
                quantity=order.quantity,
                price=order_price,
                hoga_gb=hoga_gb,
                org_order_no=""
            )

            if ret == 0:
                # Update order status
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.now()

                self.logger.info(f"Successfully submitted order {order.order_id}")
                return True
            else:
                # Order failed
                order.status = OrderStatus.FAILED
                self.logger.error(f"Failed to submit order {order.order_id}: error {ret}")
                return False

        except Exception as e:
            order.status = OrderStatus.FAILED
            self.logger.error(f"Exception submitting order {order.order_id}: {e}")
            return False

    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: Optional[int] = None,
        fill_price: Optional[float] = None,
        broker_order_no: Optional[str] = None
    ) -> None:
        """
        Update order status (called by execution callback).

        Parameters
        ----------
        order_id : str
            Order ID
        status : OrderStatus
            New status
        filled_quantity : int, optional
            Filled quantity
        fill_price : float, optional
            Fill price
        broker_order_no : str, optional
            Broker's order number
        """
        if order_id not in self.orders:
            self.logger.warning(f"Order {order_id} not found")
            return

        order = self.orders[order_id]
        old_status = order.status
        order.status = status

        # Update broker order number
        if broker_order_no:
            order.broker_order_no = broker_order_no
            self.broker_order_map[broker_order_no] = order_id

        # Update fill information
        if filled_quantity is not None:
            order.filled_quantity = filled_quantity
            order.remaining_quantity = order.quantity - filled_quantity

            if fill_price is not None:
                # Update average fill price
                order.avg_fill_price = fill_price

        # Update timestamps
        if status == OrderStatus.FILLED:
            order.filled_at = datetime.now()
            self.logger.info(
                f"Order {order_id} FILLED: {order.filled_quantity} @ {order.avg_fill_price}"
            )
        elif status == OrderStatus.CANCELLED:
            order.cancelled_at = datetime.now()
            self.logger.info(f"Order {order_id} CANCELLED")

        # Move to history if completed
        if not order.is_active():
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            self.order_history.append(order)

        self.logger.debug(f"Order {order_id} status: {old_status.value} -> {status.value}")

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.

        Parameters
        ----------
        order_id : str
            Order ID to cancel

        Returns
        -------
        bool
            True if successful
        """
        if order_id not in self.orders:
            self.logger.warning(f"Order {order_id} not found")
            return False

        order = self.orders[order_id]

        if not order.is_active():
            self.logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
            return False

        try:
            # Kiwoom cancel order (order_type 3=매수취소, 4=매도취소)
            cancel_type = 3 if order.side == OrderSide.BUY else 4

            ret = self.api.send_order(
                rqname=f"cancel_{order.order_id[:8]}",
                screen_no="0101",
                acc_no=order.account_no,
                order_type=cancel_type,
                code=order.ticker,
                quantity=order.remaining_quantity,
                price=0,
                hoga_gb="00",
                org_order_no=order.broker_order_no or ""
            )

            if ret == 0:
                self.logger.info(f"Cancel request sent for order {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order {order_id}: error {ret}")
                return False

        except Exception as e:
            self.logger.error(f"Exception cancelling order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_order_by_broker_no(self, broker_order_no: str) -> Optional[Order]:
        """Get order by broker order number."""
        order_id = self.broker_order_map.get(broker_order_no)
        if order_id:
            return self.orders.get(order_id)
        return None

    def get_active_orders(
        self,
        ticker: Optional[str] = None
    ) -> List[Order]:
        """
        Get list of active orders.

        Parameters
        ----------
        ticker : str, optional
            Filter by ticker (all if None)

        Returns
        -------
        list
            List of active orders
        """
        orders = list(self.active_orders.values())

        if ticker:
            orders = [o for o in orders if o.ticker == ticker]

        return orders

    def get_order_history(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Order]:
        """
        Get order history.

        Parameters
        ----------
        ticker : str, optional
            Filter by ticker
        start_date : datetime, optional
            Start date filter
        end_date : datetime, optional
            End date filter

        Returns
        -------
        list
            List of historical orders
        """
        orders = self.order_history.copy()

        if ticker:
            orders = [o for o in orders if o.ticker == ticker]

        if start_date:
            orders = [o for o in orders if o.created_at >= start_date]

        if end_date:
            orders = [o for o in orders if o.created_at <= end_date]

        return orders

    def get_orders_dataframe(
        self,
        include_active: bool = True,
        include_history: bool = True
    ) -> pd.DataFrame:
        """
        Get orders as DataFrame.

        Parameters
        ----------
        include_active : bool
            Include active orders
        include_history : bool
            Include historical orders

        Returns
        -------
        pd.DataFrame
            Orders DataFrame
        """
        orders = []

        if include_active:
            orders.extend(self.active_orders.values())

        if include_history:
            orders.extend(self.order_history)

        if not orders:
            return pd.DataFrame()

        data = []
        for order in orders:
            data.append({
                'order_id': order.order_id,
                'ticker': order.ticker,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'status': order.status.value,
                'quantity': order.quantity,
                'price': order.price,
                'filled_quantity': order.filled_quantity,
                'remaining_quantity': order.remaining_quantity,
                'avg_fill_price': order.avg_fill_price,
                'commission': order.commission,
                'created_at': order.created_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'cancelled_at': order.cancelled_at,
                'strategy_name': order.strategy_name,
                'broker_order_no': order.broker_order_no,
                'fill_pct': order.fill_percentage()
            })

        df = pd.DataFrame(data)

        return df

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get order statistics.

        Returns
        -------
        dict
            Order statistics
        """
        all_orders = list(self.orders.values())

        stats = {
            'total_orders': len(all_orders),
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.order_history),
            'filled_orders': sum(1 for o in all_orders if o.is_filled()),
            'cancelled_orders': sum(1 for o in all_orders if o.is_cancelled()),
            'failed_orders': sum(1 for o in all_orders if o.status == OrderStatus.FAILED),
            'buy_orders': sum(1 for o in all_orders if o.side == OrderSide.BUY),
            'sell_orders': sum(1 for o in all_orders if o.side == OrderSide.SELL),
            'total_volume': sum(o.filled_quantity for o in all_orders),
            'total_commission': sum(o.commission for o in all_orders)
        }

        return stats

    def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        """
        Cancel all active orders.

        Parameters
        ----------
        ticker : str, optional
            Cancel only orders for this ticker (all if None)

        Returns
        -------
        int
            Number of orders cancelled
        """
        orders_to_cancel = self.get_active_orders(ticker)

        cancelled_count = 0
        for order in orders_to_cancel:
            if self.cancel_order(order.order_id):
                cancelled_count += 1

        self.logger.info(f"Cancelled {cancelled_count}/{len(orders_to_cancel)} orders")

        return cancelled_count

    def clear_history(self) -> None:
        """Clear order history."""
        self.order_history.clear()
        self.logger.info("Order history cleared")
