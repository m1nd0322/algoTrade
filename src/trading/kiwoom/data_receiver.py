"""
Real-time market data receiver for Kiwoom API.

This module handles real-time price data, order book, and market events
from the Kiwoom Securities trading system.
"""

import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

from src.utils.logger import LoggerMixin


@dataclass
class MarketData:
    """Real-time market data snapshot."""
    ticker: str
    timestamp: datetime
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    volume: int
    bid_price: float
    ask_price: float
    bid_volume: int
    ask_volume: int
    change: float
    change_pct: float


@dataclass
class OrderBook:
    """Order book snapshot (호가)."""
    ticker: str
    timestamp: datetime
    bid_prices: List[float] = field(default_factory=list)  # 매수호가 (10 levels)
    bid_volumes: List[int] = field(default_factory=list)   # 매수잔량
    ask_prices: List[float] = field(default_factory=list)  # 매도호가 (10 levels)
    ask_volumes: List[int] = field(default_factory=list)   # 매도잔량
    total_bid_volume: int = 0
    total_ask_volume: int = 0


class DataReceiver(LoggerMixin):
    """
    Receive and process real-time market data from Kiwoom API.

    Features:
    - Real-time price updates
    - Order book (호가) data
    - Market event notifications
    - Data callbacks for strategies
    - Historical data buffering

    Example:
    --------
    >>> from src.trading.kiwoom import KiwoomAPI, DataReceiver
    >>> api = KiwoomAPI()
    >>> api.comm_connect()
    >>>
    >>> receiver = DataReceiver(api)
    >>> receiver.register_callback('005930', my_strategy_callback)
    >>> receiver.subscribe_realtime(['005930', '000660'])
    """

    # Real-time FID codes (필드 ID)
    FIDS = {
        'current_price': 10,      # 현재가
        'volume': 13,              # 누적거래량
        'open': 16,                # 시가
        'high': 17,                # 고가
        'low': 18,                 # 저가
        'bid_price': 27,           # 매수호가
        'ask_price': 28,           # 매도호가
        'change': 11,              # 전일대비
        'change_pct': 12,          # 등락률
        'bid_volume': 51,          # 매수잔량
        'ask_volume': 52,          # 매도잔량
    }

    def __init__(self, api):
        """
        Initialize data receiver.

        Parameters
        ----------
        api : KiwoomAPI
            Connected Kiwoom API instance
        """
        super().__init__()
        self.api = api

        # Data storage
        self.latest_data: Dict[str, MarketData] = {}
        self.latest_orderbook: Dict[str, OrderBook] = {}

        # Historical buffer (최근 N개 데이터 저장)
        self.data_buffer: Dict[str, List[MarketData]] = defaultdict(list)
        self.buffer_size = 1000

        # Callbacks for strategies
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Subscribed tickers
        self.subscribed_tickers: set = set()

        self.logger.info("DataReceiver initialized")

    def subscribe_realtime(
        self,
        tickers: List[str],
        fids: Optional[List[str]] = None
    ) -> None:
        """
        Subscribe to real-time data for tickers.

        Parameters
        ----------
        tickers : list
            List of ticker codes to subscribe
        fids : list, optional
            List of FID names to receive (all if None)
        """
        if fids is None:
            # Subscribe to all essential fields
            fids = ['current_price', 'volume', 'open', 'high', 'low',
                   'bid_price', 'ask_price', 'change', 'change_pct']

        # Convert FID names to codes
        fid_codes = [str(self.FIDS[f]) for f in fids if f in self.FIDS]
        fid_list = ';'.join(fid_codes)

        # Subscribe each ticker
        for ticker in tickers:
            # Register callback with API
            self.api.register_realtime_callback(ticker, self._on_realtime_data)

            # Subscribe to real-time data
            code_list = ticker
            ret = self.api.set_real_reg(
                screen_no="9999",
                code_list=code_list,
                fid_list=fid_list,
                opt_type="1" if ticker in self.subscribed_tickers else "0"
            )

            if ret == 0:
                self.subscribed_tickers.add(ticker)
                self.logger.info(f"Subscribed to real-time data: {ticker}")
            else:
                self.logger.error(f"Failed to subscribe: {ticker}")

    def unsubscribe(self, ticker: str) -> None:
        """
        Unsubscribe from real-time data.

        Parameters
        ----------
        ticker : str
            Ticker code to unsubscribe
        """
        if ticker in self.subscribed_tickers:
            self.api.set_real_remove("9999", ticker)
            self.subscribed_tickers.remove(ticker)
            self.logger.info(f"Unsubscribed from: {ticker}")

    def _on_realtime_data(self, code: str, real_type: str, real_data: str):
        """
        Handle real-time data callback from Kiwoom API.

        Parameters
        ----------
        code : str
            Ticker code
        real_type : str
            Real-time data type
        real_data : str
            Real-time data string
        """
        try:
            # Parse real-time data
            if real_type == "주식체결":  # Stock execution (tick data)
                market_data = self._parse_tick_data(code)

                # Store latest data
                self.latest_data[code] = market_data

                # Buffer historical data
                self.data_buffer[code].append(market_data)
                if len(self.data_buffer[code]) > self.buffer_size:
                    self.data_buffer[code].pop(0)

                # Trigger callbacks
                self._trigger_callbacks(code, market_data)

            elif real_type == "주식호가잔량":  # Order book
                orderbook = self._parse_orderbook_data(code)
                self.latest_orderbook[code] = orderbook

        except Exception as e:
            self.logger.error(f"Error processing real-time data for {code}: {e}")

    def _parse_tick_data(self, code: str) -> MarketData:
        """
        Parse tick data from Kiwoom API.

        Parameters
        ----------
        code : str
            Ticker code

        Returns
        -------
        MarketData
            Parsed market data
        """
        # Get real-time data using API
        current_price = abs(int(self.api.get_comm_real_data(code, self.FIDS['current_price'])))
        open_price = abs(int(self.api.get_comm_real_data(code, self.FIDS['open'])))
        high_price = abs(int(self.api.get_comm_real_data(code, self.FIDS['high'])))
        low_price = abs(int(self.api.get_comm_real_data(code, self.FIDS['low'])))
        volume = int(self.api.get_comm_real_data(code, self.FIDS['volume']))
        bid_price = abs(int(self.api.get_comm_real_data(code, self.FIDS['bid_price'])))
        ask_price = abs(int(self.api.get_comm_real_data(code, self.FIDS['ask_price'])))
        bid_volume = int(self.api.get_comm_real_data(code, self.FIDS['bid_volume']))
        ask_volume = int(self.api.get_comm_real_data(code, self.FIDS['ask_volume']))
        change = int(self.api.get_comm_real_data(code, self.FIDS['change']))
        change_pct = float(self.api.get_comm_real_data(code, self.FIDS['change_pct']))

        market_data = MarketData(
            ticker=code,
            timestamp=datetime.now(),
            current_price=current_price,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            volume=volume,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            change=change,
            change_pct=change_pct
        )

        return market_data

    def _parse_orderbook_data(self, code: str) -> OrderBook:
        """
        Parse order book data from Kiwoom API.

        Parameters
        ----------
        code : str
            Ticker code

        Returns
        -------
        OrderBook
            Parsed order book
        """
        # Order book FIDs (매도/매수호가 10단계)
        orderbook = OrderBook(ticker=code, timestamp=datetime.now())

        # Parse 10 levels of bid/ask
        for i in range(1, 11):
            # 매도호가 (41-50), 매도잔량 (61-70)
            ask_fid = 40 + i
            ask_vol_fid = 60 + i
            ask_price = abs(int(self.api.get_comm_real_data(code, ask_fid)))
            ask_volume = int(self.api.get_comm_real_data(code, ask_vol_fid))

            orderbook.ask_prices.append(ask_price)
            orderbook.ask_volumes.append(ask_volume)
            orderbook.total_ask_volume += ask_volume

            # 매수호가 (51-60), 매수잔량 (71-80)
            bid_fid = 50 + i
            bid_vol_fid = 70 + i
            bid_price = abs(int(self.api.get_comm_real_data(code, bid_fid)))
            bid_volume = int(self.api.get_comm_real_data(code, bid_vol_fid))

            orderbook.bid_prices.append(bid_price)
            orderbook.bid_volumes.append(bid_volume)
            orderbook.total_bid_volume += bid_volume

        return orderbook

    def register_callback(
        self,
        ticker: str,
        callback: Callable[[str, MarketData], None]
    ) -> None:
        """
        Register callback function for ticker updates.

        Parameters
        ----------
        ticker : str
            Ticker code
        callback : callable
            Callback function(ticker, market_data)
        """
        self.callbacks[ticker].append(callback)
        self.logger.info(f"Registered callback for {ticker}")

    def unregister_callback(self, ticker: str, callback: Callable) -> None:
        """
        Unregister callback function.

        Parameters
        ----------
        ticker : str
            Ticker code
        callback : callable
            Callback function to remove
        """
        if ticker in self.callbacks and callback in self.callbacks[ticker]:
            self.callbacks[ticker].remove(callback)
            self.logger.info(f"Unregistered callback for {ticker}")

    def _trigger_callbacks(self, ticker: str, market_data: MarketData) -> None:
        """
        Trigger all registered callbacks for ticker.

        Parameters
        ----------
        ticker : str
            Ticker code
        market_data : MarketData
            Market data to pass to callbacks
        """
        if ticker in self.callbacks:
            for callback in self.callbacks[ticker]:
                try:
                    callback(ticker, market_data)
                except Exception as e:
                    self.logger.error(f"Error in callback for {ticker}: {e}")

    def get_latest_data(self, ticker: str) -> Optional[MarketData]:
        """
        Get latest market data for ticker.

        Parameters
        ----------
        ticker : str
            Ticker code

        Returns
        -------
        MarketData or None
            Latest market data
        """
        return self.latest_data.get(ticker)

    def get_latest_orderbook(self, ticker: str) -> Optional[OrderBook]:
        """
        Get latest order book for ticker.

        Parameters
        ----------
        ticker : str
            Ticker code

        Returns
        -------
        OrderBook or None
            Latest order book
        """
        return self.latest_orderbook.get(ticker)

    def get_historical_buffer(
        self,
        ticker: str,
        n: Optional[int] = None
    ) -> List[MarketData]:
        """
        Get historical data buffer for ticker.

        Parameters
        ----------
        ticker : str
            Ticker code
        n : int, optional
            Number of recent data points (all if None)

        Returns
        -------
        list
            List of MarketData
        """
        buffer = self.data_buffer.get(ticker, [])

        if n is None:
            return buffer
        else:
            return buffer[-n:]

    def convert_to_dataframe(
        self,
        ticker: str,
        n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Convert historical buffer to DataFrame.

        Parameters
        ----------
        ticker : str
            Ticker code
        n : int, optional
            Number of recent data points

        Returns
        -------
        pd.DataFrame
            Historical market data
        """
        buffer = self.get_historical_buffer(ticker, n)

        if not buffer:
            return pd.DataFrame()

        data = []
        for market_data in buffer:
            data.append({
                'timestamp': market_data.timestamp,
                'open': market_data.open_price,
                'high': market_data.high_price,
                'low': market_data.low_price,
                'close': market_data.current_price,
                'volume': market_data.volume,
                'bid_price': market_data.bid_price,
                'ask_price': market_data.ask_price,
                'bid_volume': market_data.bid_volume,
                'ask_volume': market_data.ask_volume,
                'change': market_data.change,
                'change_pct': market_data.change_pct
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        return df

    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple tickers.

        Parameters
        ----------
        tickers : list
            List of ticker codes

        Returns
        -------
        dict
            Dictionary mapping ticker to current price
        """
        prices = {}

        for ticker in tickers:
            data = self.get_latest_data(ticker)
            if data:
                prices[ticker] = data.current_price

        return prices

    def clear_buffer(self, ticker: Optional[str] = None) -> None:
        """
        Clear historical data buffer.

        Parameters
        ----------
        ticker : str, optional
            Ticker to clear (all if None)
        """
        if ticker:
            if ticker in self.data_buffer:
                self.data_buffer[ticker].clear()
                self.logger.info(f"Cleared buffer for {ticker}")
        else:
            self.data_buffer.clear()
            self.logger.info("Cleared all buffers")

    def get_subscribed_tickers(self) -> List[str]:
        """
        Get list of currently subscribed tickers.

        Returns
        -------
        list
            List of subscribed ticker codes
        """
        return list(self.subscribed_tickers)
