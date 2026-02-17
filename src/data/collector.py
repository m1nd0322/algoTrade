"""
Data collection module for fetching stock market data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pandas_datareader as pdr
except ImportError:
    pdr = None

from src.data.validator import DataValidator
from src.utils.logger import LoggerMixin
from src.utils.helpers import ensure_list


class DataCollector(LoggerMixin):
    """
    Collect stock market data from various sources.
    """

    def __init__(
        self,
        tickers: Union[str, List[str]],
        start_date: str,
        end_date: str,
        config: Optional[Dict] = None
    ):
        """
        Initialize data collector.

        Parameters
        ----------
        tickers : str or list of str
            Ticker symbol(s) to collect
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        config : dict, optional
            Configuration dictionary
        """
        self.tickers = ensure_list(tickers)
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.config = config or {}

        # Get data source configuration
        self.data_sources = self.config.get('data_sources', {})
        self.primary_source = self.data_sources.get('primary', 'yfinance')
        self.secondary_source = self.data_sources.get('secondary', 'pandas-datareader')

        # Initialize validator
        self.validator = DataValidator(config)

        self.logger.info(
            f"Initialized DataCollector for {len(self.tickers)} tickers "
            f"from {start_date} to {end_date}"
        )

    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all tickers.

        Returns
        -------
        dict
            Dictionary mapping ticker to DataFrame
        """
        all_data = {}

        for ticker in self.tickers:
            self.logger.info(f"Fetching data for {ticker}...")

            try:
                data = self._fetch_ticker_data(ticker)

                if data is not None and not data.empty:
                    all_data[ticker] = data
                    self.logger.info(
                        f"✓ Successfully fetched {len(data)} rows for {ticker}"
                    )
                else:
                    self.logger.warning(f"✗ No data fetched for {ticker}")

            except Exception as e:
                self.logger.error(f"✗ Error fetching {ticker}: {e}")
                continue

        self.logger.info(
            f"Fetched data for {len(all_data)}/{len(self.tickers)} tickers"
        )

        return all_data

    def _fetch_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol

        Returns
        -------
        pd.DataFrame or None
            Stock data
        """
        # Try primary source
        data = None

        if self.primary_source == 'yfinance':
            data = self._fetch_yfinance(ticker)
        elif self.primary_source == 'pandas-datareader':
            data = self._fetch_pandas_datareader(ticker)

        # Try secondary source if primary fails
        if data is None or data.empty:
            self.logger.warning(
                f"Primary source failed for {ticker}, trying secondary source..."
            )

            if self.secondary_source == 'yfinance':
                data = self._fetch_yfinance(ticker)
            elif self.secondary_source == 'pandas-datareader':
                data = self._fetch_pandas_datareader(ticker)

        return data

    def _fetch_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch data using yfinance.

        Parameters
        ----------
        ticker : str
            Ticker symbol

        Returns
        -------
        pd.DataFrame or None
            Stock data
        """
        if yf is None:
            self.logger.error("yfinance not installed")
            return None

        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(
                start=self.start_date,
                end=self.end_date,
                auto_adjust=False
            )

            if data.empty:
                return None

            # Standardize column names
            data = data.rename(columns={
                'Adj Close': 'Adj_Close'
            })

            return data

        except Exception as e:
            self.logger.error(f"yfinance error for {ticker}: {e}")
            return None

    def _fetch_pandas_datareader(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch data using pandas-datareader.

        Parameters
        ----------
        ticker : str
            Ticker symbol

        Returns
        -------
        pd.DataFrame or None
            Stock data
        """
        if pdr is None:
            self.logger.error("pandas-datareader not installed")
            return None

        try:
            data = pdr.get_data_yahoo(
                ticker,
                start=self.start_date,
                end=self.end_date
            )

            if data.empty:
                return None

            # Standardize column names
            data = data.rename(columns={
                'Adj Close': 'Adj_Close'
            })

            return data

        except Exception as e:
            self.logger.error(f"pandas-datareader error for {ticker}: {e}")
            return None

    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate collected data.

        Parameters
        ----------
        data : dict
            Dictionary of ticker -> DataFrame

        Returns
        -------
        bool
            True if all data is valid
        """
        all_valid = True

        for ticker, df in data.items():
            is_valid, errors = self.validator.validate_data(df, ticker)

            if not is_valid:
                all_valid = False
                self.logger.error(f"Validation failed for {ticker}:")
                for error in errors:
                    self.logger.error(f"  - {error}")
            else:
                self.logger.info(f"✓ Validation passed for {ticker}")

        return all_valid

    def save_raw_data(
        self,
        data: Dict[str, pd.DataFrame],
        output_dir: Union[str, Path]
    ) -> None:
        """
        Save raw data to CSV files.

        Parameters
        ----------
        data : dict
            Dictionary of ticker -> DataFrame
        output_dir : str or Path
            Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for ticker, df in data.items():
            # Create filename with date range
            start_str = self.start_date.strftime('%Y%m%d')
            end_str = self.end_date.strftime('%Y%m%d')
            filename = f"{ticker}_{start_str}_{end_str}_raw.csv"

            file_path = output_path / filename
            df.to_csv(file_path)

            self.logger.info(f"Saved {ticker} data to {file_path}")

    def load_raw_data(
        self,
        input_dir: Union[str, Path],
        ticker: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from CSV files.

        Parameters
        ----------
        input_dir : str or Path
            Input directory
        ticker : str, optional
            Specific ticker to load (loads all if None)

        Returns
        -------
        dict
            Dictionary of ticker -> DataFrame
        """
        input_path = Path(input_dir)
        data = {}

        if ticker:
            # Load specific ticker
            files = list(input_path.glob(f"{ticker}_*_raw.csv"))
        else:
            # Load all tickers
            files = list(input_path.glob("*_raw.csv"))

        for file_path in files:
            ticker_name = file_path.stem.split('_')[0]
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            data[ticker_name] = df

            self.logger.info(f"Loaded {ticker_name} from {file_path}")

        return data

    def get_latest_data(
        self,
        ticker: str,
        days: int = 30
    ) -> Optional[pd.DataFrame]:
        """
        Get latest N days of data for a ticker.

        Parameters
        ----------
        ticker : str
            Ticker symbol
        days : int
            Number of days to fetch

        Returns
        -------
        pd.DataFrame or None
            Latest stock data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Temporarily set dates
        original_start = self.start_date
        original_end = self.end_date

        self.start_date = start_date
        self.end_date = end_date

        # Fetch data
        data = self._fetch_ticker_data(ticker)

        # Restore original dates
        self.start_date = original_start
        self.end_date = original_end

        return data

    def update_data(
        self,
        existing_data: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Update existing data with latest data.

        Parameters
        ----------
        existing_data : pd.DataFrame
            Existing data
        ticker : str
            Ticker symbol

        Returns
        -------
        pd.DataFrame
            Updated data
        """
        last_date = existing_data.index.max()
        today = datetime.now()

        if (today - last_date).days < 1:
            self.logger.info(f"{ticker} data is already up to date")
            return existing_data

        # Fetch new data
        new_start = last_date + timedelta(days=1)
        original_start = self.start_date

        self.start_date = new_start
        new_data = self._fetch_ticker_data(ticker)
        self.start_date = original_start

        if new_data is None or new_data.empty:
            self.logger.warning(f"No new data available for {ticker}")
            return existing_data

        # Combine data
        updated_data = pd.concat([existing_data, new_data])
        updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
        updated_data = updated_data.sort_index()

        self.logger.info(
            f"Updated {ticker} with {len(new_data)} new rows"
        )

        return updated_data
