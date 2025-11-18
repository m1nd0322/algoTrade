"""
Data preprocessing module for cleaning, feature engineering, and data splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.utils.logger import LoggerMixin
from src.utils.helpers import calculate_returns


class DataPreprocessor(LoggerMixin):
    """
    Preprocess stock market data.
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[pd.DataFrame] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize data preprocessor.

        Parameters
        ----------
        data_path : str or Path, optional
            Path to data file
        data : pd.DataFrame, optional
            Data DataFrame
        config : dict, optional
            Configuration dictionary
        """
        if data_path is not None:
            self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            self.logger.info(f"Loaded data from {data_path}")
        elif data is not None:
            self.data = data.copy()
        else:
            raise ValueError("Either data_path or data must be provided")

        self.config = config or {}
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.features_config = self.config.get('features', {})
        self.normalization_config = self.config.get('normalization', {})
        self.split_config = self.config.get('split', {})

        self.scalers = {}

    def clean_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to clean (uses self.data if None)

        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()

        self.logger.info("Cleaning data...")

        # Handle missing values
        data = self._handle_missing_values(data)

        # Handle outliers
        data = self._handle_outliers(data)

        # Adjust for splits and dividends if configured
        if self.preprocessing_config.get('adjust_splits', True):
            data = self._adjust_for_splits(data)

        # Remove duplicates
        data = data[~data.index.duplicated(keep='first')]

        # Sort by date
        data = data.sort_index()

        self.logger.info(f"Cleaned data: {len(data)} rows")

        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data."""
        method = self.preprocessing_config.get('handle_missing', {}).get('method', 'forward_fill')
        limit = self.preprocessing_config.get('handle_missing', {}).get('limit', 5)

        initial_nulls = data.isnull().sum().sum()

        if method == 'forward_fill':
            data = data.fillna(method='ffill', limit=limit)
        elif method == 'interpolate':
            data = data.interpolate(method='linear', limit=limit)
        elif method == 'drop':
            data = data.dropna()

        # Drop any remaining nulls
        data = data.dropna()

        final_nulls = data.isnull().sum().sum()

        self.logger.info(
            f"Handled missing values: {initial_nulls} -> {final_nulls}"
        )

        return data

    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in data."""
        method = self.preprocessing_config.get('handle_outliers', {}).get('method', 'clip')
        threshold = self.preprocessing_config.get('handle_outliers', {}).get('threshold', 5)

        if method == 'clip':
            # Calculate returns
            returns = data['Close'].pct_change()

            # Calculate z-scores
            z_scores = np.abs((returns - returns.mean()) / returns.std())

            # Clip extreme values
            outlier_mask = z_scores > threshold
            if outlier_mask.any():
                self.logger.info(f"Clipping {outlier_mask.sum()} outliers")

        elif method == 'remove':
            # Remove outlier rows
            returns = data['Close'].pct_change()
            z_scores = np.abs((returns - returns.mean()) / returns.std())
            data = data[z_scores <= threshold]

            self.logger.info(f"Removed outlier rows: {len(data)} rows remaining")

        return data

    def _adjust_for_splits(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adjust prices for stock splits."""
        if 'Adj_Close' in data.columns:
            # Calculate adjustment ratio
            adj_ratio = data['Adj_Close'] / data['Close']

            # Adjust OHLC prices
            data['Open'] = data['Open'] * adj_ratio
            data['High'] = data['High'] * adj_ratio
            data['Low'] = data['Low'] * adj_ratio
            data['Close'] = data['Adj_Close']

            self.logger.info("Adjusted prices for splits/dividends")

        return data

    def engineer_features(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Engineer features from raw data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to process (uses self.data if None)

        Returns
        -------
        pd.DataFrame
            Data with engineered features
        """
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()

        self.logger.info("Engineering features...")

        # Add technical indicators
        data = self._add_technical_indicators(data)

        # Add returns
        data = self._add_returns(data)

        # Add volatility
        data = self._add_volatility(data)

        # Drop initial NaN rows created by indicators
        data = data.dropna()

        self.logger.info(
            f"Engineered features: {len(data.columns)} total columns"
        )

        return data

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        indicators_config = self.features_config.get('technical_indicators', {})

        # Moving averages
        ma_config = indicators_config.get('moving_averages', [])
        for ma in ma_config:
            name = ma['name']
            period = ma['period']
            ma_type = ma.get('type', 'simple')

            if ma_type == 'simple':
                data[name] = data['Close'].rolling(window=period).mean()
            elif ma_type == 'exponential':
                data[name] = data['Close'].ewm(span=period, adjust=False).mean()

            self.logger.debug(f"Added {name}")

        # Momentum indicators
        momentum_config = indicators_config.get('momentum', [])
        for indicator in momentum_config:
            name = indicator['name']

            if name == 'RSI':
                period = indicator.get('period', 14)
                data['RSI'] = self._calculate_rsi(data['Close'], period)
                self.logger.debug("Added RSI")

            elif name == 'MACD':
                fast = indicator.get('fast', 12)
                slow = indicator.get('slow', 26)
                signal = indicator.get('signal', 9)
                macd, signal_line, histogram = self._calculate_macd(
                    data['Close'], fast, slow, signal
                )
                data['MACD'] = macd
                data['MACD_Signal'] = signal_line
                data['MACD_Hist'] = histogram
                self.logger.debug("Added MACD")

            elif name == 'Stochastic':
                k_period = indicator.get('k_period', 14)
                d_period = indicator.get('d_period', 3)
                k, d = self._calculate_stochastic(data, k_period, d_period)
                data['Stoch_K'] = k
                data['Stoch_D'] = d
                self.logger.debug("Added Stochastic")

        # Volatility indicators
        volatility_config = indicators_config.get('volatility', [])
        for indicator in volatility_config:
            name = indicator['name']

            if name == 'BB':
                period = indicator.get('period', 20)
                std_dev = indicator.get('std_dev', 2)
                upper, middle, lower = self._calculate_bollinger_bands(
                    data['Close'], period, std_dev
                )
                data['BB_Upper'] = upper
                data['BB_Middle'] = middle
                data['BB_Lower'] = lower
                self.logger.debug("Added Bollinger Bands")

            elif name == 'ATR':
                period = indicator.get('period', 14)
                data['ATR'] = self._calculate_atr(data, period)
                self.logger.debug("Added ATR")

        return data

    def _add_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add return calculations."""
        returns_config = self.features_config.get('returns', [])

        for ret in returns_config:
            name = ret['name']
            ret_type = ret.get('type', 'simple')

            if ret_type == 'simple':
                data['Daily_Return'] = data['Close'].pct_change()
            elif ret_type == 'logarithmic':
                data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            elif ret_type == 'cumulative':
                data['Cumulative_Return'] = (1 + data['Close'].pct_change()).cumprod() - 1

        return data

    def _add_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility calculations."""
        # Rolling volatility
        data['Volatility_20'] = data['Close'].pct_change().rolling(window=20).std()
        data['Volatility_60'] = data['Close'].pct_change().rolling(window=60).std()

        return data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        return macd, signal_line, histogram

    def _calculate_stochastic(
        self,
        data: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()

        k = 100 * (data['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()

        return k, d

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def normalize_data(
        self,
        data: Optional[pd.DataFrame] = None,
        method: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Normalize data using specified method.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to normalize
        method : str, optional
            Normalization method ('standard', 'minmax', 'robust')
        columns : list, optional
            Columns to normalize (all numeric if None)

        Returns
        -------
        pd.DataFrame
            Normalized data
        """
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()

        if method is None:
            method = self.normalization_config.get('method', 'standard')

        self.logger.info(f"Normalizing data using {method} method...")

        # Select columns to normalize
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Normalize
        data[columns] = scaler.fit_transform(data[columns])

        # Save scaler
        self.scalers[method] = scaler

        self.logger.info(f"Normalized {len(columns)} columns")

        return data

    def split_data(
        self,
        data: Optional[pd.DataFrame] = None,
        ratios: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to split
        ratios : tuple, optional
            (train, validation, test) ratios

        Returns
        -------
        dict
            Dictionary with 'train', 'validation', 'test' DataFrames
        """
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()

        if ratios is None:
            split_config = self.split_config.get('ratios', {})
            train_ratio = split_config.get('train', 0.6)
            val_ratio = split_config.get('validation', 0.2)
            test_ratio = split_config.get('test', 0.2)
            ratios = (train_ratio, val_ratio, test_ratio)

        self.logger.info(f"Splitting data with ratios: {ratios}")

        # Calculate split indices
        n = len(data)
        train_end = int(n * ratios[0])
        val_end = int(n * (ratios[0] + ratios[1]))

        # Split data
        splits = {
            'train': data.iloc[:train_end],
            'validation': data.iloc[train_end:val_end],
            'test': data.iloc[val_end:]
        }

        self.logger.info(
            f"Split sizes - Train: {len(splits['train'])}, "
            f"Val: {len(splits['validation'])}, Test: {len(splits['test'])}"
        )

        return splits

    def save_processed_data(
        self,
        processed_data: pd.DataFrame,
        splits: Dict[str, pd.DataFrame],
        ticker: str,
        output_dir: Union[str, Path]
    ) -> None:
        """
        Save processed data and splits.

        Parameters
        ----------
        processed_data : pd.DataFrame
            Processed data
        splits : dict
            Data splits
        ticker : str
            Ticker symbol
        output_dir : str or Path
            Output directory
        """
        output_path = Path(output_dir)

        # Save processed data
        processed_path = output_path / 'processed'
        processed_path.mkdir(parents=True, exist_ok=True)
        processed_file = processed_path / f"{ticker}_processed.csv"
        processed_data.to_csv(processed_file)
        self.logger.info(f"Saved processed data to {processed_file}")

        # Save splits
        splits_path = output_path / 'splits'
        splits_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            split_file = splits_path / f"{ticker}_{split_name}.csv"
            split_data.to_csv(split_file)
            self.logger.info(f"Saved {split_name} split to {split_file}")
