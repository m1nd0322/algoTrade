"""
Helper functions and utilities.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional
from datetime import datetime, timedelta


def ensure_list(value: Union[str, List]) -> List:
    """
    Ensure value is a list.

    Parameters
    ----------
    value : str or list
        Value to convert

    Returns
    -------
    list
        Value as list
    """
    if isinstance(value, list):
        return value
    return [value]


def validate_date_format(date_str: str) -> bool:
    """
    Validate if string is in YYYY-MM-DD format.

    Parameters
    ----------
    date_str : str
        Date string to validate

    Returns
    -------
    bool
        True if valid format
    """
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.

    Parameters
    ----------
    prices : pd.Series
        Price series
    method : str
        Return calculation method ('simple', 'log')

    Returns
    -------
    pd.Series
        Returns series
    """
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_volatility(returns: pd.Series, window: int = 252) -> float:
    """
    Calculate annualized volatility.

    Parameters
    ----------
    returns : pd.Series
        Returns series
    window : int
        Annualization factor (252 for daily data)

    Returns
    -------
    float
        Annualized volatility
    """
    return returns.std() * np.sqrt(window)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Returns series
    risk_free_rate : float
        Annual risk-free rate
    periods : int
        Number of periods per year (252 for daily)

    Returns
    -------
    float
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown.

    Parameters
    ----------
    equity_curve : pd.Series
        Equity curve series

    Returns
    -------
    tuple
        (max_drawdown, peak_date, trough_date)
    """
    # Calculate running maximum
    running_max = equity_curve.expanding().max()

    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max

    # Find maximum drawdown
    max_dd = drawdown.min()

    # Find peak and trough dates
    trough_date = drawdown.idxmin()
    peak_date = equity_curve[:trough_date].idxmax()

    return max_dd, peak_date, trough_date


def resample_data(
    data: pd.DataFrame,
    freq: str,
    agg_dict: Optional[dict] = None
) -> pd.DataFrame:
    """
    Resample data to different frequency.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with datetime index
    freq : str
        Target frequency ('D', 'W', 'M', 'Q', 'Y')
    agg_dict : dict, optional
        Aggregation dictionary for each column

    Returns
    -------
    pd.DataFrame
        Resampled data
    """
    if agg_dict is None:
        # Default OHLCV aggregation
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }

    # Filter columns that exist in data
    agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}

    return data.resample(freq).agg(agg_dict)


def create_time_series_split(
    data: pd.DataFrame,
    n_splits: int = 5,
    test_size: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series cross-validation splits.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    n_splits : int
        Number of splits
    test_size : int, optional
        Size of test set (if None, calculated based on n_splits)

    Returns
    -------
    list
        List of (train_idx, test_idx) tuples
    """
    n_samples = len(data)

    if test_size is None:
        test_size = n_samples // (n_splits + 1)

    indices = np.arange(n_samples)
    splits = []

    for i in range(n_splits):
        test_start = (i + 1) * test_size
        test_end = test_start + test_size

        if test_end > n_samples:
            break

        train_idx = indices[:test_start]
        test_idx = indices[test_start:test_end]

        splits.append((train_idx, test_idx))

    return splits


def normalize_data(
    data: Union[pd.DataFrame, pd.Series],
    method: str = 'standard'
) -> Union[pd.DataFrame, pd.Series]:
    """
    Normalize data using specified method.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Data to normalize
    method : str
        Normalization method ('standard', 'minmax', 'robust')

    Returns
    -------
    pd.DataFrame or pd.Series
        Normalized data
    """
    if method == 'standard':
        return (data - data.mean()) / data.std()
    elif method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    elif method == 'robust':
        median = data.median()
        q75, q25 = data.quantile([0.75, 0.25])
        iqr = q75 - q25
        return (data - median) / iqr
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix of returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns data

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    return returns.corr()


def detect_outliers(
    data: pd.Series,
    method: str = 'iqr',
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers in data.

    Parameters
    ----------
    data : pd.Series
        Input data
    method : str
        Detection method ('iqr', 'zscore')
    threshold : float
        Threshold for outlier detection

    Returns
    -------
    pd.Series
        Boolean series indicating outliers
    """
    if method == 'iqr':
        q75, q25 = data.quantile([0.75, 0.25])
        iqr = q75 - q25
        lower_bound = q25 - threshold * iqr
        upper_bound = q75 + threshold * iqr
        return (data < lower_bound) | (data > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown method: {method}")


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.

    Parameters
    ----------
    value : float
        Value to format
    decimals : int
        Number of decimal places

    Returns
    -------
    str
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, symbol: str = '$', decimals: int = 2) -> str:
    """
    Format value as currency string.

    Parameters
    ----------
    value : float
        Value to format
    symbol : str
        Currency symbol
    decimals : int
        Number of decimal places

    Returns
    -------
    str
        Formatted currency string
    """
    return f"{symbol}{value:,.{decimals}f}"
