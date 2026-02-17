"""
Data validation utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.utils.logger import LoggerMixin


class DataValidator(LoggerMixin):
    """
    Validate data quality and integrity.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data validator.

        Parameters
        ----------
        config : dict, optional
            Validation configuration
        """
        self.config = config or {}
        self.validation_config = self.config.get('validation', {})

    def validate_data(
        self,
        data: pd.DataFrame,
        ticker: str
    ) -> Tuple[bool, List[str]]:
        """
        Perform comprehensive data validation.

        Parameters
        ----------
        data : pd.DataFrame
            Data to validate
        ticker : str
            Ticker symbol

        Returns
        -------
        tuple
            (is_valid, error_messages)
        """
        errors = []

        # Check 1: Minimum data points
        min_points = self.validation_config.get('min_data_points', 252)
        if len(data) < min_points:
            errors.append(
                f"{ticker}: Insufficient data points ({len(data)} < {min_points})"
            )

        # Check 2: Missing data
        max_missing_ratio = self.validation_config.get('max_missing_ratio', 0.05)
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > max_missing_ratio:
            errors.append(
                f"{ticker}: Too many missing values ({missing_ratio:.2%} > {max_missing_ratio:.2%})"
            )

        # Check 3: Duplicates
        if self.validation_config.get('check_duplicates', True):
            if data.index.duplicated().any():
                n_duplicates = data.index.duplicated().sum()
                errors.append(
                    f"{ticker}: {n_duplicates} duplicate dates found"
                )

        # Check 4: Required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(
                f"{ticker}: Missing required columns: {missing_columns}"
            )

        # Check 5: Data types
        if not errors:  # Only if columns exist
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    errors.append(
                        f"{ticker}: Column '{col}' is not numeric"
                    )

        # Check 6: Logical consistency (High >= Low, etc.)
        if not errors:
            if (data['High'] < data['Low']).any():
                errors.append(
                    f"{ticker}: High < Low in some rows"
                )
            if (data['High'] < data['Open']).any() or (data['High'] < data['Close']).any():
                errors.append(
                    f"{ticker}: High < Open or Close in some rows"
                )
            if (data['Low'] > data['Open']).any() or (data['Low'] > data['Close']).any():
                errors.append(
                    f"{ticker}: Low > Open or Close in some rows"
                )

        # Check 7: Negative values
        if not errors:
            negative_prices = (data[['Open', 'High', 'Low', 'Close']] < 0).any()
            if negative_prices.any():
                errors.append(
                    f"{ticker}: Negative prices found"
                )

            if (data['Volume'] < 0).any():
                errors.append(
                    f"{ticker}: Negative volume found"
                )

        # Check 8: Zero volume
        if not errors:
            zero_volume_ratio = (data['Volume'] == 0).sum() / len(data)
            if zero_volume_ratio > 0.1:  # More than 10% zero volume
                errors.append(
                    f"{ticker}: Too many zero volume days ({zero_volume_ratio:.2%})"
                )

        # Check 9: Outliers
        if self.validation_config.get('check_outliers', True) and not errors:
            outliers = self._detect_outliers(data)
            if outliers.sum() > len(data) * 0.05:  # More than 5% outliers
                self.logger.warning(
                    f"{ticker}: High number of outliers detected ({outliers.sum()})"
                )

        is_valid = len(errors) == 0
        return is_valid, errors

    def _detect_outliers(self, data: pd.DataFrame, threshold: float = 5) -> pd.Series:
        """
        Detect outliers using z-score method.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        threshold : float
            Z-score threshold

        Returns
        -------
        pd.Series
            Boolean series indicating rows with outliers
        """
        # Calculate returns
        returns = data['Close'].pct_change()

        # Calculate z-scores
        z_scores = np.abs((returns - returns.mean()) / returns.std())

        # Identify outliers
        outliers = z_scores > threshold

        return outliers

    def check_data_continuity(
        self,
        data: pd.DataFrame,
        max_gap_days: int = 7
    ) -> Tuple[bool, List[Tuple]]:
        """
        Check for gaps in data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data with datetime index
        max_gap_days : int
            Maximum allowed gap in days

        Returns
        -------
        tuple
            (is_continuous, list of (start_date, end_date, gap_days))
        """
        gaps = []

        if not isinstance(data.index, pd.DatetimeIndex):
            return False, [("Index is not DatetimeIndex", "", 0)]

        # Calculate differences between consecutive dates
        date_diffs = data.index.to_series().diff()

        # Find gaps larger than threshold
        large_gaps = date_diffs > pd.Timedelta(days=max_gap_days)

        if large_gaps.any():
            gap_indices = np.where(large_gaps)[0]
            for idx in gap_indices:
                start_date = data.index[idx - 1]
                end_date = data.index[idx]
                gap_days = (end_date - start_date).days
                gaps.append((start_date, end_date, gap_days))

        is_continuous = len(gaps) == 0
        return is_continuous, gaps

    def validate_date_range(
        self,
        data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> Tuple[bool, str]:
        """
        Validate if data covers the requested date range.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        start_date : str
            Expected start date (YYYY-MM-DD)
        end_date : str
            Expected end date (YYYY-MM-DD)

        Returns
        -------
        tuple
            (is_valid, error_message)
        """
        data_start = data.index.min()
        data_end = data.index.max()

        expected_start = pd.to_datetime(start_date)
        expected_end = pd.to_datetime(end_date)

        # Allow some tolerance (e.g., weekends, holidays)
        tolerance = pd.Timedelta(days=5)

        if data_start > expected_start + tolerance:
            return False, f"Data starts too late: {data_start} > {expected_start}"

        if data_end < expected_end - tolerance:
            return False, f"Data ends too early: {data_end} < {expected_end}"

        return True, ""

    def generate_validation_report(
        self,
        data: pd.DataFrame,
        ticker: str
    ) -> Dict:
        """
        Generate comprehensive validation report.

        Parameters
        ----------
        data : pd.DataFrame
            Data to validate
        ticker : str
            Ticker symbol

        Returns
        -------
        dict
            Validation report
        """
        report = {
            'ticker': ticker,
            'data_points': len(data),
            'date_range': f"{data.index.min()} to {data.index.max()}",
            'missing_values': {},
            'outliers': 0,
            'zero_volume_days': 0,
            'gaps': [],
            'is_valid': False,
            'errors': []
        }

        # Missing values
        for col in data.columns:
            missing = data[col].isnull().sum()
            if missing > 0:
                report['missing_values'][col] = missing

        # Outliers
        outliers = self._detect_outliers(data)
        report['outliers'] = outliers.sum()

        # Zero volume
        report['zero_volume_days'] = (data['Volume'] == 0).sum()

        # Gaps
        _, gaps = self.check_data_continuity(data)
        report['gaps'] = gaps

        # Overall validation
        is_valid, errors = self.validate_data(data, ticker)
        report['is_valid'] = is_valid
        report['errors'] = errors

        return report
