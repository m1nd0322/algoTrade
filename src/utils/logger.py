"""
Logging utilities for the quant trading system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'quant_trading',
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: str = 'INFO',
    console_output: bool = True
) -> logging.Logger:
    """
    Setup and configure logger with file and console handlers.

    Parameters
    ----------
    name : str
        Logger name
    log_dir : str, optional
        Directory to store log files
    log_file : str, optional
        Log file name (auto-generated if not provided)
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    console_output : bool
        Whether to output to console

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_path / log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get or create logger for the class."""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger
