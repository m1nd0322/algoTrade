#!/usr/bin/env python3
"""
Run live trading with Kiwoom Securities API.

This script sets up and runs the live trading system for Korean stock markets
(KOSPI/KOSDAQ) using the Kiwoom Open API.

Usage:
------
    # Paper trading (recommended for testing)
    python scripts/run_live_trading.py --paper-trading

    # Live trading (real money!)
    python scripts/run_live_trading.py --live

    # With specific strategies
    python scripts/run_live_trading.py --paper-trading --strategies momentum mean_reversion

    # With custom configuration
    python scripts/run_live_trading.py --paper-trading --config config/live_trading_config.yaml

WARNING:
    Live trading involves real money. Always test with paper trading first!
    Use at your own risk.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.kiwoom.api import KiwoomAPI
from src.trading.kiwoom.trader import LiveTrader
from src.trading.risk_manager import RiskLimits
from src.strategies.traditional.momentum import MomentumStrategy
from src.strategies.traditional.mean_reversion import MeanReversionStrategy
from src.strategies.traditional.relative_momentum import RelativeMomentumStrategy
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run live trading with Kiwoom Securities API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Trading mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--paper-trading',
        action='store_true',
        help='Run in paper trading mode (simulated orders)'
    )
    mode_group.add_argument(
        '--live',
        action='store_true',
        help='Run in live trading mode (REAL MONEY!)'
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )

    # Strategies
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=['momentum', 'mean_reversion', 'relative_momentum', 'all'],
        default=['momentum'],
        help='Strategies to run'
    )

    # Tickers
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=['005930', '000660', '035420', '051910', '068270'],  # Samsung, SK Hynix, NAVER, LG Chem, Celltrion
        help='Ticker codes to trade'
    )

    # Capital
    parser.add_argument(
        '--capital',
        type=float,
        default=10000000.0,  # 10 million KRW
        help='Initial capital'
    )

    # Risk limits
    parser.add_argument(
        '--max-position-pct',
        type=float,
        default=0.20,
        help='Maximum position size as % of portfolio'
    )
    parser.add_argument(
        '--daily-loss-limit-pct',
        type=float,
        default=-0.03,
        help='Daily loss limit as % (e.g., -0.03 for -3%%)'
    )
    parser.add_argument(
        '--max-drawdown',
        type=float,
        default=-0.10,
        help='Maximum drawdown as % (e.g., -0.10 for -10%%)'
    )

    # Update interval
    parser.add_argument(
        '--update-interval',
        type=int,
        default=5,
        help='Strategy update interval in seconds'
    )

    # Logging
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/live_trading',
        help='Directory for trading logs'
    )

    return parser.parse_args()


def create_strategy(strategy_name: str):
    """
    Create strategy instance.

    Parameters
    ----------
    strategy_name : str
        Strategy name

    Returns
    -------
    BaseStrategy
        Strategy instance
    """
    if strategy_name == 'momentum':
        return MomentumStrategy(
            lookback_period=60,
            holding_period=20,
            threshold=0.05
        )
    elif strategy_name == 'mean_reversion':
        return MeanReversionStrategy(
            window=20,
            std_dev=2.0,
            entry_threshold=-2.0,
            exit_threshold=0.0,
            stop_loss=-0.05
        )
    elif strategy_name == 'relative_momentum':
        return RelativeMomentumStrategy(
            lookback_period=126,
            top_n=3,
            rebalance_frequency='monthly'
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def main():
    """Main function."""
    args = parse_args()

    # Setup logger
    logger = setup_logger(
        name='live_trading',
        log_dir=args.log_dir,
        log_file=f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level='INFO',
        console_output=True
    )

    # Trading mode
    paper_trading = args.paper_trading
    mode = "PAPER TRADING" if paper_trading else "LIVE TRADING"

    logger.info("="*60)
    logger.info(f"STARTING {mode}")
    logger.info("="*60)

    if not paper_trading:
        logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK! ⚠️")
        response = input("Are you sure you want to proceed? (type 'YES' to confirm): ")
        if response != 'YES':
            logger.info("Live trading cancelled by user")
            return

    # Load configuration
    config = {}
    if args.config:
        config_loader = ConfigLoader(args.config)
        config = config_loader.load()
        logger.info(f"Loaded configuration from {args.config}")

    # Initialize Kiwoom API
    logger.info("Initializing Kiwoom API...")
    try:
        api = KiwoomAPI()
    except Exception as e:
        logger.error(f"Failed to initialize Kiwoom API: {e}")
        logger.error("Please ensure:")
        logger.error("  1. You are running on Windows OS")
        logger.error("  2. Kiwoom OpenAPI+ is installed")
        logger.error("  3. PyQt5 is installed")
        return

    # Connect to Kiwoom
    logger.info("Connecting to Kiwoom server...")
    ret = api.comm_connect()

    if ret != 0:
        logger.error("Failed to connect to Kiwoom server")
        logger.error("Please check your Kiwoom account and try again")
        return

    # Get account info
    account_list = api.get_account_list()
    logger.info(f"Connected successfully. Accounts: {account_list}")

    if not account_list:
        logger.error("No trading accounts found")
        return

    # Setup risk limits
    risk_limits = RiskLimits(
        max_position_pct=args.max_position_pct,
        max_positions=len(args.tickers),
        daily_loss_limit_pct=args.daily_loss_limit_pct,
        max_drawdown=args.max_drawdown,
        enable_circuit_breaker=True
    )

    logger.info("Risk limits:")
    logger.info(f"  Max position: {risk_limits.max_position_pct:.1%}")
    logger.info(f"  Daily loss limit: {risk_limits.daily_loss_limit_pct:.1%}")
    logger.info(f"  Max drawdown: {risk_limits.max_drawdown:.1%}")

    # Initialize live trader
    logger.info(f"Initializing LiveTrader with capital: {args.capital:,.0f} KRW")
    trader = LiveTrader(
        api=api,
        risk_limits=risk_limits,
        initial_capital=args.capital,
        paper_trading=paper_trading,
        update_interval=args.update_interval,
        log_dir=args.log_dir
    )

    # Add strategies
    strategies_to_run = args.strategies
    if 'all' in strategies_to_run:
        strategies_to_run = ['momentum', 'mean_reversion', 'relative_momentum']

    logger.info(f"Adding strategies: {strategies_to_run}")
    for strategy_name in strategies_to_run:
        try:
            strategy = create_strategy(strategy_name)
            trader.add_strategy(strategy, args.tickers, name=strategy_name)
            logger.info(f"  ✓ Added {strategy_name} for {len(args.tickers)} tickers")
        except Exception as e:
            logger.error(f"  ✗ Failed to add {strategy_name}: {e}")

    # Verify tickers
    logger.info("Verifying ticker codes...")
    for ticker in args.tickers:
        ticker_name = api.get_master_code_name(ticker)
        if ticker_name:
            logger.info(f"  {ticker}: {ticker_name}")
        else:
            logger.warning(f"  {ticker}: Unknown ticker")

    # Print initial status
    trader.print_status()

    # Start trading
    logger.info("\nStarting trading loop...")
    logger.info("Press Ctrl+C to stop\n")

    try:
        trader.start()
    except KeyboardInterrupt:
        logger.info("\nShutdown signal received...")
    except Exception as e:
        logger.error(f"Error in trading: {e}", exc_info=True)
    finally:
        # Stop trading
        trader.stop()

        # Print final status
        logger.info("\nFinal Status:")
        trader.print_status()

        # Print statistics
        logger.info("\nOrder Statistics:")
        stats = trader.order_manager.get_statistics()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Disconnect
        api.disconnect()
        logger.info("\nDisconnected from Kiwoom server")

    logger.info("="*60)
    logger.info("TRADING SESSION ENDED")
    logger.info("="*60)


if __name__ == '__main__':
    main()
