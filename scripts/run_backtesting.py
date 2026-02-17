#!/usr/bin/env python3
"""
Backtesting Script

Runs backtesting for all configured strategies on specified tickers.
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.engine import BacktestEngine
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtesting for trading strategies"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/splits',
        help='Directory containing split data'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['all'],
        help='Strategies to test (default: all)'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Tickers to test (default: all in data dir)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/backtests',
        help='Output directory for results'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/backtest_config.yaml',
        help='Path to backtest configuration file'
    )
    parser.add_argument(
        '--strategy-config',
        type=str,
        default='config/strategy_config.yaml',
        help='Path to strategy configuration file'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run backtests in parallel'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()


def get_strategy_list(strategy_names: List[str], strategy_config: dict) -> List[str]:
    """Get list of strategies to run."""
    if 'all' in strategy_names:
        strategies = []
        for category in ['traditional', 'machine_learning', 'deep_learning']:
            if category in strategy_config:
                for name, config in strategy_config[category].items():
                    if config.get('enabled', True):
                        strategies.append(f"{category}.{name}")
        return strategies
    return strategy_names


def main():
    """Main execution function."""
    args = parse_args()

    # Setup logger
    logger = setup_logger(
        name='backtesting',
        log_dir='results/logs',
        level='DEBUG' if args.verbose else 'INFO'
    )

    logger.info("Starting backtesting...")

    # Load configurations
    backtest_config = ConfigLoader(args.config).load()
    strategy_config = ConfigLoader(args.strategy_config).load()

    # Get list of strategies to run
    strategies = get_strategy_list(args.strategies, strategy_config)
    logger.info(f"Running {len(strategies)} strategies")

    # Get list of tickers
    data_path = Path(args.data_dir)
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = sorted(set([f.stem.split('_')[0] for f in data_path.glob("*_test.csv")]))

    logger.info(f"Testing on {len(tickers)} tickers: {tickers}")

    # Initialize backtest engine
    engine = BacktestEngine(config=backtest_config)

    # Run backtests
    results = []
    total_tests = len(strategies) * len(tickers)
    completed = 0

    logger.info(f"\nRunning {total_tests} backtests...")
    logger.info("="*50)

    for strategy_name in strategies:
        for ticker in tickers:
            try:
                logger.info(f"[{completed+1}/{total_tests}] Testing {strategy_name} on {ticker}...")

                # Load data
                train_data = data_path / f"{ticker}_train.csv"
                test_data = data_path / f"{ticker}_test.csv"

                if not train_data.exists() or not test_data.exists():
                    logger.warning(f"Data not found for {ticker}, skipping...")
                    continue

                # Run backtest
                result = engine.run(
                    strategy_name=strategy_name,
                    ticker=ticker,
                    train_data=train_data,
                    test_data=test_data,
                    strategy_config=strategy_config
                )

                results.append(result)

                # Log key metrics
                logger.info(f"  ✓ Total Return: {result.total_return:.2%}")
                logger.info(f"  ✓ Sharpe Ratio: {result.sharpe_ratio:.2f}")
                logger.info(f"  ✓ Max Drawdown: {result.max_drawdown:.2%}")

                completed += 1

            except Exception as e:
                logger.error(f"Error testing {strategy_name} on {ticker}: {e}", exc_info=True)
                continue

    # Save results
    logger.info("\n" + "="*50)
    logger.info("Saving results...")
    engine.save_results(results, args.output_dir)

    logger.info("\n" + "="*50)
    logger.info("BACKTESTING SUMMARY")
    logger.info("="*50)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {total_tests - completed}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
