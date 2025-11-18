#!/usr/bin/env python3
"""
Data Collection Script

Collects real-time US stock data using configured data sources.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collector import DataCollector
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect US stock market data"
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='List of tickers to collect (default: from config)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD, default: from config)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD, default: from config)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/data_config.yaml',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: from config)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    # Setup logger
    logger = setup_logger(
        name='data_collection',
        log_dir='results/logs',
        level='DEBUG' if args.verbose else 'INFO'
    )

    logger.info("Starting data collection...")

    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.load()

    # Override config with command line arguments
    tickers = args.tickers or config.get('tickers', [])
    start_date = args.start_date or config.get('date_range', {}).get('start_date')
    end_date = args.end_date or config.get('date_range', {}).get('end_date')
    output_dir = args.output_dir or config.get('storage', {}).get('raw_data_dir')

    logger.info(f"Tickers: {tickers}")
    logger.info(f"Date range: {start_date} to {end_date}")

    # Initialize data collector
    collector = DataCollector(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        config=config
    )

    # Collect data
    try:
        data = collector.fetch_data()
        logger.info(f"Successfully collected data for {len(data)} tickers")

        # Validate data
        if collector.validate_data(data):
            logger.info("Data validation passed")
        else:
            logger.warning("Data validation failed for some tickers")

        # Save data
        collector.save_raw_data(data, output_dir)
        logger.info(f"Data saved to {output_dir}")

        # Print summary
        logger.info("\n" + "="*50)
        logger.info("DATA COLLECTION SUMMARY")
        logger.info("="*50)
        for ticker in tickers:
            if ticker in data:
                rows = len(data[ticker])
                logger.info(f"{ticker}: {rows} data points")
        logger.info("="*50)

    except Exception as e:
        logger.error(f"Error during data collection: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Data collection completed successfully!")


if __name__ == "__main__":
    main()
