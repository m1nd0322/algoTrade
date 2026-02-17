#!/usr/bin/env python3
"""
Data Preprocessing Script

Preprocesses raw stock data, performs feature engineering, and splits data.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess stock market data"
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Input directory containing raw data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/data_config.yaml',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='List of tickers to process (default: all in input dir)'
    )
    parser.add_argument(
        '--skip-feature-engineering',
        action='store_true',
        help='Skip feature engineering step'
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
        name='preprocessing',
        log_dir='results/logs',
        level='DEBUG' if args.verbose else 'INFO'
    )

    logger.info("Starting data preprocessing...")

    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.load()

    # Get list of files to process
    input_path = Path(args.input_dir)
    if args.tickers:
        files = [input_path / f"{ticker}_*.csv" for ticker in args.tickers]
    else:
        files = list(input_path.glob("*.csv"))

    logger.info(f"Found {len(files)} files to process")

    # Process each file
    for file_path in files:
        try:
            logger.info(f"\nProcessing {file_path.name}...")

            # Initialize preprocessor
            preprocessor = DataPreprocessor(
                data_path=file_path,
                config=config
            )

            # Step 1: Clean data
            logger.info("Step 1/4: Cleaning data...")
            cleaned_data = preprocessor.clean_data()

            # Step 2: Feature engineering
            if not args.skip_feature_engineering:
                logger.info("Step 2/4: Engineering features...")
                featured_data = preprocessor.engineer_features(cleaned_data)
            else:
                featured_data = cleaned_data
                logger.info("Step 2/4: Skipped (--skip-feature-engineering)")

            # Step 3: Normalize data
            logger.info("Step 3/4: Normalizing data...")
            normalized_data = preprocessor.normalize_data(featured_data)

            # Step 4: Split data
            logger.info("Step 4/4: Splitting data...")
            splits = preprocessor.split_data(normalized_data)

            # Save processed data
            ticker = file_path.stem.split('_')[0]
            preprocessor.save_processed_data(
                processed_data=normalized_data,
                splits=splits,
                ticker=ticker,
                output_dir=args.output_dir
            )

            logger.info(f"✓ Completed processing {ticker}")
            logger.info(f"  - Train: {len(splits['train'])} samples")
            logger.info(f"  - Validation: {len(splits['validation'])} samples")
            logger.info(f"  - Test: {len(splits['test'])} samples")

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}", exc_info=True)
            continue

    logger.info("\n" + "="*50)
    logger.info("Preprocessing completed successfully!")
    logger.info("="*50)


if __name__ == "__main__":
    main()
