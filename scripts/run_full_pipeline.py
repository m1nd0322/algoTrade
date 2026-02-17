#!/usr/bin/env python3
"""
Full Pipeline Script

Executes the complete trading system pipeline:
1. Data collection
2. Data preprocessing
3. Backtesting
4. Analysis and reporting
"""

import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete quant trading pipeline"
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        help='List of tickers to analyze'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--strategies',
        nargs='+',
        default=['all'],
        help='Strategies to test (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for all results'
    )
    parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='Skip data collection (use existing data)'
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing (use existing processed data)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run backtests in parallel'
    )
    parser.add_argument(
        '--report-format',
        choices=['markdown', 'html', 'pdf', 'excel', 'all'],
        default='all',
        help='Report output format'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    return parser.parse_args()


def run_command(cmd: list, logger, step_name: str) -> bool:
    """Run a command and log output."""
    logger.info(f"\n{'='*70}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"{'='*70}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        if result.stdout:
            logger.info(result.stdout)

        logger.info(f"✓ {step_name} completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} failed with error:")
        logger.error(e.stderr)
        return False


def main():
    """Main execution function."""
    args = parse_args()

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(
        name='full_pipeline',
        log_dir='results/logs',
        level='DEBUG' if args.verbose else 'INFO',
        log_file=f'pipeline_{timestamp}.log'
    )

    # Set end date to today if not provided
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    logger.info("="*70)
    logger.info("QUANT TRADING AI SYSTEM - FULL PIPELINE")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Date range: {args.start_date} to {end_date}")
    logger.info(f"Strategies: {args.strategies}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*70)

    scripts_dir = Path(__file__).parent
    success = True

    # Step 1: Data Collection
    if not args.skip_collection:
        cmd = [
            sys.executable,
            str(scripts_dir / 'run_data_collection.py'),
            '--tickers'] + args.tickers + [
            '--start-date', args.start_date,
            '--end-date', end_date
        ]
        if args.verbose:
            cmd.append('--verbose')

        if not run_command(cmd, logger, "Data Collection"):
            logger.error("Pipeline failed at data collection step")
            sys.exit(1)
    else:
        logger.info("Skipping data collection (--skip-collection)")

    # Step 2: Data Preprocessing
    if not args.skip_preprocessing:
        cmd = [
            sys.executable,
            str(scripts_dir / 'run_preprocessing.py'),
            '--input-dir', 'data/raw',
            '--output-dir', 'data/processed'
        ]
        if args.verbose:
            cmd.append('--verbose')

        if not run_command(cmd, logger, "Data Preprocessing"):
            logger.error("Pipeline failed at preprocessing step")
            sys.exit(1)
    else:
        logger.info("Skipping preprocessing (--skip-preprocessing)")

    # Step 3: Backtesting
    cmd = [
        sys.executable,
        str(scripts_dir / 'run_backtesting.py'),
        '--data-dir', 'data/splits',
        '--strategies'] + args.strategies + [
        '--output-dir', f'{args.output_dir}/backtests'
    ]
    if args.parallel:
        cmd.append('--parallel')
    if args.verbose:
        cmd.append('--verbose')

    if not run_command(cmd, logger, "Backtesting"):
        logger.error("Pipeline failed at backtesting step")
        sys.exit(1)

    # Step 4: Analysis and Reporting
    cmd = [
        sys.executable,
        str(scripts_dir / 'run_analysis.py'),
        '--results-dir', f'{args.output_dir}/backtests',
        '--output-dir', f'{args.output_dir}/reports',
        '--viz-dir', f'{args.output_dir}/visualizations',
        '--format', args.report_format
    ]
    if args.verbose:
        cmd.append('--verbose')

    if not run_command(cmd, logger, "Analysis and Reporting"):
        logger.error("Pipeline failed at analysis step")
        sys.exit(1)

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"\nResults available at:")
    logger.info(f"  - Backtests: {args.output_dir}/backtests")
    logger.info(f"  - Reports: {args.output_dir}/reports")
    logger.info(f"  - Visualizations: {args.output_dir}/visualizations")
    logger.info(f"  - Logs: results/logs/pipeline_{timestamp}.log")
    logger.info("="*70)


if __name__ == "__main__":
    main()
