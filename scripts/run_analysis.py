#!/usr/bin/env python3
"""
Analysis Script

Analyzes backtesting results, generates visualizations, and creates reports.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.performance import PerformanceAnalyzer
from src.visualization.charts import ChartGenerator
from src.reporting.generator import ReportGenerator
from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze backtesting results and generate reports"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/backtests',
        help='Directory containing backtest results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--viz-dir',
        type=str,
        default='visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/visualization_config.yaml',
        help='Path to visualization configuration file'
    )
    parser.add_argument(
        '--format',
        choices=['markdown', 'html', 'pdf', 'excel', 'all'],
        default='markdown',
        help='Report output format'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization generation'
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
        name='analysis',
        log_dir='results/logs',
        level='DEBUG' if args.verbose else 'INFO'
    )

    logger.info("Starting analysis...")

    # Load configuration
    viz_config = ConfigLoader(args.config).load()

    # Load backtest results
    logger.info(f"Loading results from {args.results_dir}...")
    results_path = Path(args.results_dir)

    if not results_path.exists():
        logger.error(f"Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Initialize analyzer
    analyzer = PerformanceAnalyzer(results_dir=args.results_dir)

    try:
        # Load all results
        all_results = analyzer.load_results()
        logger.info(f"Loaded {len(all_results)} backtest results")

        # Calculate metrics
        logger.info("\nCalculating performance metrics...")
        metrics_df = analyzer.calculate_all_metrics()
        logger.info(f"Calculated metrics for {len(metrics_df)} strategy-ticker combinations")

        # Find best strategies
        logger.info("\nFinding best strategies per ticker...")
        best_per_ticker = analyzer.find_best_strategy_per_ticker()

        logger.info("\nBest strategies:")
        for ticker, result in best_per_ticker.items():
            logger.info(f"  {ticker}: {result.strategy_name} "
                       f"(Return: {result.total_return:.2%}, "
                       f"Sharpe: {result.sharpe_ratio:.2f}, "
                       f"MDD: {result.max_drawdown:.2%})")

        # Generate visualizations
        if not args.no_viz:
            logger.info("\nGenerating visualizations...")
            chart_gen = ChartGenerator(config=viz_config)

            # Generate charts for each ticker
            for ticker in best_per_ticker.keys():
                logger.info(f"  Creating charts for {ticker}...")
                ticker_results = [r for r in all_results if r.ticker == ticker]
                chart_gen.generate_all_charts(
                    results=ticker_results,
                    ticker=ticker,
                    output_dir=args.viz_dir
                )

            # Generate comparison charts
            logger.info("  Creating comparison charts...")
            chart_gen.generate_comparison_charts(
                all_results=all_results,
                output_dir=args.viz_dir
            )

            logger.info(f"Visualizations saved to {args.viz_dir}")

        # Generate reports
        logger.info("\nGenerating reports...")
        report_gen = ReportGenerator(
            all_results=all_results,
            best_per_ticker=best_per_ticker,
            metrics_df=metrics_df
        )

        formats = ['markdown', 'html', 'pdf', 'excel'] if args.format == 'all' else [args.format]

        for fmt in formats:
            logger.info(f"  Creating {fmt.upper()} report...")
            output_path = Path(args.output_dir) / fmt / f"trading_report.{fmt}"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            report_gen.generate_comprehensive_report(
                output_path=output_path,
                format=fmt
            )
            logger.info(f"  ✓ Report saved to {output_path}")

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*70)
        logger.info(f"Total backtests analyzed: {len(all_results)}")
        logger.info(f"Unique tickers: {len(best_per_ticker)}")
        logger.info(f"Unique strategies: {len(set(r.strategy_name for r in all_results))}")
        logger.info("")
        logger.info("Overall Best Strategy:")
        best_overall = max(all_results, key=lambda x: x.sharpe_ratio)
        logger.info(f"  Strategy: {best_overall.strategy_name}")
        logger.info(f"  Ticker: {best_overall.ticker}")
        logger.info(f"  Total Return: {best_overall.total_return:.2%}")
        logger.info(f"  Sharpe Ratio: {best_overall.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {best_overall.max_drawdown:.2%}")
        logger.info("")
        logger.info("Lowest MDD Strategy:")
        best_mdd = min(all_results, key=lambda x: abs(x.max_drawdown))
        logger.info(f"  Strategy: {best_mdd.strategy_name}")
        logger.info(f"  Ticker: {best_mdd.ticker}")
        logger.info(f"  Max Drawdown: {best_mdd.max_drawdown:.2%}")
        logger.info(f"  Total Return: {best_mdd.total_return:.2%}")
        logger.info(f"  Sharpe Ratio: {best_mdd.sharpe_ratio:.2f}")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)

    logger.info("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
