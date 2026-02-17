"""
Report generation module for backtesting results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

from src.backtesting.engine import BacktestResult
from src.utils.logger import LoggerMixin


class ReportGenerator(LoggerMixin):
    """
    Generate comprehensive reports from backtesting results.
    """

    def __init__(
        self,
        all_results: List[BacktestResult],
        best_per_ticker: Dict[str, BacktestResult],
        metrics_df: Optional[pd.DataFrame] = None
    ):
        """
        Initialize report generator.

        Parameters
        ----------
        all_results : list
            All backtest results
        best_per_ticker : dict
            Best strategy per ticker
        metrics_df : pd.DataFrame, optional
            Metrics DataFrame
        """
        self.all_results = all_results
        self.best_per_ticker = best_per_ticker
        self.metrics_df = metrics_df

        self.logger.info(f"Initialized ReportGenerator with {len(all_results)} results")

    def generate_comprehensive_report(
        self,
        output_path: Union[str, Path],
        format: str = 'markdown'
    ) -> None:
        """
        Generate comprehensive report.

        Parameters
        ----------
        output_path : str or Path
            Output file path
        format : str
            Output format ('markdown', 'html', 'excel')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == 'markdown':
            content = self._generate_markdown_report()
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

        elif format == 'html':
            content = self._generate_html_report()
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

        elif format == 'excel':
            self._generate_excel_report(output_file)

        else:
            raise ValueError(f"Unknown format: {format}")

        self.logger.info(f"Generated {format} report: {output_file}")

    def _generate_markdown_report(self) -> str:
        """Generate markdown report."""
        report_lines = []

        # Header
        report_lines.append("# Quantitative Trading Strategy Analysis Report")
        report_lines.append("")
        report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Executive Summary
        report_lines.append("## 1. Executive Summary")
        report_lines.append("")
        summary = self._generate_executive_summary()
        for key, value in summary.items():
            report_lines.append(f"- **{key}:** {value}")
        report_lines.append("")

        # Overall Best Strategy
        report_lines.append("## 2. Overall Best Strategy")
        report_lines.append("")
        best_overall = max(self.all_results, key=lambda x: x.sharpe_ratio)
        report_lines.append(self._format_result_markdown(best_overall))
        report_lines.append("")

        # Best Strategy Per Ticker
        report_lines.append("## 3. Best Strategy Per Ticker")
        report_lines.append("")
        for ticker, result in sorted(self.best_per_ticker.items()):
            report_lines.append(f"### {ticker}")
            report_lines.append("")
            report_lines.append(f"**Strategy:** {result.strategy_name}")
            report_lines.append("")
            report_lines.append(self._format_metrics_table_markdown(result))
            report_lines.append("")

        # Strategy Performance Summary
        report_lines.append("## 4. Strategy Performance Summary")
        report_lines.append("")
        report_lines.append(self._generate_strategy_summary_markdown())
        report_lines.append("")

        # Risk Analysis
        report_lines.append("## 5. Risk Analysis")
        report_lines.append("")
        report_lines.append(self._generate_risk_analysis_markdown())
        report_lines.append("")

        # Conclusion
        report_lines.append("## 6. Conclusion")
        report_lines.append("")
        report_lines.append(self._generate_conclusion_markdown())
        report_lines.append("")

        return '\n'.join(report_lines)

    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        # Convert markdown to HTML (simplified version)
        markdown_content = self._generate_markdown_report()

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Strategy Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{ margin: 0; }}
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        h3 {{
            color: #764ba2;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 10px 0;
        }}
        .positive {{ color: #2ecc71; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Quantitative Trading Strategy Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <div class="content">
        {self._markdown_to_html_simple(markdown_content)}
    </div>
</body>
</html>
"""
        return html_template

    def _generate_excel_report(self, output_path: Path) -> None:
        """Generate Excel report."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = []
            summary = self._generate_executive_summary()
            for key, value in summary.items():
                summary_data.append({'Metric': key, 'Value': value})

            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name='Summary', index=False
            )

            # Sheet 2: All Results
            if self.metrics_df is not None:
                self.metrics_df.to_excel(
                    writer, sheet_name='All Results', index=False
                )

            # Sheet 3: Best Per Ticker
            best_data = []
            for ticker, result in sorted(self.best_per_ticker.items()):
                best_data.append({
                    'Ticker': ticker,
                    'Strategy': result.strategy_name,
                    'Total Return': f"{result.total_return:.2%}",
                    'Annual Return': f"{result.annual_return:.2%}",
                    'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
                    'Max Drawdown': f"{result.max_drawdown:.2%}",
                    'Win Rate': f"{result.win_rate:.2%}",
                    'Total Trades': result.total_trades
                })

            pd.DataFrame(best_data).to_excel(
                writer, sheet_name='Best Per Ticker', index=False
            )

            # Sheet 4: Strategy Averages
            strategy_avg = self._calculate_strategy_averages()
            strategy_avg.to_excel(writer, sheet_name='Strategy Averages')

    def _generate_executive_summary(self) -> Dict:
        """Generate executive summary statistics."""
        summary = {
            'Analysis Period': f"{self.all_results[0].start_date.strftime('%Y-%m-%d')} to {self.all_results[0].end_date.strftime('%Y-%m-%d')}" if self.all_results else 'N/A',
            'Total Strategies Tested': len(set(r.strategy_name for r in self.all_results)),
            'Total Tickers Analyzed': len(set(r.ticker for r in self.all_results)),
            'Total Backtests': len(self.all_results),
            'Best Overall Strategy': max(self.all_results, key=lambda x: x.sharpe_ratio).strategy_name if self.all_results else 'N/A',
            'Best Overall Ticker': max(self.all_results, key=lambda x: x.total_return).ticker if self.all_results else 'N/A',
            'Average Sharpe Ratio': f"{np.mean([r.sharpe_ratio for r in self.all_results]):.2f}" if self.all_results else 'N/A',
            'Average Return': f"{np.mean([r.total_return for r in self.all_results]):.2%}" if self.all_results else 'N/A'
        }

        return summary

    def _format_result_markdown(self, result: BacktestResult) -> str:
        """Format a single result as markdown."""
        lines = [
            f"**Strategy:** {result.strategy_name}",
            f"**Ticker:** {result.ticker}",
            f"**Period:** {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}",
            "",
            "### Performance Metrics",
            "",
            f"- **Total Return:** {result.total_return:.2%}",
            f"- **Annual Return:** {result.annual_return:.2%}",
            f"- **Sharpe Ratio:** {result.sharpe_ratio:.2f}",
            f"- **Sortino Ratio:** {result.sortino_ratio:.2f}",
            f"- **Max Drawdown:** {result.max_drawdown:.2%}",
            f"- **Volatility:** {result.volatility:.2%}",
            f"- **Win Rate:** {result.win_rate:.2%}",
            f"- **Profit Factor:** {result.profit_factor:.2f}",
            f"- **Total Trades:** {result.total_trades}"
        ]

        return '\n'.join(lines)

    def _format_metrics_table_markdown(self, result: BacktestResult) -> str:
        """Format metrics as markdown table."""
        lines = [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Return | {result.total_return:.2%} |",
            f"| Annual Return | {result.annual_return:.2%} |",
            f"| Sharpe Ratio | {result.sharpe_ratio:.2f} |",
            f"| Sortino Ratio | {result.sortino_ratio:.2f} |",
            f"| Max Drawdown | {result.max_drawdown:.2%} |",
            f"| Volatility | {result.volatility:.2%} |",
            f"| Win Rate | {result.win_rate:.2%} |",
            f"| Profit Factor | {result.profit_factor:.2f} |",
            f"| Total Trades | {result.total_trades} |"
        ]

        return '\n'.join(lines)

    def _generate_strategy_summary_markdown(self) -> str:
        """Generate strategy performance summary."""
        strategy_avg = self._calculate_strategy_averages()

        lines = ["### Average Performance by Strategy", ""]

        if not strategy_avg.empty:
            lines.append(strategy_avg.to_markdown())
        else:
            lines.append("No data available")

        return '\n'.join(lines)

    def _generate_risk_analysis_markdown(self) -> str:
        """Generate risk analysis section."""
        lines = [
            "### Risk Metrics Summary",
            ""
        ]

        # Calculate risk statistics
        max_drawdowns = [r.max_drawdown for r in self.all_results]
        volatilities = [r.volatility for r in self.all_results]

        lines.extend([
            f"- **Average Max Drawdown:** {np.mean(max_drawdowns):.2%}",
            f"- **Worst Drawdown:** {min(max_drawdowns):.2%}",
            f"- **Best Drawdown:** {max(max_drawdowns):.2%}",
            f"- **Average Volatility:** {np.mean(volatilities):.2%}",
            f"- **Lowest Volatility:** {min(volatilities):.2%}",
            f"- **Highest Volatility:** {max(volatilities):.2%}"
        ])

        # Lowest risk strategy
        lowest_risk = min(self.all_results, key=lambda x: abs(x.max_drawdown))
        lines.extend([
            "",
            "### Lowest Risk Strategy",
            "",
            f"- **Strategy:** {lowest_risk.strategy_name}",
            f"- **Ticker:** {lowest_risk.ticker}",
            f"- **Max Drawdown:** {lowest_risk.max_drawdown:.2%}",
            f"- **Return:** {lowest_risk.total_return:.2%}"
        ])

        return '\n'.join(lines)

    def _generate_conclusion_markdown(self) -> str:
        """Generate conclusion section."""
        best_overall = max(self.all_results, key=lambda x: x.sharpe_ratio)

        lines = [
            "### Key Findings",
            "",
            f"1. **Best Risk-Adjusted Performance:** {best_overall.strategy_name} on {best_overall.ticker} with Sharpe Ratio of {best_overall.sharpe_ratio:.2f}",
            "",
            f"2. **Highest Return:** {max(self.all_results, key=lambda x: x.total_return).strategy_name} achieved {max(r.total_return for r in self.all_results):.2%} total return",
            "",
            f"3. **Lowest Drawdown:** {min(self.all_results, key=lambda x: abs(x.max_drawdown)).strategy_name} with {min(abs(r.max_drawdown) for r in self.all_results):.2%} max drawdown",
            "",
            "### Recommendations",
            "",
            "1. Consider diversifying across multiple strategies to reduce risk",
            "2. Focus on strategies with high Sharpe ratios for better risk-adjusted returns",
            "3. Monitor drawdown levels and implement risk management rules",
            "4. Regular rebalancing may improve overall portfolio performance"
        ]

        return '\n'.join(lines)

    def _calculate_strategy_averages(self) -> pd.DataFrame:
        """Calculate average metrics per strategy."""
        strategy_data = {}

        for result in self.all_results:
            if result.strategy_name not in strategy_data:
                strategy_data[result.strategy_name] = {
                    'returns': [],
                    'sharpe': [],
                    'mdd': [],
                    'win_rate': []
                }

            strategy_data[result.strategy_name]['returns'].append(result.total_return)
            strategy_data[result.strategy_name]['sharpe'].append(result.sharpe_ratio)
            strategy_data[result.strategy_name]['mdd'].append(result.max_drawdown)
            strategy_data[result.strategy_name]['win_rate'].append(result.win_rate)

        rows = []
        for strategy, data in strategy_data.items():
            rows.append({
                'Strategy': strategy,
                'Avg Return': f"{np.mean(data['returns']):.2%}",
                'Avg Sharpe': f"{np.mean(data['sharpe']):.2f}",
                'Avg Max DD': f"{np.mean(data['mdd']):.2%}",
                'Avg Win Rate': f"{np.mean(data['win_rate']):.2%}"
            })

        return pd.DataFrame(rows)

    def _markdown_to_html_simple(self, markdown: str) -> str:
        """Simple markdown to HTML conversion."""
        html = markdown

        # Headers
        html = html.replace('### ', '<h3>').replace('\n', '</h3>\n', 1)
        html = html.replace('## ', '<h2>').replace('\n', '</h2>\n', 1)
        html = html.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)

        # Bold
        import re
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # Lists
        html = re.sub(r'^\- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', html, flags=re.DOTALL)

        # Tables (basic support)
        html = re.sub(r'^\|(.+)\|$', r'<tr>\1</tr>', html, flags=re.MULTILINE)
        html = re.sub(r'\|', '</td><td>', html)
        html = html.replace('<tr><td>', '<tr>')
        html = html.replace('</td></tr>', '</tr>')

        # Paragraphs
        html = html.replace('\n\n', '</p><p>')
        html = '<p>' + html + '</p>'

        return html
