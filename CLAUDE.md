# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational repository for the book "퀀트 전략을 위한 인공지능 트레이딩" (AI Trading for Quantitative Strategies). The codebase demonstrates financial data analysis and algorithmic trading strategies using Python, progressing from traditional technical analysis through machine learning to deep learning approaches.

**Target Python Version**: 3.6
**Deep Learning Stack**: TensorFlow 1.15.0, Keras 2.2.4
**Note**: This is legacy code using older TensorFlow 1.x. The newer transformer example in `scripts/aapl_transformer.py` uses TensorFlow 2.x.

## Environment Setup

### Create and activate Python 3.6 environment:

**Using Conda (recommended):**
```bash
# Create environment
conda create -n py36 python=3.6

# Activate on Linux/Mac
source activate py36

# Activate on Windows
activate py36
```

### Install dependencies:
```bash
# Install all required packages
pip install -r requirements.txt
```

**Important**: `requirements.txt` includes Windows-specific packages (`pywin32`, `pywinpty`, `win-inet-pton`, `wincertstore`). On Linux/Mac, these may fail to install but can be safely ignored as they're not required for core functionality.

## Running Code

### Jupyter Notebooks (Primary format)
```bash
# Start JupyterLab
jupyter lab

# Or start classic Jupyter Notebook
jupyter notebook
```

Most content is in Jupyter notebooks organized by chapter (ch02-ch08). Open the desired chapter folder and run notebooks interactively.

### Standalone Python Scripts
```bash
# Run the transformer-based stock prediction (TensorFlow 2.x)
python scripts/aapl_transformer.py --window 30 --epochs 5

# Run CNN backtesting pipeline (ch08/8.1)
cd "ch08/8.1 CNN을 활용한 캔들차트 예측분석"
python run_all_process.py
```

## Codebase Architecture

### Chapter Organization (Progressive Complexity)

The repository is structured as a textbook, with each chapter building on the previous:

- **ch02**: Foundation - Date/time handling, pandas time series, financial data fetching
- **ch03**: Basic backtesting - Buy and hold strategies
- **ch04**: Technical analysis - Bollinger Bands, momentum strategies, magic formula
- **ch05**: Machine learning regression - Linear regression for stock prediction
- **ch06**: Machine learning classification - KNN, K-Means clustering, ETF-based predictions
- **ch07**: Deep learning basics - Keras API patterns and cheat sheets
- **ch08**: Advanced deep learning - CNN for candlestick charts, LSTM for direction prediction, autoencoders

### Two Development Paradigms

**1. Notebook-First (Educational)**
- Primary teaching format using Jupyter notebooks
- Interactive, exploratory code execution
- Each notebook is self-contained with markdown explanations
- Found in: `ch02/` through `ch08/` and `부록B/`

**2. Modular Script Approach (Production)**
- Reusable Python modules with clear separation of concerns
- Command-line interfaces using argparse
- Found in: `ch08/8.1 CNN을 활용한 캔들차트 예측분석/` and `scripts/`

### Key Modules and Utilities

**CNN Candlestick Prediction Pipeline** (`ch08/8.1 CNN을 활용한 캔들차트 예측분석/`):
```
utils/
├── dataset.py               # Image dataset loader for candlestick PNG charts
├── dataset_traditional.py   # Alternative CSV-based data format
└── get_data.py             # Fetch data from Yahoo Finance/Tiingo

generatedata.py              # Generate training data from price series
preproccess_binclass.py      # Binary classification preprocessing
myDeepCNN.py                 # Custom CNN model architecture
run_all_process.py           # Main orchestration script
```

**Data Flow Pattern:**
```
Financial APIs (Yahoo/Tiingo/FinanceDataReader)
    ↓
Raw CSV/Excel data (stored in data/ and ch*/directories)
    ↓
Preprocessing (normalization, feature engineering, image generation)
    ↓
Model Training (sklearn, Keras, XGBoost, LightGBM)
    ↓
Backtesting (Backtrader framework)
    ↓
Performance metrics and visualization
```

### Data Organization

Data files are stored in two patterns:

1. **Chapter-specific data**: `data/ch03/`, `data/ch04/`, `data/ch08/` - Contains datasets used in specific chapters
2. **Shared ETF data**: `data/us_etf_data/` - Common US stock and ETF data (SPY, AAPL, MSFT, GLD, etc.)
3. **Inline data**: Some chapters (ch04, ch06) contain CSV files directly in their directories

**Large datasets**:
- `ch04/new_us_etf_stock.csv` (4.6 MB)
- `ch08/8.1/stockdatas/` (100+ CSV files for Asian stocks)

## Important Technical Details

### TensorFlow Version Conflicts
- **Legacy code** (ch05-ch08 notebooks): Uses TensorFlow 1.15.0 and Keras 2.2.4
- **Modern code** (`scripts/aapl_transformer.py`): Uses TensorFlow 2.x with built-in Keras

When working with the transformer script, you may need a separate environment with TensorFlow 2.x.

### Model Artifacts
- Trained models are saved as `.h5` files (Keras format)
- Example: `ch08/8.1/50epochs_8batch_cnn_model_dataset_BBNIJK_20_50.h5`
- Training labels stored as `.txt` files
- Generated candlestick chart images stored as PNG files

### Backtesting Framework
The `부록B(파이썬을 이용한 백테스팅 API)` appendix demonstrates:
- Backtrader framework integration
- FinanceDataReader for data fetching
- SQLite storage for historical data
- Web scraping for financial metrics (ROA, P/E ratios)

## Data Sources

The codebase integrates multiple financial data sources:
- **finance-datareader** (0.9.6): Korean and US stock data
- **pandas-datareader** (0.8.1): Yahoo Finance, Google Finance
- **Quandl** (3.4.6): Economic and financial datasets
- **Web scraping**: BeautifulSoup4, Selenium for custom data extraction

## Working with CNN Candlestick Models

The CNN implementation in ch08/8.1 converts OHLCV price data into candlestick chart images, then trains a CNN for pattern recognition:

1. **Data Generation**: `generatedata.py` creates candlestick charts as PNG images
2. **Preprocessing**: `preproccess_binclass.py` labels images for binary classification
3. **Model Training**: `myDeepCNN.py` defines CNN architecture
4. **Execution**: `run_all_process.py` orchestrates the entire pipeline

This approach treats technical analysis as a computer vision problem.

## Language and Localization

- Code comments and notebook markdown cells are in **Korean**
- Variable names and function signatures are in **English**
- File paths may contain Korean characters (e.g., `부록B`)
- When working with paths, use raw strings or proper encoding

## Version Control Notes

Recent commit indicates ongoing modernization:
- Latest PR (#1): "Add transformer example for AAPL" - introduces modern TensorFlow 2.x code
- Previous commits focused on requirement updates and bug fixes for deprecated libraries (e.g., mplfinance)
