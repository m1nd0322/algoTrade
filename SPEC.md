# 퀀트 AI 트레이딩 시스템 명세서

## 1. 프로젝트 개요

### 1.1 목적
"퀀트 전략을 위한 인공지능 트레이딩" 책의 모든 투자 전략을 통합하여, 실시간 미국 주식 데이터를 기반으로 자동화된 백테스팅, 성과 분석, 전략 선택 및 리포트 생성을 수행하는 통합 시스템 구축

### 1.2 핵심 기능
- 실시간 미국 주식 데이터 자동 수집
- 데이터 전처리 및 교차 검증
- 다중 전략 백테스팅 (전통 퀀트 + 머신러닝 + 딥러닝)
- 성과 지표 시각화 및 비교 분석
- 최적 전략 자동 선택 및 리포트 생성

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Acquisition Layer                   │
│  - Real-time US Stock Data Fetcher                          │
│  - API Integration (yfinance, pandas-datareader, etc.)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Processing Layer                       │
│  - Data Cleaning & Normalization                            │
│  - Feature Engineering                                      │
│  - Cross-Validation Split                                   │
│  - CSV Storage Management                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Strategy Execution Layer                   │
│  ┌───────────────┐ ┌───────────────┐ ┌──────────────────┐ │
│  │ Traditional   │ │ Machine       │ │ Deep Learning    │ │
│  │ Quant         │ │ Learning      │ │                  │ │
│  │ Strategies    │ │ Strategies    │ │ Strategies       │ │
│  └───────────────┘ └───────────────┘ └──────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Analysis & Reporting Layer                  │
│  - Performance Metrics Calculation                          │
│  - Visualization                                            │
│  - Strategy Comparison                                      │
│  - Report Generation                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 모듈별 상세 명세

### 3.1 데이터 수집 모듈 (Data Acquisition)

#### 3.1.1 기능
- 미국 주식 시장 데이터 실시간 수집
- 복수 티커에 대한 병렬 데이터 다운로드
- 데이터 품질 검증

#### 3.1.2 데이터 소스
- **Primary**: yfinance
- **Secondary**: pandas-datareader, Quandl
- **Backup**: Alpha Vantage API

#### 3.1.3 수집 데이터 항목
```python
{
    'OHLCV': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'Adjusted': ['Adj Close'],
    'Metadata': ['Ticker', 'Date', 'Sector', 'Market Cap']
}
```

#### 3.1.4 주요 클래스
```python
class DataCollector:
    def __init__(self, tickers: List[str], start_date: str, end_date: str)
    def fetch_data(self) -> pd.DataFrame
    def validate_data(self) -> bool
    def save_raw_data(self, path: str) -> None
```

#### 3.1.5 파일명 규칙
```
data/raw/{TICKER}_{START_DATE}_{END_DATE}_raw.csv
예: data/raw/AAPL_20200101_20231231_raw.csv
```

---

### 3.2 데이터 전처리 모듈 (Data Processing)

#### 3.2.1 전처리 단계

**Step 1: 데이터 정제**
- 결측치 처리 (forward fill, interpolation)
- 이상치 탐지 및 처리
- 주식 분할/배당 조정

**Step 2: 특성 공학 (Feature Engineering)**
- 기술적 지표 생성
  - 이동평균 (SMA, EMA)
  - RSI, MACD, Bollinger Bands
  - Stochastic Oscillator
- 수익률 계산
  - 일간 수익률
  - 누적 수익률
  - 로그 수익률
- 가격 패턴
  - 캔들스틱 패턴
  - 차트 패턴

**Step 3: 정규화/표준화**
- MinMax Scaling
- Standard Scaling (Z-score)
- Robust Scaling

**Step 4: 교차 검증 분할**
- Time Series Split
- Walk-Forward Analysis
- Train/Validation/Test 분할 비율: 60/20/20

#### 3.2.2 주요 클래스
```python
class DataPreprocessor:
    def __init__(self, raw_data: pd.DataFrame)
    def clean_data(self) -> pd.DataFrame
    def engineer_features(self) -> pd.DataFrame
    def normalize_data(self, method: str) -> pd.DataFrame
    def split_data(self, ratios: Tuple[float, float, float]) -> Dict
    def save_processed_data(self, path: str) -> None
```

#### 3.2.3 파일명 규칙
```
data/processed/{TICKER}_processed.csv
data/features/{TICKER}_features.csv
data/splits/{TICKER}_train.csv
data/splits/{TICKER}_val.csv
data/splits/{TICKER}_test.csv
```

---

### 3.3 전략 실행 모듈 (Strategy Execution)

#### 3.3.1 전통 퀀트 전략 (Traditional Quant Strategies)

**A. 바이앤홀드 (Buy and Hold)**
```python
class BuyAndHoldStrategy(BaseStrategy):
    strategy_name = "Buy and Hold"
    params = {}
```

**B. 평균 회귀 (Mean Reversion)**
```python
class MeanReversionStrategy(BaseStrategy):
    strategy_name = "Mean Reversion"
    params = {
        'window': 20,
        'std_dev': 2.0,
        'entry_threshold': -2.0,
        'exit_threshold': 0.0
    }
```

**C. 모멘텀 (Momentum)**
```python
class MomentumStrategy(BaseStrategy):
    strategy_name = "Momentum"
    params = {
        'lookback_period': 60,
        'holding_period': 20
    }
```

**D. 절대 모멘텀 (Absolute Momentum)**
```python
class AbsoluteMomentumStrategy(BaseStrategy):
    strategy_name = "Absolute Momentum"
    params = {
        'lookback_period': 252,
        'risk_free_rate': 0.02
    }
```

**E. 상대 모멘텀 (Relative Momentum)**
```python
class RelativeMomentumStrategy(BaseStrategy):
    strategy_name = "Relative Momentum"
    params = {
        'lookback_period': 126,
        'top_n': 5
    }
```

**F. 가치 투자 (Value Investing)**
```python
class ValueInvestingStrategy(BaseStrategy):
    strategy_name = "Value Investing"
    params = {
        'pe_threshold': 15,
        'pb_threshold': 1.5,
        'rebalance_frequency': 'quarterly'
    }
```

#### 3.3.2 머신러닝 전략 (Machine Learning Strategies)

**A. k-최근접 이웃 (k-Nearest Neighbors)**
```python
class KNNStrategy(MLStrategy):
    strategy_name = "KNN"
    params = {
        'n_neighbors': 5,
        'features': ['returns', 'volume', 'volatility'],
        'prediction_horizon': 5
    }
```

**B. 랜덤 포레스트 (Random Forest)**
```python
class RandomForestStrategy(MLStrategy):
    strategy_name = "Random Forest"
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5
    }
```

**C. XGBoost**
```python
class XGBoostStrategy(MLStrategy):
    strategy_name = "XGBoost"
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5
    }
```

**D. 클러스터링 (Clustering)**
```python
class ClusteringStrategy(MLStrategy):
    strategy_name = "Clustering"
    params = {
        'n_clusters': 5,
        'algorithm': 'kmeans',
        'features': ['returns', 'volatility', 'volume']
    }
```

#### 3.3.3 딥러닝 전략 (Deep Learning Strategies)

**A. CNN 캔들차트 예측 (CNN for Candlestick Prediction)**
```python
class CNNCandlestickStrategy(DLStrategy):
    strategy_name = "CNN Candlestick"
    params = {
        'image_size': (64, 64),
        'lookback_window': 20,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001
    }
```

**B. RNN/LSTM 주가 방향성 예측**
```python
class LSTMDirectionStrategy(DLStrategy):
    strategy_name = "LSTM Direction"
    params = {
        'sequence_length': 60,
        'lstm_units': [128, 64],
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32
    }
```

**C. GRU 전략**
```python
class GRUStrategy(DLStrategy):
    strategy_name = "GRU"
    params = {
        'sequence_length': 60,
        'gru_units': [128, 64],
        'dropout': 0.2,
        'epochs': 100
    }
```

**D. 오토인코더 (Autoencoder)**
```python
class AutoencoderStrategy(DLStrategy):
    strategy_name = "Autoencoder"
    params = {
        'encoding_dim': 32,
        'epochs': 100,
        'anomaly_threshold': 0.95
    }
```

**E. Transformer**
```python
class TransformerStrategy(DLStrategy):
    strategy_name = "Transformer"
    params = {
        'sequence_length': 60,
        'num_heads': 8,
        'num_layers': 4,
        'd_model': 128
    }
```

#### 3.3.4 전략 베이스 클래스
```python
class BaseStrategy(ABC):
    def __init__(self, data: pd.DataFrame, params: Dict)

    @abstractmethod
    def generate_signals(self) -> pd.Series

    @abstractmethod
    def backtest(self) -> Dict

    def calculate_positions(self, signals: pd.Series) -> pd.Series
    def calculate_returns(self, positions: pd.Series) -> pd.Series
```

---

### 3.4 백테스팅 엔진 (Backtesting Engine)

#### 3.4.1 백테스팅 설정
```python
class BacktestConfig:
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    position_size: float = 1.0  # 전액 투자
    rebalance_frequency: str = 'daily'
```

#### 3.4.2 백테스팅 실행
```python
class BacktestEngine:
    def __init__(self, strategy: BaseStrategy, config: BacktestConfig)

    def run(self) -> BacktestResult
    def calculate_metrics(self) -> Dict
    def generate_trades_log(self) -> pd.DataFrame
```

#### 3.4.3 백테스팅 결과
```python
@dataclass
class BacktestResult:
    strategy_name: str
    ticker: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.Series
    trades_log: pd.DataFrame
```

---

### 3.5 성과 분석 모듈 (Performance Analysis)

#### 3.5.1 성과 지표

**수익률 지표**
- 총 수익률 (Total Return)
- 연간 수익률 (Annual Return)
- 월간 수익률 (Monthly Return)
- CAGR (Compound Annual Growth Rate)

**위험 지표**
- 변동성 (Volatility)
- 최대 낙폭 (Maximum Drawdown)
- 평균 낙폭 (Average Drawdown)
- 낙폭 기간 (Drawdown Duration)

**위험 조정 수익률**
- 샤프 비율 (Sharpe Ratio)
- 소르티노 비율 (Sortino Ratio)
- 칼마 비율 (Calmar Ratio)
- 정보 비율 (Information Ratio)

**거래 통계**
- 승률 (Win Rate)
- 손익비 (Profit Factor)
- 평균 이익/손실
- 최대 연속 승/패

#### 3.5.2 성과 분석 클래스
```python
class PerformanceAnalyzer:
    def __init__(self, backtest_results: List[BacktestResult])

    def calculate_all_metrics(self) -> pd.DataFrame
    def compare_strategies(self) -> pd.DataFrame
    def rank_strategies(self, metric: str = 'sharpe_ratio') -> pd.DataFrame
    def find_best_strategy_per_ticker(self) -> Dict[str, BacktestResult]
```

---

### 3.6 시각화 모듈 (Visualization)

#### 3.6.1 차트 종류

**1. 수익 곡선 (Equity Curve)**
```python
def plot_equity_curve(results: List[BacktestResult], ticker: str)
    # 전략별 수익 곡선 비교
```

**2. 낙폭 차트 (Drawdown Chart)**
```python
def plot_drawdown(results: List[BacktestResult], ticker: str)
    # 전략별 낙폭 비교
```

**3. 월간 수익률 히트맵 (Monthly Returns Heatmap)**
```python
def plot_monthly_returns_heatmap(result: BacktestResult)
    # 월별 수익률 시각화
```

**4. 성과 지표 비교 차트 (Performance Metrics Comparison)**
```python
def plot_metrics_comparison(results: List[BacktestResult])
    # 레이더 차트, 바 차트
```

**5. 상관관계 매트릭스 (Correlation Matrix)**
```python
def plot_strategy_correlation(results: List[BacktestResult])
    # 전략간 수익률 상관관계
```

**6. 리스크-리턴 산점도 (Risk-Return Scatter)**
```python
def plot_risk_return_scatter(results: List[BacktestResult])
    # X: 변동성, Y: 수익률
```

#### 3.6.2 시각화 설정
```python
class VisualizationConfig:
    figure_size: Tuple[int, int] = (15, 10)
    dpi: int = 100
    style: str = 'seaborn'
    color_palette: str = 'Set2'
    save_format: str = 'png'
```

---

### 3.7 리포트 생성 모듈 (Report Generation)

#### 3.7.1 리포트 구조

```markdown
# 퀀트 트레이딩 전략 분석 리포트

## 1. Executive Summary
- 분석 기간
- 분석 대상 티커 수
- 총 전략 수
- 최고 성과 전략

## 2. 전략별 성과 요약
### 2.1 전통 퀀트 전략
- 전략별 수익률 테이블
- 주요 지표 비교

### 2.2 머신러닝 전략
- 전략별 수익률 테이블
- 주요 지표 비교

### 2.3 딥러닝 전략
- 전략별 수익률 테이블
- 주요 지표 비교

## 3. 티커별 최고 전략
### 3.1 [TICKER 1]
- 최고 전략: [전략명]
- 총 수익률: [XX.XX%]
- 샤프 비율: [X.XX]
- 최대 낙폭: [XX.XX%]
- 수익 곡선 차트
- 낙폭 차트

### 3.2 [TICKER 2]
...

## 4. 종합 분석
### 4.1 최고 성과 전략
- 전략명
- 적용 티커
- 성과 지표

### 4.2 최저 MDD 전략
- 전략명
- 적용 티커
- MDD 값

### 4.3 전략 카테고리별 분석
- 전통 퀀트 vs 머신러닝 vs 딥러닝
- 평균 성과 비교

## 5. 추천 포트폴리오
- 다양성을 고려한 전략 조합
- 예상 포트폴리오 성과

## 6. 부록
- 전체 백테스팅 결과 테이블
- 상세 거래 내역
```

#### 3.7.2 리포트 생성 클래스
```python
class ReportGenerator:
    def __init__(self,
                 all_results: List[BacktestResult],
                 best_per_ticker: Dict[str, BacktestResult],
                 output_format: str = 'markdown')

    def generate_executive_summary(self) -> str
    def generate_strategy_comparison_table(self) -> str
    def generate_ticker_analysis(self, ticker: str) -> str
    def generate_comprehensive_report(self) -> str
    def export_report(self, path: str) -> None
```

#### 3.7.3 출력 형식
- Markdown (.md)
- HTML (.html)
- PDF (.pdf) - via markdown-to-pdf
- Excel (.xlsx) - 데이터 테이블

---

## 4. 디렉토리 구조

```
algoTrade/
│
├── config/
│   ├── data_config.yaml           # 데이터 수집 설정
│   ├── strategy_config.yaml       # 전략 파라미터 설정
│   ├── backtest_config.yaml       # 백테스팅 설정
│   └── visualization_config.yaml  # 시각화 설정
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── collector.py           # 데이터 수집
│   │   ├── preprocessor.py        # 데이터 전처리
│   │   └── validator.py           # 데이터 검증
│   │
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base.py                # 전략 베이스 클래스
│   │   ├── traditional/
│   │   │   ├── __init__.py
│   │   │   ├── buy_and_hold.py
│   │   │   ├── mean_reversion.py
│   │   │   ├── momentum.py
│   │   │   ├── absolute_momentum.py
│   │   │   ├── relative_momentum.py
│   │   │   └── value_investing.py
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── knn.py
│   │   │   ├── random_forest.py
│   │   │   ├── xgboost.py
│   │   │   └── clustering.py
│   │   └── dl/
│   │       ├── __init__.py
│   │       ├── cnn_candlestick.py
│   │       ├── lstm_direction.py
│   │       ├── gru.py
│   │       ├── autoencoder.py
│   │       └── transformer.py
│   │
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── engine.py              # 백테스팅 엔진
│   │   ├── metrics.py             # 성과 지표 계산
│   │   └── portfolio.py           # 포트폴리오 관리
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── performance.py         # 성과 분석
│   │   └── comparison.py          # 전략 비교
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── charts.py              # 차트 생성
│   │   └── plots.py               # 플롯 유틸리티
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── generator.py           # 리포트 생성
│   │   └── templates/
│   │       ├── markdown_template.md
│   │       └── html_template.html
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # 로깅
│       ├── config_loader.py       # 설정 로더
│       └── helpers.py             # 헬퍼 함수
│
├── data/
│   ├── raw/                       # 원본 데이터
│   ├── processed/                 # 전처리된 데이터
│   ├── features/                  # 특성 데이터
│   └── splits/                    # 학습/검증/테스트 분할
│
├── models/
│   ├── saved_models/              # 저장된 ML/DL 모델
│   └── checkpoints/               # 체크포인트
│
├── results/
│   ├── backtests/                 # 백테스팅 결과
│   ├── metrics/                   # 성과 지표
│   └── logs/                      # 실행 로그
│
├── reports/
│   ├── markdown/                  # 마크다운 리포트
│   ├── html/                      # HTML 리포트
│   ├── pdf/                       # PDF 리포트
│   └── excel/                     # 엑셀 리포트
│
├── visualizations/
│   ├── equity_curves/             # 수익 곡선
│   ├── drawdowns/                 # 낙폭 차트
│   ├── comparisons/               # 비교 차트
│   └── heatmaps/                  # 히트맵
│
├── notebooks/                     # Jupyter 노트북
│   ├── data_exploration.ipynb
│   ├── strategy_development.ipynb
│   └── results_analysis.ipynb
│
├── scripts/
│   ├── run_data_collection.py     # 데이터 수집 실행
│   ├── run_preprocessing.py       # 전처리 실행
│   ├── run_backtesting.py         # 백테스팅 실행
│   ├── run_analysis.py            # 분석 실행
│   └── run_full_pipeline.py       # 전체 파이프라인 실행
│
├── tests/                         # 테스트 코드
│   ├── test_data.py
│   ├── test_strategies.py
│   ├── test_backtesting.py
│   └── test_analysis.py
│
├── ch02/ ... ch08/                # 책 예제 코드 (기존)
├── requirements.txt
├── setup.py
├── README.md
├── SPEC.md                        # 본 문서
└── .gitignore
```

---

## 5. 데이터 플로우

```
[1. 데이터 수집]
    ↓
[실시간 미국 주식 데이터 (yfinance)]
    ↓
[2. 데이터 검증 및 저장]
    ↓
[data/raw/*.csv]
    ↓
[3. 데이터 전처리]
    ├─ 결측치 처리
    ├─ 이상치 처리
    ├─ 특성 공학
    └─ 정규화
    ↓
[4. 데이터 분할 및 저장]
    ├─ Train (60%)
    ├─ Validation (20%)
    └─ Test (20%)
    ↓
[data/splits/*.csv]
    ↓
[5. 전략 실행]
    ├─ 전통 퀀트 전략 (6개)
    ├─ 머신러닝 전략 (4개)
    └─ 딥러닝 전략 (5개)
    ↓
[6. 백테스팅]
    ├─ 시뮬레이션 실행
    ├─ 거래 내역 생성
    └─ 수익 곡선 생성
    ↓
[7. 성과 지표 계산]
    ├─ 수익률 지표
    ├─ 위험 지표
    └─ 위험조정 수익률
    ↓
[8. 전략 비교 및 선택]
    ├─ 티커별 최고 전략
    ├─ 최고 수익률 전략
    └─ 최저 MDD 전략
    ↓
[9. 시각화]
    ├─ 수익 곡선
    ├─ 낙폭 차트
    └─ 비교 차트
    ↓
[10. 리포트 생성]
    └─ 종합 분석 리포트
```

---

## 6. 기술 스택

### 6.1 데이터 수집 및 처리
- **yfinance**: 실시간 주식 데이터
- **pandas**: 데이터 프레임 처리
- **numpy**: 수치 계산
- **pandas-datareader**: 추가 데이터 소스

### 6.2 전통 퀀트 분석
- **TA-Lib**: 기술적 지표
- **backtrader**: 백테스팅 프레임워크
- **zipline**: 대안 백테스팅 프레임워크

### 6.3 머신러닝
- **scikit-learn**: 전통 ML 알고리즘
- **XGBoost**: 그래디언트 부스팅
- **LightGBM**: 경량 그래디언트 부스팅

### 6.4 딥러닝
- **TensorFlow 2.x**: 딥러닝 프레임워크
- **Keras**: 고수준 API
- **PyTorch**: 대안 딥러닝 프레임워크

### 6.5 시각화
- **matplotlib**: 기본 차트
- **seaborn**: 통계 시각화
- **plotly**: 인터랙티브 차트
- **mplfinance**: 금융 차트

### 6.6 리포트 생성
- **jinja2**: 템플릿 엔진
- **markdown**: 마크다운 생성
- **WeasyPrint**: PDF 생성
- **openpyxl**: 엑셀 생성

### 6.7 유틸리티
- **PyYAML**: 설정 파일 관리
- **loguru**: 로깅
- **tqdm**: 진행 바
- **joblib**: 병렬 처리

---

## 7. 실행 플로우

### 7.1 전체 파이프라인 실행

```bash
python scripts/run_full_pipeline.py \
    --tickers AAPL MSFT GOOGL AMZN TSLA \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --strategies all \
    --output-dir ./results
```

### 7.2 단계별 실행

**Step 1: 데이터 수집**
```bash
python scripts/run_data_collection.py \
    --tickers AAPL MSFT GOOGL \
    --start-date 2020-01-01 \
    --end-date 2023-12-31
```

**Step 2: 데이터 전처리**
```bash
python scripts/run_preprocessing.py \
    --input-dir ./data/raw \
    --output-dir ./data/processed
```

**Step 3: 백테스팅**
```bash
python scripts/run_backtesting.py \
    --data-dir ./data/splits \
    --strategies all \
    --output-dir ./results/backtests
```

**Step 4: 분석 및 리포트**
```bash
python scripts/run_analysis.py \
    --results-dir ./results/backtests \
    --output-dir ./reports
```

### 7.3 Python API 사용

```python
from src.data.collector import DataCollector
from src.data.preprocessor import DataPreprocessor
from src.backtesting.engine import BacktestEngine
from src.analysis.performance import PerformanceAnalyzer
from src.reporting.generator import ReportGenerator
from src.strategies import *

# 1. 데이터 수집
collector = DataCollector(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2023-12-31'
)
raw_data = collector.fetch_data()

# 2. 데이터 전처리
preprocessor = DataPreprocessor(raw_data)
processed_data = preprocessor.process()
splits = preprocessor.split_data()

# 3. 전략 초기화
strategies = [
    BuyAndHoldStrategy(),
    MomentumStrategy(),
    LSTMDirectionStrategy(),
    # ... 모든 전략
]

# 4. 백테스팅 실행
results = []
for strategy in strategies:
    for ticker in ['AAPL', 'MSFT', 'GOOGL']:
        engine = BacktestEngine(strategy, splits[ticker])
        result = engine.run()
        results.append(result)

# 5. 성과 분석
analyzer = PerformanceAnalyzer(results)
best_per_ticker = analyzer.find_best_strategy_per_ticker()
metrics_df = analyzer.calculate_all_metrics()

# 6. 리포트 생성
report_gen = ReportGenerator(results, best_per_ticker)
report_gen.generate_comprehensive_report()
report_gen.export_report('./reports/final_report.md')
```

---

## 8. 성과 지표 상세

### 8.1 수익률 지표

**Total Return (총 수익률)**
```
Total Return = (Final Portfolio Value - Initial Capital) / Initial Capital
```

**Annual Return (연간 수익률)**
```
Annual Return = (1 + Total Return) ^ (252 / Number of Trading Days) - 1
```

**CAGR (연평균 성장률)**
```
CAGR = (Final Value / Initial Value) ^ (1 / Years) - 1
```

### 8.2 위험 지표

**Volatility (변동성)**
```
Volatility = Std(Daily Returns) * sqrt(252)
```

**Maximum Drawdown (최대 낙폭)**
```
MDD = max(Peak - Trough) / Peak
```

### 8.3 위험조정 수익률

**Sharpe Ratio (샤프 비율)**
```
Sharpe = (Annual Return - Risk Free Rate) / Annual Volatility
```

**Sortino Ratio (소르티노 비율)**
```
Sortino = (Annual Return - Risk Free Rate) / Downside Deviation
```

**Calmar Ratio (칼마 비율)**
```
Calmar = Annual Return / Maximum Drawdown
```

### 8.4 거래 통계

**Win Rate (승률)**
```
Win Rate = Number of Winning Trades / Total Trades
```

**Profit Factor (손익비)**
```
Profit Factor = Gross Profit / Gross Loss
```

---

## 9. 리포트 예시

### 9.1 Executive Summary

```markdown
# 퀀트 트레이딩 전략 종합 분석 리포트

**분석 기간**: 2020-01-01 ~ 2023-12-31
**분석 티커**: AAPL, MSFT, GOOGL, AMZN, TSLA (5종목)
**적용 전략**: 15개 (전통 퀀트 6개, 머신러닝 4개, 딥러닝 5개)
**총 백테스팅 수**: 75회

## 핵심 발견사항

1. **최고 성과 전략**: LSTM Direction Strategy on TSLA
   - 총 수익률: 245.3%
   - 샤프 비율: 2.15
   - 최대 낙폭: -18.5%

2. **최저 MDD 전략**: Value Investing Strategy on AAPL
   - 최대 낙폭: -8.2%
   - 총 수익률: 68.5%
   - 샤프 비율: 1.42

3. **카테고리별 평균 성과**
   - 전통 퀀트: 평균 수익률 52.3%, 샤프 1.12
   - 머신러닝: 평균 수익률 78.6%, 샤프 1.45
   - 딥러닝: 평균 수익률 95.2%, 샤프 1.68
```

### 9.2 티커별 최고 전략

```markdown
## AAPL - 최고 전략: Transformer Strategy

**성과 지표**
- 총 수익률: 128.4%
- 연간 수익률: 32.1%
- 샤프 비율: 1.87
- 소르티노 비율: 2.34
- 최대 낙폭: -12.3%
- 승률: 58.2%

**수익 곡선**
[차트 이미지]

**월간 수익률 히트맵**
[히트맵 이미지]
```

---

## 10. 확장 가능성

### 10.1 추가 기능
- 실시간 거래 시스템 연동
- 알림/알람 시스템
- 웹 대시보드 (Streamlit/Dash)
- 클라우드 배포 (AWS/GCP)

### 10.2 추가 전략
- Pairs Trading
- Statistical Arbitrage
- Factor Investing
- Reinforcement Learning

### 10.3 데이터 확장
- 글로벌 시장 (유럽, 아시아)
- 암호화폐
- 선물/옵션
- 대체 데이터 (뉴스, 소셜 미디어)

---

## 11. 주의사항 및 면책조항

1. 본 시스템은 교육 및 연구 목적으로 개발되었습니다.
2. 백테스팅 결과는 과거 데이터 기반이며, 미래 성과를 보장하지 않습니다.
3. 실제 투자 시 거래 비용, 슬리피지, 시장 충격 등 추가 요인을 고려해야 합니다.
4. 투자 결정은 본인의 책임이며, 본 시스템은 투자 조언이 아닙니다.
5. 실전 투자 전 충분한 검증과 리스크 관리가 필요합니다.

---

## 12. 참고 문헌

- 퀀트 전략을 위한 인공지능 트레이딩 (도서)
- scikit-learn Documentation
- TensorFlow/Keras Documentation
- backtrader Documentation
- 기타 관련 논문 및 자료

---

**문서 버전**: 1.0
**작성일**: 2025-10-24
**작성자**: AI Trading System Development Team
