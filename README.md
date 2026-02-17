# 퀀트 전략을 위한 인공지능 트레이딩

![book](./표지.jpg)

**"퀀트 전략을 위한 인공지능 트레이딩"** 도서 기반의 종합 퀀트 트레이딩 시스템입니다.
전통적 퀀트 전략부터 머신러닝/딥러닝 전략까지 구현하고, 백테스팅 엔진과 키움증권 실시간 트레이딩까지 지원합니다.

---

## 주요 기능

- **데이터 수집 및 전처리** — yfinance 기반 미국 주식 데이터 수집, 기술적 지표 자동 생성
- **15가지 트레이딩 전략** — 전통 퀀트 6종, 머신러닝 4종, 딥러닝 5종
- **백테스팅 엔진** — 수수료/슬리피지 포함, 30+ 성과 지표 산출
- **분석 및 시각화** — 전략 비교, 리스크 분석, 대시보드, 리포트 생성
- **실시간 트레이딩** — 키움증권 Open API 연동 (KOSPI/KOSDAQ)
- **모의투자 모드** — 실제 주문 없이 전략 검증 가능

---

## 프로젝트 구조

```
algoTrade/
├── config/                          # 설정 파일
│   ├── data_config.yaml             #   데이터 수집/전처리 설정
│   ├── strategy_config.yaml         #   전략 파라미터 설정
│   ├── backtest_config.yaml         #   백테스팅 엔진 설정
│   └── visualization_config.yaml    #   시각화/차트 설정
│
├── src/                             # 소스 코드
│   ├── data/                        # 데이터 레이어
│   │   ├── collector.py             #   데이터 수집 (yfinance, pandas-datareader)
│   │   ├── preprocessor.py          #   전처리 및 기술적 지표 생성
│   │   └── validator.py             #   데이터 품질 검증
│   │
│   ├── strategies/                  # 전략 레이어
│   │   ├── base.py                  #   전략 추상 베이스 클래스
│   │   ├── traditional/             #   전통 퀀트 전략 (6종)
│   │   │   ├── buy_and_hold.py      #     바이앤홀드
│   │   │   ├── momentum.py          #     모멘텀
│   │   │   ├── mean_reversion.py    #     평균회귀
│   │   │   ├── absolute_momentum.py #     절대 모멘텀
│   │   │   ├── relative_momentum.py #     상대 모멘텀
│   │   │   └── value_investing.py   #     가치투자
│   │   ├── ml/                      #   머신러닝 전략 (4종)
│   │   └── dl/                      #   딥러닝 전략 (5종)
│   │
│   ├── backtesting/                 # 백테스팅 엔진
│   │   ├── engine.py                #   백테스트 실행 엔진
│   │   ├── portfolio.py             #   포트폴리오 관리 (포지션, 주문)
│   │   └── metrics.py               #   성과 지표 계산 (30+)
│   │
│   ├── analysis/                    # 분석
│   │   ├── performance.py           #   성과 분석기
│   │   └── comparison.py            #   전략 비교기
│   │
│   ├── visualization/               # 시각화
│   │   ├── charts.py                #   차트 생성기 (대시보드 포함)
│   │   └── plots.py                 #   플롯 유틸리티
│   │
│   ├── reporting/                   # 리포트
│   │   └── generator.py             #   보고서 생성 (Markdown/HTML/Excel)
│   │
│   ├── trading/                     # 실시간 트레이딩
│   │   ├── order_manager.py         #   주문 관리 (주문 생명주기)
│   │   ├── risk_manager.py          #   리스크 관리 (포지션/손실 한도)
│   │   └── kiwoom/                  #   키움증권 연동
│   │       ├── api.py               #     Kiwoom Open API 래퍼
│   │       ├── data_receiver.py     #     실시간 시세 수신
│   │       └── trader.py            #     라이브 트레이더 (오케스트레이터)
│   │
│   └── utils/                       # 유틸리티
│       ├── logger.py                #   로깅
│       ├── config_loader.py         #   YAML 설정 로더
│       └── helpers.py               #   헬퍼 함수
│
├── scripts/                         # 실행 스크립트
│   ├── run_data_collection.py       #   데이터 수집 실행
│   ├── run_preprocessing.py         #   전처리 실행
│   ├── run_backtesting.py           #   백테스팅 실행
│   ├── run_analysis.py              #   분석 및 리포트 실행
│   ├── run_full_pipeline.py         #   전체 파이프라인 실행
│   └── run_live_trading.py          #   실시간 트레이딩 실행
│
├── data/                            # 데이터 저장소
│   ├── raw/                         #   원본 데이터
│   ├── processed/                   #   전처리된 데이터
│   └── features/                    #   피처 데이터
│
├── models/                          # 학습된 모델 저장소
├── results/                         # 백테스팅 결과
├── reports/                         # 생성된 리포트
├── visualizations/                  # 생성된 차트
├── tests/                           # 테스트
├── SPEC.md                          # 상세 기술 명세서
├── setup.py                         # 패키지 설정
├── requirements.txt                 # 의존성 패키지
└── .gitignore
```

---

## 환경 설정

### 요구사항

- Python 3.6+
- (실시간 트레이딩 시) Windows OS, 키움증권 Open API+, PyQt5

### 설치

```bash
# 저장소 클론
git clone https://github.com/m1nd0322/algoTrade.git
cd algoTrade

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 의존성 설치
pip install -r requirements.txt

# 또는 패키지로 설치
pip install -e .
```

---

## 사용법

### 1. 데이터 수집

```bash
python scripts/run_data_collection.py \
    --tickers AAPL MSFT GOOGL TSLA \
    --start-date 2020-01-01 \
    --end-date 2023-12-31
```

### 2. 데이터 전처리

```bash
python scripts/run_preprocessing.py \
    --input-dir data/raw \
    --output-dir data/processed
```

### 3. 백테스팅

```bash
# 특정 전략
python scripts/run_backtesting.py \
    --strategies momentum mean_reversion \
    --tickers AAPL MSFT

# 전체 전략
python scripts/run_backtesting.py --strategies all --tickers AAPL
```

### 4. 분석 및 리포트

```bash
python scripts/run_analysis.py \
    --results-dir results/backtests \
    --format all
```

### 5. 전체 파이프라인 (수집 ~ 리포트 일괄 실행)

```bash
python scripts/run_full_pipeline.py \
    --tickers AAPL MSFT GOOGL \
    --strategies all
```

### 6. 실시간 트레이딩 (키움증권)

```bash
# 모의투자 모드 (권장)
python scripts/run_live_trading.py --paper-trading \
    --strategies momentum mean_reversion \
    --tickers 005930 000660 035420 \
    --capital 10000000

# 실제 트레이딩 (주의!)
python scripts/run_live_trading.py --live \
    --strategies momentum \
    --tickers 005930 \
    --capital 5000000
```

---

## 트레이딩 전략

### 전통 퀀트 (6종)

| 전략 | 설명 |
|------|------|
| Buy & Hold | 매수 후 보유, 벤치마크 전략 |
| Momentum | 추세 추종, 최근 상승 종목 매수 |
| Mean Reversion | 평균 회귀, 과매도 시 매수 |
| Absolute Momentum | 절대 모멘텀 (vs 무위험 수익률) |
| Relative Momentum | 상대 모멘텀 (종목 간 순위 비교) |
| Value Investing | 가치투자 (PER/PBR 기반) |

### 머신러닝 (4종)

| 전략 | 설명 |
|------|------|
| KNN | K-최근접 이웃 분류기 |
| Random Forest | 랜덤 포레스트 앙상블 |
| XGBoost | 그래디언트 부스팅 |
| Clustering | 비지도 클러스터링 기반 |

### 딥러닝 (5종)

| 전략 | 설명 |
|------|------|
| CNN Candlestick | CNN 캔들차트 패턴 인식 |
| LSTM Direction | LSTM 주가 방향 예측 |
| GRU | GRU 시계열 예측 |
| Autoencoder | 오토인코더 이상 탐지 |
| Transformer | 어텐션 기반 시계열 예측 |

---

## 백테스팅 성과 지표

엔진이 산출하는 주요 지표:

- **수익률**: 총 수익률, 연간 수익률, 월간 수익률
- **리스크**: 변동성, 최대 낙폭(MDD), VaR, CVaR
- **리스크 조정 수익률**: 샤프 비율, 소르티노 비율, 칼마 비율
- **트레이딩**: 승률, 손익비, 프로핏 팩터, 평균 보유일수

---

## 리스크 관리 (실시간 트레이딩)

| 항목 | 설명 |
|------|------|
| 포지션 한도 | 종목별 최대 투자 비율 제한 |
| 일일 손실 한도 | 일일 손실 금액/비율 제한 |
| 최대 낙폭 제한 | 전체 자산 대비 최대 낙폭 제한 |
| 주문 크기 제한 | 주문당 최대 수량/금액 제한 |
| 서킷 브레이커 | 손실 한도 초과 시 자동 거래 중단 |

---

## 데이터 파이프라인

```
데이터 수집 (yfinance)
    ↓
데이터 검증 (결측치, 이상치, 연속성)
    ↓
전처리 (정리, 기술적 지표 생성)
    ↓
정규화 / 표준화
    ↓
학습/검증/테스트 분할 (60/20/20)
    ↓
전략 실행 (시그널 생성)
    ↓
백테스팅 (포트폴리오 시뮬레이션)
    ↓
성과 분석 (30+ 지표)
    ↓
시각화 및 리포트 생성
```

---

## CLI 명령어 (패키지 설치 후)

```bash
quant-collect      # 데이터 수집
quant-preprocess   # 전처리
quant-backtest     # 백테스팅
quant-analyze      # 분석
quant-pipeline     # 전체 파이프라인
```

---

## 참고 자료

- **도서**: 퀀트 전략을 위한 인공지능 트레이딩
- **상세 명세서**: [SPEC.md](./SPEC.md)
- **키움 Open API**: Windows 환경에서만 실행 가능

---

## 라이선스

MIT License
