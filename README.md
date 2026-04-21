# Liquidity Tracer

한국 증시 테마별 자금 흐름(Sector Rotation) 추적 및 예측 알고리즘

## Overview

한국 증시 특유의 테마별 순환매 패턴을 정량적으로 포착하고, 자금이 이동하는 궤적을 추적하여 **차순위 급등 섹터를 예측**한다.

### Core Hypothesis
> 거래 대금의 급증(Energy)과 외인/기관의 수급 집중(Quality)은 섹터 이동의 선행 지표가 된다.

## Architecture

```
Data Collection (PyKRX)
        ↓
Dynamic Theme Clustering (Correlation-based Agglomerative)
        ↓
Signal Engine (Volume Surge + Net Institutional)
        ↓
Prediction Model (Gradient Boosting + TimeSeriesSplit CV)
        ↓
Streamlit Dashboard + Alert
```

## Key Features

- **Dynamic Clustering** — 고정 업종 대신 실제 주가 상관관계로 테마 자동 분류
- **Energy Signal** — 거래대금 급증 비율 (20일 평균 대비 2배 이상)
- **Quality Signal** — 외인+기관 합산 순매수 비율
- **Backtested** — 2022~2024 (3년), TimeSeriesSplit으로 미래 데이터 누수 차단

## Setup

```bash
conda create -n liquidity-tracer python=3.12.3 -y
conda activate liquidity-tracer
pip install pykrx finance-datareader pandas numpy scikit-learn streamlit plotly
```

## Usage

```bash
# Full pipeline (data collection takes several hours for all tickers)
python run_pipeline.py

# Dashboard only (after pipeline completes)
streamlit run src/dashboard/app.py
```

## Project Structure

```
liquidity-tracer/
├── config.py                         # Parameters (dates, thresholds)
├── run_pipeline.py                   # End-to-end runner
├── src/
│   ├── collector/krx_collector.py    # KRX data ingestion
│   ├── clustering/correlation_cluster.py
│   ├── signals/signal_engine.py
│   ├── prediction/predictor.py
│   └── dashboard/app.py
└── data/
    ├── raw/                          # Per-ticker parquet cache
    └── processed/                    # Clusters, signals, model
```

## Data Sources

| Source | Usage |
|--------|-------|
| PyKRX | OHLCV, 외인/기관 순매수 |
| FinanceDataReader | 보조 가격 데이터 |
| 증권사 API (예정) | 실시간 수급 + 알림 |
