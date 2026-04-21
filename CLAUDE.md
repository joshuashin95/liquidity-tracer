# Liquidity Tracer — Claude Code Guide

## Project Overview
한국 증시 테마별 자금 흐름(Sector Rotation) 추적 및 예측 알고리즘.
거래대금 급증(Energy)과 외인/기관 수급 집중(Quality)을 선행 지표로 사용하여 차순위 급등 섹터를 예측한다.

## Environment
- Python 3.12.3, conda env: `liquidity-tracer`
- Run commands with: `conda run -n liquidity-tracer python ...`
- Streamlit dashboard: `conda run -n liquidity-tracer streamlit run src/dashboard/app.py`

## Project Structure
```
config.py                          # All parameters (dates, thresholds, cluster settings)
run_pipeline.py                    # Full pipeline runner (collect → cluster → signals → train)
src/
  collector/krx_collector.py       # PyKRX data collection (OHLCV + investor trading)
  clustering/correlation_cluster.py # Agglomerative clustering on return correlations
  signals/signal_engine.py         # Volume surge + net institutional signals per cluster
  prediction/predictor.py          # GradientBoosting classifier with TimeSeriesSplit CV
  dashboard/app.py                 # Streamlit dashboard (predictions + heatmap + timeline)
data/
  raw/        # Per-ticker parquet files ({ticker}_ohlcv.parquet, {ticker}_investor.parquet)
  processed/  # cluster_map.csv, cluster_signals.parquet, model.pkl
```

## Pipeline Order
1. `src/collector/krx_collector.py` — fetch raw data from KRX
2. `src/clustering/correlation_cluster.py` — build dynamic theme clusters
3. `src/signals/signal_engine.py` — compute Energy + Quality signals per cluster
4. `src/prediction/predictor.py` — train and save model
5. `src/dashboard/app.py` — visualize via Streamlit

## Key Design Decisions
- **Dynamic clustering** over fixed sector labels — captures Korean 테마주 rotation patterns
- **Correlation window**: 60 days rolling (config.py `CORRELATION_WINDOW`)
- **Signals**: volume surge ratio ≥ 2.0x 20-day avg (Energy), net institutional ratio ≥ 5% (Quality)
- **Model**: GradientBoostingClassifier with TimeSeriesSplit (no data leakage)
- **Lookahead**: 5 days (`LOOKAHEAD` in predictor.py)
- Data is cached as parquet; re-collection is skipped if file already exists

## KPIs
- Signal precision / recall on held-out 2024 data
- Cost-efficiency: minimize API calls via parquet caching

## What NOT to do
- Do not use train/test split that ignores time order — always use TimeSeriesSplit
- Do not fetch data without the delay (`delay=0.3s`) — KRX rate limits
- Do not hardcode dates — use config.py `START_DATE` / `END_DATE`
