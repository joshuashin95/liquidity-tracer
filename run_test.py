"""Quick smoke test with 30 tickers before running full pipeline."""
from src.collector.krx_collector import collect_all, get_all_tickers
from src.clustering.correlation_cluster import load_returns, build_clusters, save_clusters
from src.signals.signal_engine import load_cluster_map, build_cluster_signals
from src.prediction.predictor import load_signals, build_features_labels, train, save_model, predict_next, load_model

TEST_TICKERS = [
    "005930", "000660", "035420", "051910", "006400",  # 삼성전자, SK하이닉스, NAVER, LG화학, 삼성SDI
    "035720", "105560", "055550", "086790", "032830",  # 카카오, KB금융, 신한지주, 하나금융, 삼성생명
    "012330", "009150", "018260", "034020", "096770",  # 현대모비스, 삼성전기, 삼성물산, 두산에너빌리티, SK이노베이션
    "011200", "010950", "003670", "028260", "066570",  # HMM, S-Oil, 포스코퓨처엠, 삼성물산, LG전자
    "000270", "207940", "068270", "326030", "091990",  # 기아, 삼성바이오로직스, 셀트리온, SK바이오팜, 셀트리온헬스케어
    "000810", "011070", "042660", "047050", "064350",  # 삼성화재, LG이노텍, 한화에어로스페이스, 대한항공, 현대로템
]

def main():
    print("Step 1: Collecting test tickers...")
    collect_all(TEST_TICKERS)

    print("\nStep 2: Clustering...")
    returns = load_returns()
    print(f"  Returns shape: {returns.shape}")
    cluster_map = build_clusters(returns, n_clusters=5)
    save_clusters(cluster_map)
    print(f"  {cluster_map.nunique()} clusters")

    print("\nStep 3: Signals...")
    cluster_map = load_cluster_map()
    signals = build_cluster_signals(cluster_map)
    print(f"  {len(signals)} rows, {signals['combined_signal'].sum()} combined signals")

    print("\nStep 4: Training model...")
    signals = load_signals()
    X, y = build_features_labels(signals)
    print(f"  X shape: {X.shape}, positive labels: {y.sum()}")
    model = train(X, y)
    save_model(model)

    print("\nStep 5: Predictions...")
    predictions = predict_next(signals, model)
    print(predictions)

    print("\nSmoke test passed. Run `python run_pipeline.py` for full dataset.")

if __name__ == "__main__":
    main()
