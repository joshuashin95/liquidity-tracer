"""
Full pipeline runner. Run this after environment setup.
순서: collect → cluster → signals → train model
"""
from src.collector.krx_collector import collect_all, get_all_tickers
from src.clustering.correlation_cluster import load_returns, build_clusters, save_clusters
from src.signals.signal_engine import load_cluster_map, build_cluster_signals
from src.prediction.predictor import load_signals, build_features_labels, train, save_model


def main():
    print("=" * 50)
    print("Step 1: Collecting data from KRX...")
    tickers = get_all_tickers()
    print(f"  {len(tickers)} tickers found")
    collect_all(tickers)

    print("\nStep 2: Building correlation clusters...")
    returns = load_returns()
    cluster_map = build_clusters(returns)
    save_clusters(cluster_map)
    print(f"  {cluster_map.nunique()} clusters built")

    print("\nStep 3: Computing signals...")
    cluster_map = load_cluster_map()
    signals = build_cluster_signals(cluster_map)
    print(f"  {len(signals)} signal rows computed")

    print("\nStep 4: Training prediction model...")
    signals = load_signals()
    X, y = build_features_labels(signals)
    model = train(X, y)
    save_model(model)

    print("\nPipeline complete. Run dashboard with:")
    print("  streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
