from src.clustering.correlation_cluster import load_returns, build_clusters, save_clusters

returns = load_returns()
print(f"Tickers: {len(returns.columns)}")
cluster_map = build_clusters(returns, n_clusters=5)
save_clusters(cluster_map)
print(f"Clusters: {cluster_map.nunique()}")
print(cluster_map.value_counts().sort_index())
