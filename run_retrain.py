from src.signals.signal_engine import load_cluster_map, build_cluster_signals
from src.prediction.predictor import load_signals, build_features_labels, train, save_model

cluster_map = load_cluster_map()
signals = build_cluster_signals(cluster_map)
X, y = build_features_labels(signals)
print(f"Samples: {len(X)}, Positive labels: {y.sum()} ({y.mean():.1%})")
model = train(X, y)
save_model(model)
print("Done.")
