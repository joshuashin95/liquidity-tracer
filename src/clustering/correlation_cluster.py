import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_RAW, DATA_PROCESSED, CORRELATION_WINDOW, N_CLUSTERS, MIN_CLUSTER_SIZE


def load_returns() -> pd.DataFrame:
    frames = []
    for f in DATA_RAW.glob("*_ohlcv.parquet"):
        df = pd.read_parquet(f)[["종가"]].rename(columns={"종가": f.stem.replace("_ohlcv", "")})
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No OHLCV data found. Run collector first.")
    prices = pd.concat(frames, axis=1).sort_index()
    return prices.pct_change().dropna(how="all")


def build_clusters(returns: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.Series:
    recent = returns.tail(CORRELATION_WINDOW)

    # Remove daily cross-sectional mean to neutralize market factor
    market_return = recent.mean(axis=1)
    residuals = recent.sub(market_return, axis=0)

    corr = residuals.corr().fillna(0)
    distance = 1 - corr
    np.fill_diagonal(distance.values, 0)
    distance = distance.clip(lower=0)

    model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    labels = model.fit_predict(distance)

    cluster_map = pd.Series(labels, index=returns.columns, name="cluster")

    # Drop tiny clusters
    valid = cluster_map.value_counts()[cluster_map.value_counts() >= MIN_CLUSTER_SIZE].index
    cluster_map = cluster_map[cluster_map.isin(valid)]

    return cluster_map


def save_clusters(cluster_map: pd.Series):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cluster_map.to_csv(DATA_PROCESSED / "cluster_map.csv")


if __name__ == "__main__":
    print("Loading returns...")
    returns = load_returns()
    print(f"Building clusters from {len(returns.columns)} tickers...")
    cluster_map = build_clusters(returns)
    save_clusters(cluster_map)
    print(f"Saved {cluster_map.nunique()} clusters.")
