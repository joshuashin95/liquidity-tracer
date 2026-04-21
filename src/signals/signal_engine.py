import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_RAW, DATA_PROCESSED, VOLUME_SURGE_THRESHOLD, INSTITUTIONAL_THRESHOLD


def load_cluster_map() -> pd.Series:
    path = DATA_PROCESSED / "cluster_map.csv"
    return pd.read_csv(path, index_col=0).squeeze()


def compute_volume_surge(ticker: str, window: int = 20) -> pd.Series:
    path = DATA_RAW / f"{ticker}_ohlcv.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    vol = df["거래량"]
    avg = vol.rolling(window).mean()
    return vol / avg  # surge ratio


def compute_net_institutional(ticker: str) -> pd.Series:
    path = DATA_RAW / f"{ticker}_investor.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    # 기관 + 외국인 합산 순매수 비율
    if "기관합계" in df.columns and "외국인합계" in df.columns:
        net = df["기관합계"] + df["외국인합계"]
        total = df.abs().sum(axis=1).replace(0, np.nan)
        return net / total
    return pd.Series(dtype=float)


def build_cluster_signals(cluster_map: pd.Series) -> pd.DataFrame:
    records = []
    for cluster_id in cluster_map.unique():
        tickers = cluster_map[cluster_map == cluster_id].index.tolist()

        surge_list, inst_list = [], []
        for t in tickers:
            surge = compute_volume_surge(t)
            inst = compute_net_institutional(t)
            if not surge.empty:
                surge_list.append(surge)
            if not inst.empty:
                inst_list.append(inst)

        if not surge_list:
            continue

        avg_surge = pd.concat(surge_list, axis=1).mean(axis=1)
        avg_inst = pd.concat(inst_list, axis=1).mean(axis=1) if inst_list else pd.Series(0, index=avg_surge.index)

        df = pd.DataFrame({
            "cluster": cluster_id,
            "volume_surge": avg_surge,
            "net_institutional": avg_inst,
        })
        records.append(df)

    if not records:
        raise ValueError("No signal data built.")

    signals = pd.concat(records).reset_index()
    signals.rename(columns={signals.columns[0]: "date"}, inplace=True)
    signals["energy_signal"] = signals["volume_surge"] >= VOLUME_SURGE_THRESHOLD
    has_investor_data = signals["net_institutional"].abs().sum() > 0
    signals["quality_signal"] = signals["net_institutional"] >= INSTITUTIONAL_THRESHOLD if has_investor_data else True
    signals["combined_signal"] = signals["energy_signal"] & signals["quality_signal"]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(DATA_PROCESSED / "cluster_signals.parquet", index=False)
    return signals


if __name__ == "__main__":
    cluster_map = load_cluster_map()
    print("Building signals...")
    signals = build_cluster_signals(cluster_map)
    print(f"Signals built: {len(signals)} rows")
