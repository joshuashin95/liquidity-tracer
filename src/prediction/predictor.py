import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from pathlib import Path
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_PROCESSED

LOOKAHEAD = 5  # predict if cluster surges within next N days
SURGE_RETURN_THRESHOLD = 0.05  # 5% return = "surge"


def load_signals() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "cluster_signals.parquet")


def build_features_labels(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    signals = signals.sort_values(["cluster", "date"]).copy()
    signals["date"] = pd.to_datetime(signals["date"])

    feature_cols = ["volume_surge", "net_institutional"]
    frames = []

    for cluster_id, grp in signals.groupby("cluster"):
        grp = grp.set_index("date").sort_index()
        features = grp[feature_cols].copy()

        # Rolling stats as additional features
        features["surge_ma5"] = grp["volume_surge"].rolling(5).mean()
        features["inst_ma5"] = grp["net_institutional"].rolling(5).mean()

        # Label: does this cluster have combined_signal in next LOOKAHEAD days?
        label = grp["combined_signal"].rolling(LOOKAHEAD).max().shift(-LOOKAHEAD)
        label = label.fillna(0).astype(int)

        df = features.copy()
        df["label"] = label
        df["cluster"] = cluster_id
        frames.append(df)

    all_data = pd.concat(frames).dropna()
    X = all_data[["volume_surge", "net_institutional", "surge_ma5", "inst_ma5"]]
    y = all_data["label"]
    return X, y


def train(X: pd.DataFrame, y: pd.Series) -> GradientBoostingClassifier:
    if y.nunique() < 2:
        print("WARNING: only one class in labels — check signal thresholds or data coverage.")
        print(f"  Label distribution: {y.value_counts().to_dict()}")

    tscv = TimeSeriesSplit(n_splits=5)
    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if y_tr.nunique() < 2:
            continue
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        print(f"Fold {fold+1}:\n{classification_report(y_val, preds, zero_division=0)}")

    model.fit(X, y)
    return model


def save_model(model: GradientBoostingClassifier):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(DATA_PROCESSED / "model.pkl", "wb") as f:
        pickle.dump(model, f)


def load_model() -> GradientBoostingClassifier:
    with open(DATA_PROCESSED / "model.pkl", "rb") as f:
        return pickle.load(f)


def predict_next(signals: pd.DataFrame, model: GradientBoostingClassifier) -> pd.DataFrame:
    latest = signals.sort_values("date").groupby("cluster").last().reset_index()

    latest["surge_ma5"] = signals.sort_values("date").groupby("cluster")["volume_surge"].last()
    latest["inst_ma5"] = signals.sort_values("date").groupby("cluster")["net_institutional"].last()

    feature_cols = ["volume_surge", "net_institutional", "surge_ma5", "inst_ma5"]
    X = latest[feature_cols].fillna(0)
    latest["surge_prob"] = model.predict_proba(X)[:, 1]

    return latest[["cluster", "volume_surge", "net_institutional", "surge_prob"]].sort_values(
        "surge_prob", ascending=False
    )


if __name__ == "__main__":
    signals = load_signals()
    print("Building features...")
    X, y = build_features_labels(signals)
    print(f"Training on {len(X)} samples, {y.sum()} positive labels...")
    model = train(X, y)
    save_model(model)
    print("Model saved.")

    results = predict_next(signals, model)
    print("\nTop clusters by surge probability:")
    print(results.head(10))
