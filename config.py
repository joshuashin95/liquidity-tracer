from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"

START_DATE = "20220101"
END_DATE = "20241231"

# Clustering
CORRELATION_WINDOW = 60   # days for rolling correlation
N_CLUSTERS = 20           # initial number of theme clusters
MIN_CLUSTER_SIZE = 3      # minimum stocks per cluster

# Signals
VOLUME_SURGE_THRESHOLD = 2.0    # x times 20-day avg volume
INSTITUTIONAL_THRESHOLD = 0.05  # net buy ratio threshold
