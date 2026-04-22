from pykrx import stock
from pathlib import Path
import pandas as pd

tickers = [f.stem.replace("_ohlcv", "") for f in Path("data/raw").glob("*_ohlcv.parquet")]
print(f"Fetching names for {len(tickers)} tickers...")

rows = []
for ticker in tickers:
    try:
        name = stock.get_market_ticker_name(ticker)
        rows.append({"ticker": ticker, "name": name})
    except Exception:
        rows.append({"ticker": ticker, "name": ticker})

df = pd.DataFrame(rows)
df.to_csv("data/processed/ticker_names.csv", index=False, encoding="utf-8-sig")
print(f"Saved {len(df)} ticker names.")
print(df.head(10))
