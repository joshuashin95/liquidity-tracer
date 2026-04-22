import pandas as pd
from pykrx import stock
from pathlib import Path
import time
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATA_RAW, START_DATE, END_DATE


def get_all_tickers(date: str = "20241229") -> list[str]:
    return stock.get_market_ticker_list(date, market="KOSPI") + \
           stock.get_market_ticker_list(date, market="KOSDAQ")


def fetch_ohlcv(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    df = stock.get_market_ohlcv_by_date(start, end, ticker)
    df["ticker"] = ticker
    return df


def fetch_investor_trading(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    df = stock.get_market_trading_value_by_date(start, end, ticker)
    df["ticker"] = ticker
    return df


def collect_all(tickers: list[str] | None = None, delay: float = 0.3):
    if tickers is None:
        tickers = get_all_tickers()

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        ohlcv_path = DATA_RAW / f"{ticker}_ohlcv.parquet"
        inv_path = DATA_RAW / f"{ticker}_investor.parquet"

        if not ohlcv_path.exists():
            try:
                df = fetch_ohlcv(ticker)
                if not df.empty:
                    df.to_parquet(ohlcv_path)
            except Exception as e:
                print(f"[OHLCV] {ticker} failed: {e}")
            time.sleep(delay)

        if not inv_path.exists():
            try:
                df = fetch_investor_trading(ticker)
                if not df.empty:
                    df.to_parquet(inv_path)
            except Exception as e:
                print(f"[INVESTOR] {ticker} failed: {e}")
            time.sleep(delay)


if __name__ == "__main__":
    print("Collecting data for all tickers...")
    collect_all()
    print("Done.")
