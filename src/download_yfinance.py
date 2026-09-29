"""Download daily OHLCV, dividends and splits for the ETFs from Yahoo Finance.

Raw (unadjusted) open and close are kept alongside Adj Close so overnight and
intraday returns can be rebuilt with dividends handled explicitly.
"""
import time

import yfinance as yf

from config import RAW, START, TICKERS

yf.config.debug.hide_exceptions = False  # raise instead of returning an empty frame


def fetch(ticker, tries=4):
    for i in range(tries):
        try:
            return yf.Ticker(ticker).history(start=START, auto_adjust=False, actions=True)
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"{ticker}: attempt {i + 1} failed ({e.__class__.__name__}), retrying")
            time.sleep(2 ** (i + 1))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for t in TICKERS:
        df = fetch(t)
        df.index = df.index.tz_localize(None).normalize()
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        out = RAW / f"yf_{t}.parquet"
        df.to_parquet(out)
        print(f"{t}: {len(df)} rows {df.index.min().date()} to {df.index.max().date()} -> {out.name}")


if __name__ == "__main__":
    main()
