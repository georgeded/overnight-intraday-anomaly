"""One place that turns raw downloads into a standard daily price panel.

    load_prices("yfinance")  # works now
    load_prices("crsp")      # same schema, from download_wrds.py output (untested until WRDS access)

Every source returns one row per ticker and trading day with these columns:

    date, ticker              trading day (tz-naive), ETF ticker
    open, high, low, close    unadjusted prices, as traded that day
    bid, ask                  closing bid and ask (NaN for yfinance)
    ret                       total return close to close, dividends included
    retx                      return close to close without dividends (splits handled)
    volume                    shares traded, unadjusted
    dividend                  cash dividend per share, on its ex-date (0 otherwise)
    split_factor              shares after / shares before on a split's ex-date (1 otherwise)

Because ret and the prices are on the same unadjusted basis, the two legs of a day
can be split consistently (see split_legs):

    intraday  = close / open - 1
    overnight = (1 + ret) / (close / open) - 1   # previous close to open, incl. dividend
"""
import numpy as np
import pandas as pd

from config import PERMNOS, RAW, TICKERS

COLUMNS = [
    "date", "ticker", "open", "high", "low", "close", "bid", "ask",
    "ret", "retx", "volume", "dividend", "split_factor",
]


def load_prices(source="yfinance", tickers=TICKERS):
    if source == "yfinance":
        df = pd.concat([_yfinance(t) for t in tickers], ignore_index=True)
    elif source == "crsp":
        df = _crsp(tickers)
    else:
        raise ValueError(f"source must be 'yfinance' or 'crsp', got {source!r}")
    return df[COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def split_legs(df):
    """Add overnight (previous close to open) and intraday (open to close) returns."""
    df = df.copy()
    df["intraday"] = df["close"] / df["open"] - 1
    df["overnight"] = (1 + df["ret"]) / (df["close"] / df["open"]) - 1
    return df


def _yfinance(ticker):
    """Yahoo prices come split-adjusted back in time; undo that so they match CRSP.

    Yahoo's Close is adjusted for splits only and Adj Close for splits and dividends,
    so ret comes from Adj Close and retx from Close (both give the same answer
    whether or not the split adjustment is undone).
    """
    raw = pd.read_parquet(RAW / f"yf_{ticker}.parquet").sort_index()
    split = raw["stock_splits"].replace(0, 1.0).fillna(1.0)
    # shares-per-share factor still to come after each date: 2.0 before a 2:1 split, 1 after
    later = split[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)

    df = pd.DataFrame({"date": raw.index, "ticker": ticker})
    for c in ["open", "high", "low", "close"]:
        df[c] = (raw[c] * later).to_numpy()
    df["bid"] = np.nan
    df["ask"] = np.nan
    df["ret"] = raw["adj_close"].pct_change(fill_method=None).to_numpy()
    df["retx"] = raw["close"].pct_change(fill_method=None).to_numpy()
    df["volume"] = (raw["volume"] / later).to_numpy()
    df["dividend"] = (raw["dividends"] * later).to_numpy()
    df["split_factor"] = split.to_numpy()
    return df


def _crsp(tickers):
    """Legacy CRSP daily stock file (crsp.dsf) as saved by download_wrds.py.

    prc is negative when there was no closing trade and CRSP used the bid/ask
    midpoint; askhi/bidlo are likewise negative when they hold the closing ask/bid
    instead of the day's high/low. Signs are dropped here; the checks module flags
    those days. Not yet run against real data.
    """
    raw = pd.read_parquet(RAW / "crsp_dsf.parquet")
    names = {v: k for k, v in PERMNOS.items()}
    raw = raw[raw["permno"].isin([PERMNOS[t] for t in tickers])].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["permno", "date"])

    df = pd.DataFrame({"date": raw["date"], "ticker": raw["permno"].map(names)})
    df["open"] = raw["openprc"].abs()
    df["high"] = raw["askhi"].abs()
    df["low"] = raw["bidlo"].abs()
    df["close"] = raw["prc"].abs()
    df["bid"] = raw["bid"]
    df["ask"] = raw["ask"]
    df["ret"] = pd.to_numeric(raw["ret"], errors="coerce")
    df["retx"] = pd.to_numeric(raw["retx"], errors="coerce")
    df["volume"] = raw["vol"]
    # cfacpr falls on a split's ex-date; the ratio gives the split for that day
    prev_cfac = raw.groupby("permno")["cfacpr"].shift()
    df["split_factor"] = (prev_cfac / raw["cfacpr"]).fillna(1.0)
    # dsf carries no dividend amount; back it out of ret - retx and the previous close
    prev_close = df.groupby("ticker")["close"].shift()
    df["dividend"] = ((df["ret"] - df["retx"]) * prev_close / df["split_factor"]).fillna(0.0)
    df.loc[df["dividend"].abs() < 1e-6, "dividend"] = 0.0
    return df.reset_index(drop=True)
