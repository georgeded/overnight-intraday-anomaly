# Loads raw data into one standard daily price panel

import numpy as np
import pandas as pd

from config import PERMNOS, PROCESSED, RAW, TICKERS

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


# Add overnight (previous close to open) and intraday (open to close) returns
def split_legs(df):
    df = df.copy()
    df["intraday"] = df["close"] / df["open"] - 1
    df["overnight"] = (1 + df["ret"]) / (df["close"] / df["open"]) - 1
    return df


# FRED DTB3, annual % on a discount basis
def load_tbill():
    return pd.read_parquet(RAW / "tbill3m_fred.parquet")["DTB3"].dropna()


# VIX close per NYSE session, CBOE first, then FRED, then the last close
# CBOE open/high/low are unreliable before 2006 so only the close is used
def load_vix():
    import exchange_calendars as xc

    cboe = pd.read_parquet(RAW / "vix_cboe.parquet")["close"]
    fred = pd.read_parquet(RAW / "vix_fred.parquet")["VIXCLS"].dropna()
    sessions = xc.get_calendar("XNYS", start="1990-01-01").sessions_in_range(cboe.index.min(), cboe.index.max())

    vix = cboe.reindex(sessions).fillna(fred.reindex(sessions)).ffill()
    vix.index.name = "date"
    return vix.rename("vix")


def save_processed(df, name):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_processed(name):
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


# Yahoo prices are split-adjusted back in time, undo that so they match CRSP
# Close is split-adjusted only and Adj Close also has dividends, so ret comes
# from Adj Close and retx from Close
def _yfinance(ticker):
    raw = pd.read_parquet(RAW / f"yf_{ticker}.parquet").sort_index()
    split = raw["stock_splits"].replace(0, 1.0).fillna(1.0)

    # splits still to come after each date, e.g. 2.0 before a 2:1 split
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


# Legacy CRSP daily file (crsp.dsf) from download_wrds.py, not run on real data yet
# prc, askhi and bidlo are negative when CRSP filled them from quotes, signs are
# dropped here and checks.py flags those days
def _crsp(tickers):
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

    # cfacpr changes on a split's ex-date, the ratio gives the split
    prev_cfac = raw.groupby("permno")["cfacpr"].shift()
    df["split_factor"] = (prev_cfac / raw["cfacpr"]).fillna(1.0)

    # dsf has no dividend amount, back it out of ret - retx
    prev_close = df.groupby("ticker")["close"].shift()
    df["dividend"] = ((df["ret"] - df["retx"]) * prev_close / df["split_factor"]).fillna(0.0)
    df.loc[df["dividend"].abs() < 1e-6, "dividend"] = 0.0
    return df.reset_index(drop=True)
