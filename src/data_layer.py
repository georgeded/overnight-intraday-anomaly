# Loads raw WRDS data into one standard daily price panel

import pandas as pd

from config import PERMNOS, PROCESSED, RAW, TICKERS

COLUMNS = [
    "date", "ticker", "open", "high", "low", "close", "bid", "ask",
    "ret", "retx", "volume", "dividend", "split_factor",
]

CRSP_FILE = RAW / "crsp_daily.csv"
VIX_FILE = RAW / "wrds_vix.csv"
RF_FILE = RAW / "ff_rf_daily.csv"
TBILL_FILE = RAW / "wrds_tbill3m.csv"
BILL_DAYS = 91  # the 3-month bill is a 13-week bill


def load_prices(source="crsp", tickers=TICKERS):
    if source != "crsp":
        raise ValueError(f"source must be 'crsp', got {source!r}")

    df = _crsp(tickers)
    return df[COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


# Raw CRSP rows for our three PERMNOs before any cleaning, used by the data checks
def load_crsp_raw():
    raw = pd.read_csv(CRSP_FILE, parse_dates=["DlyCalDt"])
    return raw[raw["PERMNO"].isin(PERMNOS.values())]


# Add overnight (previous close to open) and intraday (open to close) returns
def split_legs(df):
    df = df.copy()
    df["intraday"] = df["close"] / df["open"] - 1
    df["overnight"] = (1 + df["ret"]) / (df["close"] / df["open"]) - 1
    return df


# Daily risk-free return per NYSE session, from the previous session's close to this one.
# Uses the 3-month T-bill (wrds_tbill3m.csv) when present, else the Fama-French daily rf,
# which WRDS rounds to 4 decimals (whole basis points per day) and is too coarse.
def load_rf():
    if TBILL_FILE.exists():
        return tbill_to_daily(read_tbill())

    import warnings
    warnings.warn("wrds_tbill3m.csv not found: using the rounded Fama-French rf", stacklevel=2)
    rf = pd.read_csv(RF_FILE, parse_dates=["date"]).set_index("date")["rf"]
    return rf.dropna()


# The T-bill file from the download script or a WRDS web query: a date column and one rate
# column. Column names differ between the two, so match them case-insensitively.
TBILL_COLUMNS = ["tb_m3", "tbsm3m", "dtb3", "tb3ms"]


def read_tbill():
    df = pd.read_csv(TBILL_FILE)
    cols = {c.lower(): c for c in df.columns}
    date = next((cols[c] for c in ["date", "caldt"] if c in cols), None)
    rate = next((cols[c] for c in TBILL_COLUMNS if c in cols), None)
    if date is None or rate is None:
        raise SystemExit(f"{TBILL_FILE.name}: need a date column and one of {TBILL_COLUMNS}, got {list(df.columns)}")
    bill = df.set_index(pd.to_datetime(df[date]))[rate]
    return pd.to_numeric(bill, errors="coerce").dropna().sort_index()


# T-bill rate (annual %, bank-discount basis) -> return per NYSE session.
# Price of the bill = 1 - d * 91/360, its 91-day return is 1/price - 1, compounded per calendar
# day so a Friday-to-Monday period earns 3 days. The rate known at the previous close applies;
# bond-market holidays (NYSE open) carry the last rate forward.
def tbill_to_daily(bill):
    import exchange_calendars as xc

    sessions = xc.get_calendar("XNYS", start="1990-01-01").sessions_in_range(bill.index.min(), bill.index.max())
    d = bill.reindex(sessions.union(bill.index)).ffill().reindex(sessions) / 100
    per_day = (1 / (1 - d * BILL_DAYS / 360)) ** (1 / BILL_DAYS)
    days = sessions.to_series().diff().dt.days
    rf = per_day.shift() ** days - 1
    rf.index.name = "date"
    return rf.dropna().rename("rf")


# VIX close per NYSE session, gaps filled with the last close
def load_vix():
    import exchange_calendars as xc

    vix = pd.read_csv(VIX_FILE, parse_dates=["date"]).set_index("date")["vix"].dropna()
    sessions = xc.get_calendar("XNYS", start="1990-01-01").sessions_in_range(vix.index.min(), vix.index.max())

    vix = vix.reindex(sessions).ffill()
    vix.index.name = "date"
    return vix.rename("vix")


def save_processed(df, name):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_processed(name):
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


# CRSP v2 daily file, matched on PERMNO only since tickers change (QQQ was QQQQ)
def _crsp(tickers):
    names = {v: k for k, v in PERMNOS.items()}
    raw = load_crsp_raw()
    raw = raw[raw["PERMNO"].isin([PERMNOS[t] for t in tickers])]

    # the export repeats a few rows exactly, keep one of each
    raw = raw.drop_duplicates().sort_values(["PERMNO", "DlyCalDt"])

    df = pd.DataFrame({"date": raw["DlyCalDt"], "ticker": raw["PERMNO"].map(names)})
    df["open"] = raw["DlyOpen"]
    df["high"] = raw["DlyHigh"]
    df["low"] = raw["DlyLow"]
    df["close"] = raw["DlyPrc"]  # DlyClose, or the bid/ask midpoint on a day with no trade
    df["bid"] = raw["DlyBid"]
    df["ask"] = raw["DlyAsk"]
    df["ret"] = raw["DlyRet"]
    df["retx"] = raw["DlyRetx"]
    df["volume"] = raw["DlyVol"]

    # the cumulative price factor drops on a split's ex-date, the ratio gives the split
    prev_cfac = raw.groupby("PERMNO")["DlyCumFacPr"].shift()
    df["split_factor"] = (prev_cfac / raw["DlyCumFacPr"]).fillna(1.0)

    # no dividend amount in the file, back it out of ret - retx
    prev_close = df.groupby("ticker")["close"].shift()
    df["dividend"] = ((df["ret"] - df["retx"]) * prev_close / df["split_factor"]).fillna(0.0)
    df.loc[df["dividend"].abs() < 1e-6, "dividend"] = 0.0
    return df.reset_index(drop=True)
