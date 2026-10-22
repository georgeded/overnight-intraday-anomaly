import exchange_calendars as xc
import pandas as pd

from config import PERMNOS, PROCESSED, RAW, TICKERS

COLUMNS = ["date", "ticker", "open", "high", "low", "close", "bid", "ask", "ret", "retx", "volume", "dividend", "split_factor"]
CRSP_NAMES = {"DlyCalDt": "date", "DlyOpen": "open", "DlyHigh": "high", "DlyLow": "low", "DlyPrc": "close", "DlyBid": "bid",
              "DlyAsk": "ask", "DlyRet": "ret", "DlyRetx": "retx", "DlyVol": "volume"}
CRSP_FILE = RAW / "crsp_daily.csv"
VIX_FILE = RAW / "wrds_vix.csv"
TBILL_FILE = RAW / "wrds_tbill3m.csv"
TBILL_EXT_FILE = RAW / "fred_dtb3_extension.csv"
TBILL_COLUMNS = ["tb_m3", "tbsm3m", "dtb3", "tb3ms"]  # the rate column name differs between WRDS exports
BILL_DAYS = 91
NYSE = xc.get_calendar("XNYS", start="1990-01-01")

def load_crsp_raw():
    raw = pd.read_csv(CRSP_FILE, parse_dates=["DlyCalDt"])
    return raw[raw["PERMNO"].isin(PERMNOS.values())]

# Matched on PERMNO since tickers change (QQQ was QQQQ for a while)
def load_prices(source="crsp", tickers=TICKERS):
    if source != "crsp":
        raise ValueError(f"source must be 'crsp', got {source!r}")
    raw = load_crsp_raw()
    raw = raw[raw["PERMNO"].isin([PERMNOS[t] for t in tickers])].drop_duplicates().sort_values(["PERMNO", "DlyCalDt"])
    df = raw.rename(columns=CRSP_NAMES)[list(CRSP_NAMES.values())]
    df["ticker"] = raw["PERMNO"].map({v: k for k, v in PERMNOS.items()})

    # the price factor drops on a split's ex-date, and the dividend is backed out of ret - retx
    df["split_factor"] = (raw.groupby("PERMNO")["DlyCumFacPr"].shift() / raw["DlyCumFacPr"]).fillna(1.0)
    df["dividend"] = ((df["ret"] - df["retx"]) * df.groupby("ticker")["close"].shift() / df["split_factor"]).fillna(0.0)
    df.loc[df["dividend"].abs() < 1e-6, "dividend"] = 0.0
    return df[COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)

def split_legs(df):
    return df.assign(intraday=df["close"] / df["open"] - 1, overnight=(1 + df["ret"]) / (df["close"] / df["open"]) - 1)

# FRED DTB3 only after the last WRDS value, the two match on every overlapping day
def read_tbill():
    df = pd.read_csv(TBILL_FILE)
    cols = {c.lower(): c for c in df.columns}
    rate = next(cols[c] for c in TBILL_COLUMNS if c in cols)
    bill = pd.to_numeric(df.set_index(pd.to_datetime(df[cols["date"]]))[rate], errors="coerce").dropna().sort_index()
    ext = pd.read_csv(TBILL_EXT_FILE, parse_dates=["date"]).set_index("date")["dtb3"].dropna()
    return pd.concat([bill, ext[ext.index > bill.index.max()]]).sort_index()

# Bill price is 1 - d * 91/360, compounded per calendar day from the rate known at the previous close
def load_rf():
    bill = read_tbill()
    sessions = NYSE.sessions_in_range(bill.index.min(), bill.index.max())
    d = bill.reindex(sessions.union(bill.index)).ffill().reindex(sessions) / 100
    per_day = (1 / (1 - d * BILL_DAYS / 360)) ** (1 / BILL_DAYS)
    rf = per_day.shift() ** sessions.to_series().diff().dt.days - 1
    return rf.dropna().rename("rf").rename_axis("date")

def load_vix():
    vix = pd.read_csv(VIX_FILE, parse_dates=["date"]).set_index("date")["vix"]
    vix = vix.groupby(level=0).first().dropna()
    return vix.reindex(NYSE.sessions_in_range(vix.index.min(), vix.index.max())).ffill().rename("vix").rename_axis("date")

def save_processed(df, name):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path

def load_processed(name):
    return pd.read_parquet(PROCESSED / f"{name}.parquet")
