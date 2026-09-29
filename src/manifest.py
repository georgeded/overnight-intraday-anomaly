"""Step 1: record what is in data/raw (source, download date, coverage, checksum).

Writes data/raw/manifest.csv, which is tracked in git even though the data files
are not, so a report can always say exactly which snapshot it was run on.
"""
import hashlib
from datetime import datetime, timezone

import pandas as pd

from config import RAW

SOURCES = {
    "yf_SPY.parquet": ("Yahoo Finance via yfinance", "SPY daily, auto_adjust=False, actions=True"),
    "yf_QQQ.parquet": ("Yahoo Finance via yfinance", "QQQ daily, auto_adjust=False, actions=True"),
    "yf_IWM.parquet": ("Yahoo Finance via yfinance", "IWM daily, auto_adjust=False, actions=True"),
    "vix_cboe.parquet": ("CBOE", "cdn.cboe.com VIX_History.csv"),
    "vix_fred.parquet": ("FRED", "VIXCLS"),
    "tbill3m_fred.parquet": ("FRED", "DTB3 (3-month T-bill, secondary market, % p.a.)"),
    "crsp_dsf.parquet": ("WRDS CRSP", "crsp.dsf (legacy SIZ daily stock file)"),
    "crsp_dsf_v2.parquet": ("WRDS CRSP", "crsp.dsf_v2 (CIZ daily stock file)"),
}

# first date each file must cover (fund launch dates; VIX and T-bill from SPY start)
REQUIRED = {
    "yf_SPY.parquet": "1993-01-29",
    "yf_QQQ.parquet": "1999-03-10",
    "yf_IWM.parquet": "2000-05-26",
    "vix_cboe.parquet": "1993-01-04",
    "vix_fred.parquet": "1993-01-04",
    "tbill3m_fred.parquet": "1993-01-04",
}


def sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def describe(path):
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    idx = pd.to_datetime(df.index)
    valid = df.notna().any(axis=1).to_numpy()  # FRED rows on holidays are all NaN
    source, detail = SOURCES.get(path.name, ("unknown", ""))
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {
        "file": path.name,
        "source": source,
        "series": detail,
        "downloaded_utc": mtime.strftime("%Y-%m-%d %H:%M"),
        "rows": len(df),
        "rows_with_data": int(valid.sum()),
        "first_date": idx[valid].min().date(),
        "last_date": idx[valid].max().date(),
        "sha256": sha256(path)[:16],
    }


def build():
    rows = [describe(p) for p in sorted(RAW.glob("*.parquet"))]
    man = pd.DataFrame(rows)
    missing = sorted(set(REQUIRED) - set(man["file"]))
    late = [
        r["file"] for r in rows
        if r["file"] in REQUIRED and pd.Timestamp(r["first_date"]) > pd.Timestamp(REQUIRED[r["file"]])
    ]
    return man, missing, late


def main():
    man, missing, late = build()
    man.to_csv(RAW / "manifest.csv", index=False)
    print(man.to_string(index=False))
    print("missing files:", missing or "none")
    print("starting later than required:", late or "none")


if __name__ == "__main__":
    main()
