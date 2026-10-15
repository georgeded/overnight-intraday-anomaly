# Step 1: record what is in data/raw, with full SHA-256 hashes

import hashlib
from datetime import datetime, timezone

import pandas as pd

from config import RAW

SOURCES = {
    "crsp_daily.csv": ("WRDS CRSP", "CRSP v2 daily stock file, PERMNO 84398 SPY, 86755 QQQ, 88222 IWM"),
    "wrds_vix.csv": ("WRDS CBOE", "VIX index, vix column = close"),
    "ff_rf_daily.csv": ("WRDS Fama-French", "daily factors, rf = daily risk-free return in decimals"),
}

# Date column and first date each file must cover
DATE_COL = {"crsp_daily.csv": "DlyCalDt", "wrds_vix.csv": "date", "ff_rf_daily.csv": "date"}
REQUIRED = {"crsp_daily.csv": "1993-01-29", "wrds_vix.csv": "1993-01-04", "ff_rf_daily.csv": "1993-01-04"}


def sha256(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def describe(path):
    df = pd.read_csv(path, parse_dates=[DATE_COL[path.name]])
    dates = df[DATE_COL[path.name]]
    source, detail = SOURCES[path.name]
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    return {
        "file": path.name,
        "source": source,
        "series": detail,
        "downloaded_utc": mtime.strftime("%Y-%m-%d %H:%M"),
        "rows": len(df),
        "first_date": dates.min().date(),
        "last_date": dates.max().date(),
        "sha256": sha256(path),
    }


def build():
    rows = [describe(RAW / f) for f in SOURCES if (RAW / f).exists()]
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
