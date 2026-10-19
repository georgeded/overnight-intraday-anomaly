# Step 1: record source, coverage and SHA-256 of every raw file

import hashlib
from datetime import datetime, timezone

import pandas as pd

from config import RAW

# file: (source, description, date column, first date it must cover)
SOURCES = {
    "crsp_daily.csv": ("WRDS CRSP", "CRSP v2 daily, PERMNO 84398 SPY, 86755 QQQ, 88222 IWM", "DlyCalDt", "1993-01-29"),
    "wrds_vix.csv": ("WRDS CBOE", "VIX index, vix column = close", "date", "1993-01-04"),
    "wrds_tbill3m.csv": ("WRDS FRB H.15", "3-month T-bill, annual % discount basis", "date", "1993-01-01"),
    "fred_dtb3_extension.csv": ("FRED", "DTB3 for 2025-02-12 to 2025-12-31, after WRDS ends", "date", "2025-02-12"),
}

def describe(name):
    path = RAW / name
    source, series, date_col, _ = SOURCES[name]
    dates = pd.read_csv(path, parse_dates=[date_col])[date_col]
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {"file": name, "source": source, "series": series, "downloaded_utc": mtime.strftime("%Y-%m-%d %H:%M"),
            "rows": len(dates), "first_date": dates.min().date(), "last_date": dates.max().date(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    man = pd.DataFrame([describe(f) for f in SOURCES if (RAW / f).exists()])
    man.to_csv(RAW / "manifest.csv", index=False)
    print(man.to_string(index=False))

    missing = [f for f in SOURCES if f not in set(man["file"])]
    late = [r.file for r in man.itertuples() if pd.Timestamp(r.first_date) > pd.Timestamp(SOURCES[r.file][3])]
    print("missing files:", missing or "none")
    print("starting later than required:", late or "none")

if __name__ == "__main__":
    main()
