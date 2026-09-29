"""Download VIX (CBOE, with FRED as a check) and the 3-month T-bill rate (FRED)."""
import pandas as pd

from config import RAW, START

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
CBOE_VIX = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def fred(series):
    df = pd.read_csv(FRED.format(series), parse_dates=[0], index_col=0, na_values=".")
    df.index.name = "date"
    return df.loc[START:]


def main():
    RAW.mkdir(parents=True, exist_ok=True)

    vix = pd.read_csv(CBOE_VIX, parse_dates=["DATE"], index_col="DATE")
    vix.index.name = "date"
    vix.columns = [c.lower() for c in vix.columns]
    vix = vix.loc[START:]
    vix.to_parquet(RAW / "vix_cboe.parquet")
    print(f"VIX (CBOE): {len(vix)} rows {vix.index.min().date()} to {vix.index.max().date()}")

    for series, name in [("VIXCLS", "vix_fred"), ("DTB3", "tbill3m_fred")]:
        df = fred(series)
        df.to_parquet(RAW / f"{name}.parquet")
        print(f"{series}: {df[series].notna().sum()} obs {df.index.min().date()} to {df.index.max().date()}")


if __name__ == "__main__":
    main()
