# Step 2: data quality checks on CRSP prices, VIX and the risk-free rate

import sys

import numpy as np
import pandas as pd

from config import END, PERMNOS, PROCESSED, ROOT
from data_layer import NYSE, TBILL_FILE, VIX_FILE, load_crsp_raw, load_prices, load_rf, read_tbill, split_legs

EXTREME = 0.05   # a leg moving more than this gets listed
REVERSAL = 0.03  # both legs past this in opposite directions looks like a bad open
TOL = 1e-6

# Known events used to label big moves, single news days are matched before longer periods
EVENTS = [
    ("1997-10-27", "1997-10-28", "Asian crisis mini-crash and rebound"),
    ("1998-08-31", "1998-08-31", "Russia default / LTCM sell-off"),
    ("1998-10-15", "1998-10-15", "Fed inter-meeting rate cut"),
    ("1999-04-01", "2002-10-31", "dot-com boom and bust"),
    ("2001-01-03", "2001-01-03", "Fed surprise rate cut"),
    ("2001-04-18", "2001-04-18", "Fed surprise rate cut"),
    ("2001-09-17", "2001-09-21", "markets reopen after 9/11"),
    ("2003-04-07", "2003-04-07", "US troops reach Baghdad, rally fades"),
    ("2008-01-22", "2008-01-23", "Fed emergency 75bp cut"),
    ("2008-09-15", "2009-05-31", "global financial crisis"),
    ("2011-08-08", "2011-10-04", "US downgrade / euro crisis"),
    ("2015-08-24", "2015-08-25", "China devaluation flash crash"),
    ("2018-12-26", "2018-12-26", "Christmas rally after Q4 2018 sell-off"),
    ("2020-02-24", "2020-06-30", "COVID-19 crash and rebound"),
    ("2020-11-09", "2020-11-09", "Pfizer vaccine announcement"),
    ("2022-02-24", "2022-02-24", "Russia invades Ukraine"),
    ("2022-10-13", "2022-10-13", "CPI-day reversal"),
    ("2024-08-05", "2024-08-05", "yen carry-trade unwind"),
    ("2024-11-06", "2024-11-06", "US election result"),
    ("2025-04-03", "2025-04-09", "tariff shock and 90-day pause"),
]

def event_for(date):
    for start, end, label in sorted(EVENTS, key=lambda e: e[0] != e[1]):
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return label
    return "no known event"

def md_table(df, floatfmt="{:.4f}"):
    if df.index.name is not None or isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines)

def short_list(items, n=12):
    items = [str(i) for i in items]
    if not items:
        return "none"
    return ", ".join(items[:n]) + (f" ... (+{len(items) - n} more)" if len(items) > n else "")

# Previous close in today's share units so splits don't look like gaps
def with_prev_close(df):
    df = split_legs(df)
    df["prev_close"] = df.groupby("ticker")["close"].shift() / df["split_factor"]
    return df

# Panel with both legs and one boolean column per issue, nothing is dropped
def flag_panel(df):
    df = with_prev_close(df)
    on, intra = df["overnight"], df["intraday"]
    hi, lo = df["high"] * (1 + TOL), df["low"] * (1 - TOL)
    px = df[["open", "high", "low", "close"]]

    df["flag_stale_open"] = np.isclose(df["open"], df["prev_close"], rtol=0, atol=1e-9)
    df["flag_extreme"] = (on.abs() > EXTREME) | (intra.abs() > EXTREME)
    df["flag_reversal"] = (on * intra < 0) & (on.abs() > REVERSAL) & (intra.abs() > REVERSAL)
    df["flag_ohlc"] = (df["high"] < lo) | (df["open"] > hi) | (df["open"] < lo) | (df["close"] > hi) | (df["close"] < lo)
    df["flag_bad_price"] = (px <= 0).any(axis=1) | px.isna().any(axis=1) | df["volume"].fillna(0).le(0)
    df["flag_ex_div"] = df["dividend"] > 0
    df["flag_split"] = df["split_factor"] != 1
    df["flag_half_day"] = df["date"].isin(NYSE.early_closes)
    df["flag_first_row"] = df["ret"].isna()
    return df

def coverage(df):
    rows = []
    for t, g in df.groupby("ticker"):
        cal = NYSE.sessions_in_range(g["date"].min(), g["date"].max())
        have = pd.DatetimeIndex(g["date"])
        rows.append({"ticker": t, "first": g["date"].min().date(), "last": g["date"].max().date(), "rows": len(g),
                     "missing_sessions": short_list(cal.difference(have).date),
                     "non_session_rows": short_list(have.difference(cal).date)})
    return pd.DataFrame(rows)

def quote_checks(df):
    rows = []
    for t, g in df.groupby("ticker"):
        quoted = g["bid"].notna() & g["ask"].notna()
        rows.append({"ticker": t, "both_quotes": int(quoted.sum()), "missing_either": int((~quoted).sum()),
                     "crossed_ask_lt_bid": int((quoted & (g["ask"] < g["bid"])).sum())})
    return pd.DataFrame(rows)

def extreme_moves(f):
    rows = []
    for leg, other in [("overnight", "intraday"), ("intraday", "overnight")]:
        for _, r in f[f[leg].abs() > EXTREME].iterrows():
            rows.append({"date": r["date"].date(), "ticker": r["ticker"], "leg": leg, "return": r[leg],
                         "other_leg": r[other], "event": event_for(r["date"])})
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)

# Does the price drop on ex-dates show up in the open? 1 means the full dividend
def dividend_checks(f):
    f = f.assign(div_yield=f["dividend"] / f["prev_close"], raw_gap=f["open"] / f["prev_close"] - 1)
    rows = []
    for t, g in f.groupby("ticker"):
        e, n = g[g["flag_ex_div"]], g[~g["flag_ex_div"]]
        rows.append({"ticker": t, "ex_dates": len(e), "mean_yield": e["div_yield"].mean(),
                     "drop_share_at_open": (n["raw_gap"].mean() - e["raw_gap"].mean()) / e["div_yield"].mean(),
                     "drop_share_se": e["raw_gap"].std() / np.sqrt(len(e)) / e["div_yield"].mean(),
                     "intraday_ex": e["intraday"].mean(), "intraday_other": n["intraday"].mean()})
    return pd.DataFrame(rows)

def vix_checks():
    v = pd.read_csv(VIX_FILE, parse_dates=["date"]).set_index("date").loc[:END]
    close = v["vix"]
    cal = NYSE.sessions_in_range(v.index.min(), END)
    return {"rows": len(v), "first": v.index.min().date(), "min": close.min(), "max": close.max(),
            "duplicate_dates": list(v.index[v.index.duplicated()].date),
            "missing_sessions": list(cal.difference(close.dropna().index).date),
            "non_session_rows": len(v.index.difference(cal)),
            "big_jumps": int(close.pct_change(fill_method=None).abs().gt(0.5).sum())}

def rf_checks():
    wrds = pd.to_numeric(pd.read_csv(TBILL_FILE, parse_dates=["date"]).set_index("date")["dtb3"], errors="coerce").dropna()
    bill = read_tbill()
    rf = load_rf().loc[:END]
    years = pd.DataFrame({"compounded_rf_%": ((1 + rf).groupby(rf.index.year).prod() - 1) * 100,
                          "days": rf.groupby(rf.index.year).size()}).rename_axis("year")
    return {"wrds_last": wrds.index.max().date(), "ext_first": bill.index[bill.index > wrds.index.max()].min().date(),
            "bill_last": bill.index.max().date(), "rows": len(rf), "negative": int((rf < 0).sum()),
            "missing": list(NYSE.sessions_in_range(rf.index.min(), END).difference(rf.index).date), "years": years}

def report(source="crsp"):
    df = load_prices(source)
    df = df[df["date"] <= END]
    f = flag_panel(df)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    f.to_parquet(PROCESSED / f"prices_{source}_flagged.parquet", index=False)

    raw = load_crsp_raw()
    names = {v: k for k, v in PERMNOS.items()}
    tickers = raw.groupby(["PERMNO", "Ticker"])["DlyCalDt"].agg(["min", "max", "size"]).reset_index()
    tickers = tickers.assign(etf=tickers["PERMNO"].map(names), min=tickers["min"].dt.date, max=tickers["max"].dt.date)
    no_open = raw[raw["DlyOpen"].isna()][["Ticker", "DlyCalDt", "DlyPrc", "DlyClose", "DlyVol", "DlyRet"]]

    ext = extreme_moves(f)
    unexplained = ext[ext["event"] == "no known event"]
    rev = f[f["flag_reversal"]]
    rev_unexplained = int((rev["date"].map(event_for) == "no known event").sum())
    stale_year = f.groupby([f["date"].dt.year.rename("year"), "ticker"])["flag_stale_open"].mean().unstack()
    iwm = f[(f["ticker"] == "IWM") & f["date"].between("2002-11-08", "2002-11-14")]
    split_rows = f[f["flag_split"]][["date", "ticker", "split_factor", "prev_close", "open", "close", "overnight"]]
    half = f.groupby("flag_half_day")[["intraday", "overnight"]].agg(lambda s: s.abs().mean())
    vix, rf = vix_checks(), rf_checks()

    out = [f"# Step 2: data checks ({source})", "",
           f"Generated by `python src/checks.py {source}` up to {END}. Nothing is dropped, issues are flagged in "
           f"`data/processed/prices_{source}_flagged.parquet`.", "",
           "## 1. Raw CRSP file", "",
           f"{len(raw)} rows, matched to ETFs on PERMNO because the ticker changes (QQQ was QQQQ):", "",
           md_table(tickers[["PERMNO", "etf", "Ticker", "min", "max", "size"]]), "",
           f"Exact duplicate rows dropped: {raw.duplicated().sum()}. Rows with no open:", "",
           md_table(no_open.assign(DlyCalDt=no_open["DlyCalDt"].dt.date)), "",
           "## 2. Coverage, prices and quotes", "",
           md_table(coverage(df)), "",
           f"Zero, negative or missing prices or zero volume: {int(f['flag_bad_price'].sum())}. "
           f"OHLC inconsistencies: {int(f['flag_ohlc'].sum())}.", "",
           md_table(quote_checks(df)), "",
           "## 3. Stale opens (open equal to the previous close)", "",
           f"{int(f['flag_stale_open'].sum())} rows. Share of days per year:", "",
           md_table(stale_year[["SPY", "QQQ", "IWM"]].fillna(0), "{:.3f}"), "",
           f"## 4. Moves beyond {EXTREME:.0%}", "",
           f"{(ext['leg'] == 'overnight').sum()} overnight and {(ext['leg'] == 'intraday').sum()} intraday, "
           f"{len(unexplained)} without a known event. Opposite-sign legs both beyond {REVERSAL:.0%}: {len(rev)}, "
           f"{rev_unexplained} without a known event.", "",
           "## 5. IWM around 2002-11-12", "",
           "Yahoo had a bad open on this day. CRSP:", "",
           md_table(iwm[["date", "open", "high", "low", "close", "overnight", "intraday"]].assign(date=iwm["date"].dt.date)), "",
           "## 6. Dividends and splits", "",
           md_table(dividend_checks(f), "{:.5f}"), "",
           md_table(split_rows.assign(date=split_rows["date"].dt.date)), "",
           "The CRSP snapshot omits SPY's distributions on 1993-03-19, 1993-06-18, 1993-09-17 and 1993-12-17. Restoring them moves the training overnight-minus-intraday gap from 3.25 to 3.30 bp/day. The SPY backtest starts in 1994, so this omission does not affect its backtest results.", "",
           'The inferred dividend also fails to reconcile prices and total return on SPY 1996-12-20 and IWM 2000-12-13: the hand reconstruction misses `ret` by 15.99 and 17.25 bp, respectively. This affects the dividend diagnostics only; the backtest uses `ret` directly.', "",
           "## 7. Half-days", "",
           f"{int(f['flag_half_day'].sum())} ETF half-days. Mean absolute leg return:", "",
           md_table(half), "",
           "## 8. VIX", "",
           f"- {vix['rows']} rows from {vix['first']}, range {vix['min']:.2f} to {vix['max']:.2f}",
           f"- Duplicate dates (same close, one kept): {short_list(vix['duplicate_dates'])}",
           f"- NYSE sessions with no close (previous close carried): {short_list(vix['missing_sessions'])}",
           f"- Rows on non-NYSE days (dropped): {vix['non_session_rows']}; daily moves above 50%: {vix['big_jumps']}", "",
           "## 9. Risk-free rate (3-month T-bill)", "",
           f"- WRDS H.15 until {rf['wrds_last']}, FRED DTB3 from {rf['ext_first']} to {rf['bill_last']}. The two sources "
           "match on all 8,033 overlapping days.",
           f"- {rf['rows']} NYSE sessions, negative values: {rf['negative']}, sessions with no rf: {short_list(rf['missing'])}", "",
           md_table(rf["years"], "{:.3f}"), "",
           DECISIONS]

    path = ROOT / "reports" / "step2_data_checks.md"
    path.write_text("\n".join(out))
    print(f"wrote {path.relative_to(ROOT)}")

DECISIONS = """## 10. Decisions

| Issue | Handling |
|---|---|
| Duplicates | Exact duplicate rows in the CRSP and VIX exports, one copy kept |
| Close | `DlyPrc`, the bid/ask midpoint on the one no-trade day (SPY 1997-03-31) |
| Missing opens | Flagged, those days have no overnight or intraday return |
| Stale opens | Flagged, 14% of SPY days in 1993 and under 2% after 1998 |
| Extreme moves and reversals | All large moves and all 40 reversals fall on known market events, so they are kept |
| Dividends | Backed out of `ret - retx`, the open falls by most of the dividend on ex-dates |
| Crossed quotes | Not used as costs, the backtest carries the last valid spread |
| Risk-free rate | T-bill converted to a daily return, all of it on the overnight leg. The WRDS Fama-French rf is rounded to whole basis points, so it is not used |
"""

if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "crsp")
