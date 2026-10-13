# Step 2: data quality checks on prices, VIX and T-bill

import sys

import exchange_calendars as xc
import numpy as np
import pandas as pd

from config import PROCESSED, RAW, ROOT
from data_layer import load_prices, split_legs

EXTREME = 0.05   # list any leg that moves more than this
REVERSAL = 0.03  # both legs past this in opposite directions, maybe a bad open
TOL = 1e-6       # tolerance for OHLC ordering

NYSE = xc.get_calendar("XNYS", start="1990-01-01")

# Known events used to label big moves
# Single days are news days and get matched before the longer volatile periods
EVENTS = [
    ("1997-10-27", "1997-10-28", "Asian crisis mini-crash and rebound"),
    ("1998-08-31", "1998-08-31", "Russia default / LTCM sell-off"),
    ("1998-10-15", "1998-10-15", "Fed inter-meeting rate cut"),
    ("1999-04-01", "2002-10-31", "dot-com boom and bust (tech volatility)"),
    ("2001-01-03", "2001-01-03", "Fed surprise rate cut"),
    ("2001-04-18", "2001-04-18", "Fed surprise rate cut"),
    ("2001-09-17", "2001-09-21", "markets reopen after 9/11"),
    ("2008-01-22", "2008-01-23", "Fed emergency 75bp cut"),
    ("2008-09-15", "2009-05-31", "global financial crisis"),
    ("2011-08-08", "2011-10-04", "US downgrade / euro crisis"),
    ("2015-08-24", "2015-08-25", "China devaluation flash crash"),
    ("2018-12-26", "2018-12-26", "Christmas rally after Q4 2018 sell-off"),
    ("2020-02-24", "2020-06-30", "COVID-19 crash and rebound"),
    ("2020-11-09", "2020-11-09", "Pfizer vaccine announcement"),
    ("2022-02-24", "2022-02-24", "Russia invades Ukraine"),
    ("2022-10-13", "2022-10-13", "CPI-day reversal"),
    ("2003-04-07", "2003-04-07", "US troops reach Baghdad, rally fades"),
    ("2024-08-05", "2024-08-05", "yen carry-trade unwind"),
    ("2024-11-06", "2024-11-06", "US election result"),
    ("2025-04-03", "2025-04-09", "tariff shock and 90-day pause"),
]


def event_for(date):
    for start, end, label in sorted(EVENTS, key=lambda e: e[0] != e[1]):
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return label
    return "no known event: verify against CRSP"


def sessions(start, end):
    return NYSE.sessions_in_range(start, end)


# Price panel checks

# Per ticker, NYSE sessions with no row and rows on non-session dates
def coverage(df):
    out = []
    for t, g in df.groupby("ticker"):
        cal = sessions(g["date"].min(), g["date"].max())
        have = pd.DatetimeIndex(g["date"])
        out.append({
            "ticker": t, "first": g["date"].min().date(), "last": g["date"].max().date(),
            "rows": len(g), "nyse_sessions": len(cal),
            "missing_sessions": list(cal.difference(have).date),
            "non_session_rows": list(have.difference(cal).date),
        })
    return pd.DataFrame(out)


def duplicates(df):
    return df[df.duplicated(["ticker", "date"], keep=False)]


def bad_prices(df):
    px = df[["open", "high", "low", "close"]]
    return df[(px <= 0).any(axis=1) | px.isna().any(axis=1) | df["volume"].fillna(0).le(0)]


def ohlc_violations(df):
    hi, lo = df["high"] * (1 + TOL), df["low"] * (1 - TOL)
    bad = (df["high"] < lo) | (df["open"] > hi) | (df["open"] < lo) | (df["close"] > hi) | (df["close"] < lo)
    return df[bad]


def with_prev_close(df):
    df = split_legs(df)
    # previous close in today's share units so splits don't look like gaps
    df["prev_close"] = df.groupby("ticker")["close"].shift() / df["split_factor"]
    return df


# Open equal to the previous close, usually a filled-in open rather than a trade
def stale_opens(df):
    df = with_prev_close(df)
    return df[np.isclose(df["open"], df["prev_close"], rtol=0, atol=1e-9)]


def extreme_moves(df):
    df = with_prev_close(df)
    rows = []
    for leg in ["overnight", "intraday"]:
        big = df[df[leg].abs() > EXTREME]
        for _, r in big.iterrows():
            rows.append({"date": r["date"].date(), "ticker": r["ticker"], "leg": leg,
                         "return": r[leg], "other_leg": r["intraday" if leg == "overnight" else "overnight"],
                         "event": event_for(r["date"])})
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


# Big overnight move undone intraday or the reverse, typical of a bad open
def reversals(df):
    df = with_prev_close(df)
    m = (df["overnight"] * df["intraday"] < 0) & (df["overnight"].abs() > REVERSAL) & (df["intraday"].abs() > REVERSAL)
    out = df.loc[m, ["date", "ticker", "prev_close", "open", "close", "overnight", "intraday", "ret"]].copy()
    out["event"] = out["date"].map(event_for)
    return out


# Is each dividend on its ex-date, and does the price drop show up in the open?
# By construction it can only land overnight, so this checks the source data
def dividend_checks(df):
    df = with_prev_close(df)

    df["div_yield"] = df["dividend"] / df["prev_close"]
    df["raw_gap"] = df["open"] / df["prev_close"] - 1
    ex = df["dividend"] > 0

    rows = []
    for t, g in df.groupby("ticker"):
        e, n = g[ex.loc[g.index]], g[~ex.loc[g.index]]
        rows.append({
            "ticker": t,
            "ex_dates": len(e),
            "first_ex": e["date"].min().date(),
            "max_|ret-retx-yield|_ex": ((e["ret"] - e["retx"]) - e["div_yield"]).abs().max(),
            "max_|ret-retx|_other": (n["ret"] - n["retx"]).abs().max(),
            "mean_yield_ex": e["div_yield"].mean(),
            "mean_raw_gap_ex": e["raw_gap"].mean(),
            "mean_raw_gap_other": n["raw_gap"].mean(),
            # share of the dividend seen in the open, 1 means the full drop
            "drop_share_at_open": (n["raw_gap"].mean() - e["raw_gap"].mean()) / e["div_yield"].mean(),
            "drop_share_se": e["raw_gap"].std() / np.sqrt(len(e)) / e["div_yield"].mean(),
            "mean_overnight_ex": e["overnight"].mean(),
            "mean_intraday_ex": e["intraday"].mean(),
            "mean_intraday_other": n["intraday"].mean(),
        })

    per_year = df[ex].groupby([df["ticker"], df["date"].dt.year.rename("year")]).size().unstack(0).fillna(0).astype(int)
    return pd.DataFrame(rows), per_year


def splits(df):
    df = with_prev_close(df)
    return df.loc[df["split_factor"] != 1, ["date", "ticker", "split_factor", "prev_close", "open", "close", "retx", "overnight", "intraday"]]


def half_days(df):
    early = NYSE.early_closes
    early = early[(early >= df["date"].min()) & (early <= df["date"].max())]
    d = split_legs(df)
    d["half_day"] = d["date"].isin(early)
    summary = d.groupby("half_day")[["intraday", "overnight"]].agg(lambda s: s.abs().mean())
    return pd.DatetimeIndex(early), summary


# Panel with both legs and one boolean column per issue
def flag_panel(df):
    df = with_prev_close(df)
    key = ["date", "ticker"]
    idx = pd.MultiIndex.from_frame(df[key])

    def has(rows):
        return idx.isin(pd.MultiIndex.from_frame(rows[key])) if len(rows) else np.zeros(len(df), bool)

    ext = extreme_moves(df.drop(columns=["overnight", "intraday", "prev_close"]))
    ext["date"] = pd.to_datetime(ext["date"])

    df["flag_stale_open"] = has(stale_opens(df[COLS]))
    df["flag_extreme"] = has(ext)
    df["flag_reversal"] = has(reversals(df[COLS]))
    df["flag_ohlc"] = has(ohlc_violations(df))
    df["flag_bad_price"] = has(bad_prices(df))
    df["flag_ex_div"] = df["dividend"] > 0
    df["flag_split"] = df["split_factor"] != 1
    df["flag_half_day"] = df["date"].isin(NYSE.early_closes)
    df["flag_first_row"] = df["ret"].isna()
    return df


COLS = ["date", "ticker", "open", "high", "low", "close", "bid", "ask",
        "ret", "retx", "volume", "dividend", "split_factor"]


# VIX and T-bill

def vix_checks(end):
    cboe = pd.read_parquet(RAW / "vix_cboe.parquet")
    fred = pd.read_parquet(RAW / "vix_fred.parquet")["VIXCLS"].dropna()
    cal = sessions(cboe.index.min(), end)
    both = cboe[["close"]].join(fred, how="inner")
    diff = (both["close"] - both["VIXCLS"]).abs()

    ohlc_bad = cboe[(cboe["high"] < cboe["low"]) | (cboe["open"] > cboe["high"]) | (cboe["open"] < cboe["low"])
                    | (cboe["close"] > cboe["high"]) | (cboe["close"] < cboe["low"])]

    return {
        "common_days": len(both),
        "max_abs_diff": diff.max(),
        "days_diff_over_0.01": int((diff > 0.01).sum()),
        "cboe_missing_sessions": list(cal.difference(cboe.index).date),
        "fred_missing_sessions": list(cal.difference(fred.index).date),
        "cboe_non_session_rows": len(cboe.index.difference(cal)),
        "cboe_non_session_first": cboe.index.difference(cal).min().date(),
        "ohlc_bad": ohlc_bad,
    }


def tbill_checks(end):
    tb = pd.read_parquet(RAW / "tbill3m_fred.parquet")["DTB3"].dropna()
    cal = sessions(tb.index.min(), end)
    missing = cal.difference(tb.index)
    by_day = pd.Series([d.strftime("%m-%d") for d in missing]).value_counts()
    gaps = tb.index.to_series().diff().dt.days

    return {
        "obs": len(tb), "first": tb.index.min().date(), "last": tb.index.max().date(),
        "missing_sessions": missing, "missing_by_day": by_day,
        "longest_gap_days": int(gaps.max()), "min": tb.min(), "max": tb.max(),
        "negative_days": list(tb[tb < 0].index.date),
    }


# Report

def md_table(df, floatfmt="{:.4f}"):
    if df.index.name is not None or isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)

    lines = [head, sep]
    for r in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def short_list(items, n=12):
    items = [str(i) for i in items]
    return ", ".join(items[:n]) + (f" ... (+{len(items) - n} more)" if len(items) > n else "") if items else "none"


def report(source="yfinance"):
    df = load_prices(source)
    end = df["date"].max()

    flagged = flag_panel(df)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    flagged.to_parquet(PROCESSED / f"prices_{source}_flagged.parquet", index=False)

    cov = coverage(df)
    dup, bad, ohlc = duplicates(df), bad_prices(df), ohlc_violations(df)
    stale = stale_opens(df)
    ext = extreme_moves(df)
    rev = reversals(df)
    div, div_year = dividend_checks(df)
    spl = splits(df)
    early, early_summary = half_days(df)
    vix = vix_checks(end)
    tb = tbill_checks(end)

    stale_year = (stale.groupby(["ticker", stale["date"].dt.year]).size()
                  / df.groupby(["ticker", df["date"].dt.year]).size()).unstack(0).fillna(0)
    stale_year.index = stale_year.index.astype(int).rename("year")
    unexplained = ext[ext["event"].str.startswith("no known")]

    out = [f"# Step 2: data checks ({source})", ""]
    out += [f"Generated by `python src/checks.py {source}` from the snapshot in `data/raw/manifest.csv`. "
            f"Flagged panel: `data/processed/prices_{source}_flagged.parquet` (nothing dropped).", ""]

    out += ["## 1. Coverage vs the NYSE calendar", "",
            md_table(cov.assign(missing_sessions=cov["missing_sessions"].map(short_list),
                                non_session_rows=cov["non_session_rows"].map(short_list))), ""]
    out += [f"Duplicate (ticker, date) rows: **{len(dup)}**. "
            f"Zero, negative or NaN prices or zero volume: **{len(bad)}**. "
            f"High < low, or open/close outside [low, high]: **{len(ohlc)}**.", ""]

    out += ["## 2. Stale opens (open == previous close)", "",
            f"{len(stale)} rows in total. Share of days per year:", "",
            md_table(stale_year.round(3), "{:.3f}"), ""]

    out += [f"## 3. Moves beyond ±{EXTREME:.0%}", "",
            f"{(ext['leg'] == 'overnight').sum()} overnight and {(ext['leg'] == 'intraday').sum()} intraday. "
            f"{len(unexplained)} have no matching event:", "",
            md_table(unexplained) if len(unexplained) else "none", "",
            f"Large opposite-sign legs (both beyond ±{REVERSAL:.0%}), the pattern a bad open would leave:", "",
            md_table(rev.assign(date=rev["date"].dt.date)), "",
            "<details><summary>All moves beyond the threshold</summary>", "", md_table(ext), "", "</details>", ""]

    out += ["## 4. Dividends and splits", "", md_table(div.set_index("ticker").T.rename_axis("metric"), "{:.6f}"), "",
            "Ex-dates per year:", "", md_table(div_year), "",
            "Splits:", "", md_table(spl.assign(date=spl["date"].dt.date)), ""]

    out += ["## 5. Half-days", "",
            f"{len(early)} NYSE early closes in the sample: {short_list(early.date, 50)}.", "",
            "Mean absolute leg return, half-days vs full days:", "", md_table(early_summary), ""]

    out += ["## 6. VIX: CBOE vs FRED", "",
            f"- Days in both: {vix['common_days']}; largest |close difference|: {vix['max_abs_diff']:.4f}; "
            f"days differing by more than 0.01: {vix['days_diff_over_0.01']}",
            f"- NYSE sessions missing from CBOE: {short_list(vix['cboe_missing_sessions'])}",
            f"- NYSE sessions missing from FRED: {short_list(vix['fred_missing_sessions'])}",
            f"- CBOE rows on non-NYSE days: {vix['cboe_non_session_rows']} (first {vix['cboe_non_session_first']})",
            f"- CBOE rows with inconsistent open/high/low: {len(vix['ohlc_bad'])}, "
            f"last one {vix['ohlc_bad'].index.max().date() if len(vix['ohlc_bad']) else 'n/a'}", ""]

    out += ["## 7. 3-month T-bill (FRED DTB3)", "",
            f"- {tb['obs']} observations, {tb['first']} to {tb['last']}, range {tb['min']:.2f}% to {tb['max']:.2f}%",
            f"- NYSE sessions with no rate: {len(tb['missing_sessions'])}; longest gap between observations: "
            f"{tb['longest_gap_days']} calendar days",
            f"- Most common missing days (MM-DD): {short_list([f'{k} ({v})' for k, v in tb['missing_by_day'].items()], 10)}",
            f"- Negative-rate days: {short_list(tb['negative_days'])}", ""]

    out += [DECISIONS]

    path = ROOT / "reports" / "step2_data_checks.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out))
    print(f"wrote {path.relative_to(ROOT)} and {PROCESSED.name}/prices_{source}_flagged.parquet")


DECISIONS = """## 8. Decisions

| Issue | Handling |
|---|---|
| Yahoo prices are split-adjusted back in time | `data_layer` multiplies them back by later splits (QQQ 2:1 on 2000-03-20, IWM 2:1 on 2005-06-09) so open/close are as traded, like CRSP. Returns are unaffected. |
| ret from Adj Close vs CRSP-style (close + dividend) / previous close | Kept Adj Close as specified; the two differ by at most ~2.5e-4 on an ex-date (Yahoo divides by the ex-dividend price, CRSP adds the dividend). |
| Dividend leg | By construction the dividend lands on the overnight leg: intraday uses same-day open and close, overnight is total return over the intraday factor. The data mostly agree: on SPY and IWM ex-dates the open is lower than usual by 77% and 109% of the dividend (s.e. 16% and 22%). QQQ yields are too small to tell (7%, s.e. 49%). But mean intraday return on ex-dates is about -0.23% to -0.25% for SPY and IWM against ~0 on other days, so some of the drop may show up after the open (a Yahoo open that is not fully ex-dividend, or a real ex-day pattern). Kept and flagged (`flag_ex_div`); to be checked against CRSP `openprc`, and step 3 will report results with and without ex-dates. |
| Stale opens | Kept and flagged (`flag_stale_open`). They force the overnight price return to zero, so they bias overnight towards zero. Robustness: rerun with these days excluded, and compare with CRSP opens once available. |
| Extreme moves and reversals | Kept and flagged. Almost all line up with known events. Unexplained ones are listed above and will be checked against CRSP; they are not dropped on the basis of Yahoo alone. |
| Half-days | Kept and flagged (`flag_half_day`). The intraday leg is 3.5 hours on these days; robustness check excluding them. |
| First row per ticker | No return (no previous close); excluded from return statistics. |
| VIX | Use CBOE close; FRED is identical where both exist. Ignore VIX open/high/low (inconsistent in early years). Use only NYSE sessions (CBOE has rows on market holidays since 2022). Fill the few CBOE gaps from FRED, otherwise carry the previous close forward. |
| T-bill | Missing days are bond-market holidays (Columbus Day, Veterans Day) when NYSE is open. Carry the last rate forward. Daily risk-free = DTB3 / 100 / 360 × calendar days to the next session (discount-basis approximation, documented in step 3). |
| Sample end | Snapshot is 2026-10-06; FRED series end 2026-10-02 and prices 2026-10-05. Not re-downloaded: Yahoo revises Adj Close history on every pull, so a fixed snapshot is more reproducible. Common end date: 2026-10-02. |
"""


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "yfinance")
