# Step 2: data quality checks on CRSP prices, VIX and the risk-free rate

import sys

import exchange_calendars as xc
import numpy as np
import pandas as pd

from config import END, PERMNOS, PROCESSED, ROOT
from data_layer import (
    RF_FILE,
    TBILL_EXT_FILE,
    TBILL_FILE,
    VIX_FILE,
    load_crsp_raw,
    load_prices,
    load_rf,
    read_tbill,
    split_legs,
)

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
    return "no known event"


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


# Raw file checks, before load_prices cleans anything

def raw_checks():
    raw = load_crsp_raw()
    names = {v: k for k, v in PERMNOS.items()}

    tickers = (raw.groupby(["PERMNO", "Ticker"])["DlyCalDt"].agg(["min", "max", "size"])
               .reset_index().rename(columns={"min": "first", "max": "last", "size": "rows"}))
    tickers.insert(1, "mapped_to", tickers["PERMNO"].map(names))

    exact = raw[raw.duplicated(keep=False)]
    key_only = raw[raw.duplicated(["PERMNO", "DlyCalDt"], keep=False) & ~raw.duplicated(keep=False)]
    no_trade = raw[raw["DlyClose"].isna() | (raw["DlyVol"] == 0)]
    no_open = raw[raw["DlyOpen"].isna()]

    cols = ["PERMNO", "Ticker", "DlyCalDt", "DlyPrc", "DlyClose", "DlyOpen", "DlyVol", "DlyRet"]
    return {"rows": len(raw), "tickers": tickers, "exact_dups": exact[cols].drop_duplicates(),
            "n_exact": len(exact) - len(exact.drop_duplicates()), "key_dups": len(key_only),
            "no_trade": no_trade[cols], "no_open": no_open[cols]}


# One suspicious IWM day: Yahoo's open was far above the close
def iwm_check(flagged):
    d = flagged[(flagged["ticker"] == "IWM") & flagged["date"].between("2002-11-07", "2002-11-15")]
    d = d.assign(date=d["date"].dt.date)
    return d[["date", "open", "high", "low", "close", "prev_close", "overnight", "intraday", "ret",
              "flag_stale_open", "flag_extreme", "flag_reversal", "flag_ohlc"]]


# VIX and risk-free rate

def vix_checks():
    v = pd.read_csv(VIX_FILE, parse_dates=["date"]).set_index("date")
    v = v[v.index <= END]
    cal = sessions(v.index.min(), END)
    close = v["vix"]

    return {
        "rows": len(v), "first": v.index.min().date(), "last": v.index.max().date(),
        "nan_close": list(close[close.isna()].index.date),
        "missing_sessions": list(cal.difference(close.dropna().index).date),
        "non_session_rows": list(v.index.difference(cal).date),
        "min": close.min(), "max": close.max(),
        "max_date": close.idxmax().date(),
        "non_positive": int((close <= 0).sum()),
        "big_jumps": close.pct_change(fill_method=None).abs().gt(0.5).sum(),
        "open_above_close_x2": int((v["vixo"] > 2 * close).sum()),
    }


def rf_checks():
    wrds_raw = pd.read_csv(TBILL_FILE, parse_dates=["date"]).set_index("date")["dtb3"]
    wrds = pd.to_numeric(wrds_raw, errors="coerce").dropna()
    ext_raw = pd.read_csv(TBILL_EXT_FILE, parse_dates=["date"]).set_index("date")["dtb3"]
    ext = pd.to_numeric(ext_raw, errors="coerce").dropna()
    bill = read_tbill()
    rf = load_rf().loc[:END]
    cal = sessions(rf.index.min(), END)

    by_year = (1 + rf).groupby(rf.index.year).prod() - 1
    years = pd.DataFrame({"compounded_rf_%": by_year * 100,
                          "mean_daily_bp": rf.groupby(rf.index.year).mean() * 1e4,
                          "days": rf.groupby(rf.index.year).size()}).rename_axis("year")

    # The old WRDS Fama-French export is retained only for an audit comparison.
    ff = pd.read_csv(RF_FILE, parse_dates=["date"]).set_index("date")["rf"].loc[:END].dropna()
    ff_by_year = (1 + ff).groupby(ff.index.year).prod() - 1
    compare_years = [1995, 2003, 2017, 2025]
    comparison = pd.DataFrame({
        "T-bill rf %": (by_year.reindex(compare_years) * 100),
        "rounded FF rf %": (ff_by_year.reindex(compare_years) * 100),
    }).rename_axis("year")

    return {
        "rows": len(rf), "first": rf.index.min().date(), "last": rf.index.max().date(),
        "nan": int(rf.isna().sum()), "negative": int((rf < 0).sum()),
        "min": rf.min(), "max": rf.max(), "mean": rf.mean(),
        "missing_sessions": list(cal.difference(rf.index).date),
        "non_session_rows": list(rf.index.difference(cal).date),
        "years": years, "comparison": comparison,
        "wrds_raw_rows": len(wrds_raw), "wrds_raw_first": wrds_raw.index.min().date(),
        "wrds_raw_last": wrds_raw.index.max().date(), "wrds_values": len(wrds),
        "wrds_first": wrds.index.min().date(), "wrds_last": wrds.index.max().date(),
        "wrds_empty_after_last": list(
            wrds_raw.index[(wrds_raw.index > wrds.index.max()) & wrds_raw.isna()].date
        ),
        "ext_rows": len(ext), "ext_first": ext.index.min().date(), "ext_last": ext.index.max().date(),
        "bill_values": len(bill), "bill_first": bill.index.min().date(), "bill_last": bill.index.max().date(),
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


def report(source="crsp"):
    df = load_prices(source)
    df = df[df["date"] <= END]

    flagged = flag_panel(df)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    flagged.to_parquet(PROCESSED / f"prices_{source}_flagged.parquet", index=False)

    raw = raw_checks()
    cov = coverage(df)
    dup, bad, ohlc = duplicates(df), bad_prices(df), ohlc_violations(df)
    stale = stale_opens(df)
    ext = extreme_moves(df)
    rev = reversals(df)
    div, div_year = dividend_checks(df)
    spl = splits(df)
    early, early_summary = half_days(df)
    iwm = iwm_check(flagged)
    vix = vix_checks()
    rf = rf_checks()

    stale_year = (stale.groupby(["ticker", stale["date"].dt.year]).size()
                  / df.groupby(["ticker", df["date"].dt.year]).size()).unstack(0).fillna(0)
    stale_year.index = stale_year.index.astype(int).rename("year")
    unexplained = ext[ext["event"].str.startswith("no known")]

    out = [f"# Step 2: data checks ({source})", ""]
    out += [f"Generated by `python src/checks.py {source}` from the snapshot in `data/raw/manifest.csv` "
            f"(`crsp_daily.csv`, `wrds_vix.csv`, `wrds_tbill3m.csv`, `fred_dtb3_extension.csv`), up to {END}. "
            f"Flagged panel: `data/processed/prices_{source}_flagged.parquet` (nothing dropped).", ""]

    out += ["## 1. Raw CRSP file", "",
            f"{raw['rows']} rows for the three PERMNOs. Rows are matched to ETFs on PERMNO only; the ticker "
            "field changes over time:", "", md_table(raw["tickers"].assign(first=raw["tickers"]["first"].dt.date,
                                                                          last=raw["tickers"]["last"].dt.date)), "",
            f"Exact duplicate rows: **{raw['n_exact']}** extra copies, dropped in `data_layer` (one copy kept). "
            f"Same PERMNO and date with different values: **{raw['key_dups']}**.", "",
            md_table(raw["exact_dups"].assign(DlyCalDt=raw["exact_dups"]["DlyCalDt"].dt.date)), "",
            "Days with no trade (no `DlyClose` or zero volume; `DlyPrc` is then the bid/ask midpoint):", "",
            md_table(raw["no_trade"].assign(DlyCalDt=raw["no_trade"]["DlyCalDt"].dt.date)), "",
            "Days with no opening price (no overnight or intraday return on these days):", "",
            md_table(raw["no_open"].assign(DlyCalDt=raw["no_open"]["DlyCalDt"].dt.date)), ""]

    out += ["## 2. Coverage vs the NYSE calendar", "",
            md_table(cov.assign(missing_sessions=cov["missing_sessions"].map(short_list),
                                non_session_rows=cov["non_session_rows"].map(short_list))), ""]
    out += [f"Duplicate (ticker, date) rows after loading: **{len(dup)}**. "
            f"Zero, negative or NaN prices or zero volume: **{len(bad)}**. "
            f"High < low, or open/close outside [low, high]: **{len(ohlc)}**.", ""]

    out += ["## 3. Stale opens (open == previous close)", "",
            f"{len(stale)} rows in total. Share of days per year:", "",
            md_table(stale_year.round(3), "{:.3f}"), ""]

    out += [f"## 4. Moves beyond ±{EXTREME:.0%}", "",
            f"{(ext['leg'] == 'overnight').sum()} overnight and {(ext['leg'] == 'intraday').sum()} intraday. "
            f"{len(unexplained)} have no matching event:", "",
            md_table(unexplained) if len(unexplained) else "none", "",
            f"Large opposite-sign legs (both beyond ±{REVERSAL:.0%}), the pattern a bad open would leave:", "",
            md_table(rev.assign(date=rev["date"].dt.date)) if len(rev) else "none", "",
            "<details><summary>All moves beyond the threshold</summary>", "", md_table(ext), "", "</details>", ""]

    out += ["## 5. IWM around 2002-11-12", "",
            "Yahoo's open on 2002-11-12 was about 6% above the close, and it "
            "showed up as an unexplained reversal in the Yahoo checks. CRSP:", "",
            md_table(iwm, "{:.4f}"), ""]

    out += ["## 6. Dividends and splits", "", md_table(div.set_index("ticker").T.rename_axis("metric"), "{:.6f}"), "",
            "Ex-dates per year:", "", md_table(div_year), "",
            "Splits:", "", md_table(spl.assign(date=spl["date"].dt.date)), ""]

    out += ["## 7. Half-days", "",
            f"{len(early)} NYSE early closes in the sample: {short_list(early.date, 50)}.", "",
            "Mean absolute leg return, half-days vs full days:", "", md_table(early_summary), ""]

    out += ["## 8. VIX (`wrds_vix.csv`, column `vix`)", "",
            f"- {vix['rows']} rows, {vix['first']} to {vix['last']}; range {vix['min']:.2f} to {vix['max']:.2f} "
            f"(max on {vix['max_date']}); values at or below zero: {vix['non_positive']}",
            f"- Rows with no close: {short_list(vix['nan_close'])}",
            f"- NYSE sessions with no close: {short_list(vix['missing_sessions'])} (filled with the previous close)",
            f"- Rows on non-NYSE days: {short_list(vix['non_session_rows'])} (dropped)",
            f"- Day-to-day moves above 50%: {vix['big_jumps']}; open more than twice the close: "
            f"{vix['open_above_close_x2']}", ""]

    out += ["## 9. Risk-free rate (3-month T-bill)", "",
            f"- WRDS H.15 (`wrds_tbill3m.csv`): {rf['wrds_raw_rows']:,} rows from "
            f"{rf['wrds_raw_first']} to {rf['wrds_raw_last']}; {rf['wrds_values']:,} non-missing observations. "
            f"The last valid value is {rf['wrds_last']}; later empty rows: "
            f"{short_list(rf['wrds_empty_after_last'])}.",
            "- Source audit performed when the extension was built: archived FRED DTB3 and WRDS matched on all "
            "8,033 overlapping observations, with no value mismatches (largest absolute difference 0) and no "
            "dates present in only one source.",
            f"- FRED DTB3 extension (`fred_dtb3_extension.csv`): {rf['ext_rows']:,} values from "
            f"{rf['ext_first']} to {rf['ext_last']}. It is appended only after the last valid WRDS value, giving "
            f"{rf['bill_values']:,} source observations through {rf['bill_last']}.",
            "- Conversion: for annual discount rate d, bill price = 1 - d×91/360 and the 91-day holding return "
            "is 1/price - 1. It is compounded per calendar day; the rate known at the previous close is used, "
            "with the last observation carried across bond-market holidays.",
            f"- Daily close-to-close rf: {rf['rows']:,} NYSE sessions, {rf['first']} to {rf['last']}; "
            f"missing values: {rf['nan']}; negative values: {rf['negative']}; range {rf['min']:.7f} to "
            f"{rf['max']:.7f}.",
            f"- NYSE sessions with no rf: {short_list(rf['missing_sessions'])}",
            f"- rf rows on non-NYSE days: {short_list(rf['non_session_rows'])}", "",
            "Compounded rf by year:", "", md_table(rf["years"], "{:.3f}"), "",
            "Selected years compared with the old 4-decimal Fama-French export (audit only):", "",
            md_table(rf["comparison"], "{:.3f}"), ""]

    out += [DECISIONS]

    path = ROOT / "reports" / "step2_data_checks.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out))
    print(f"wrote {path.relative_to(ROOT)} and {PROCESSED.name}/prices_{source}_flagged.parquet")


DECISIONS = """## 10. Decisions

| Issue | Handling |
|---|---|
| Source | CRSP v2 daily file for ETF returns, rows picked and named by PERMNO (84398 SPY, 86755 QQQ, 88222 IWM). The ticker field is not used: QQQ was listed as QQQQ from 2004-12-01 to 2011-03-22. WRDS CBOE supplies VIX; WRDS H.15 supplies the T-bill through 2025-02-11 and FRED DTB3 extends it through 2025-12-31. |
| Duplicate rows | 6 rows appear twice, identical in every column, all on ex-dividend days. One copy of each is kept. |
| Close | `DlyPrc`, which equals `DlyClose` on every day with a trade. On 1997-03-31 SPY did not trade and `DlyPrc` is the bid/ask midpoint, the price CRSP's return is based on. |
| Missing opens | 5 rows have no `DlyOpen`: the first row of SPY and QQQ, SPY 1993-02-01 and 1994-11-07, and the no-trade day 1997-03-31. Kept and flagged (`flag_bad_price`). Overnight and intraday are missing on those days, close-to-close `ret` is not. |
| Stale opens | Kept and flagged (`flag_stale_open`). CRSP has the same pattern as Yahoo: 14% of SPY sessions in 1993, falling below 2% after 1998. So switching source does not remove this problem from the early years. |
| Extreme moves and reversals | All moves beyond 5% match a known event, and there are no opposite-sign reversals beyond 3%. The unexplained reversals in the Yahoo data were Yahoo errors. Kept and flagged as before. |
| IWM 2002-11-12 | CRSP's open is well below Yahoo's, so the Yahoo open was wrong and the day is ordinary in CRSP. The CRSP high looks like the same bad print: it is 7% above both the open and the close. Only a high-low spread estimate would use it, and with CRSP bid/ask available none is needed. |
| Dividends | The file has no dividend amount, so it is backed out as (`ret` - `retx`) × previous close. `ret` - `retx` therefore matches the yield exactly by construction, and is zero on all other days. With CRSP opens, the open falls by 76% (SPY) and 110% (IWM) of the dividend on ex-dates, and intraday is about -0.22% to -0.25% on ex-dates. That matches the Yahoo results, so it is a real ex-day pattern rather than a Yahoo artefact. Kept and flagged (`flag_ex_div`). |
| Splits | From the drop in `DlyCumFacPr`: QQQ 2:1 on 2000-03-20 and IWM 2:1 on 2005-06-09, both showing ordinary moves. |
| Half-days | Kept and flagged (`flag_half_day`). |
| VIX | `vix` column (close) on NYSE sessions only. Rows on market holidays are dropped, and the 2 NYSE sessions with no value carry the previous close forward. The 7 moves above 50% are all known shocks (2007-02-27, 2018-02-05, 2021-01-27, 2021-11-26, 2024-08-05, 2024-12-18, 2025-04-04). The open (`vixo`) has obvious errors (two days above twice the close) and is not used. |
| Risk-free rate | WRDS/FRED DTB3 is converted from its annual bank-discount yield to a 91-day bill holding return, then to a return over the calendar days between NYSE closes using the rate available at the previous close. All rf is assigned to the overnight leg. The old 4-decimal Fama-French file is retained only for comparison; its rounding materially distorted low-rate years. The two T-bill sources matched exactly over 8,033 archived overlapping observations, and the FRED extension is used only after WRDS ends. |
| Sample end | 2025-12-31, the last CRSP day and the last FRED extension value. VIX goes further and is cut there. |
"""


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "crsp")
