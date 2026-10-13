# Step 3: build and verify overnight, intraday and risk-free returns

import sys

import exchange_calendars as xc
import numpy as np
import pandas as pd

from checks import NYSE, flag_panel, md_table
from config import END, ROOT, SAMPLES
from data_layer import load_prices, load_tbill, save_processed

BILL_DAYS = 91  # DTB3 is the 13 week bill


# Risk-free return from each close to the next, from DTB3
# Bill price is 1 - d * 91/360, compounded per calendar day so weekends earn too
# Bond market holidays carry the last rate forward
def daily_rf(dates):
    sessions = pd.DatetimeIndex(sorted(set(dates)))
    d = load_tbill().reindex(sessions.union(load_tbill().index)).ffill().reindex(sessions) / 100

    per_day = (1 / (1 - d * BILL_DAYS / 360)) ** (1 / BILL_DAYS)
    prev = pd.Series(sessions, index=sessions).shift()
    days = (sessions.to_series() - prev).dt.days
    rf = per_day.shift() ** days - 1
    return pd.DataFrame({"date": sessions, "dtb3_prev": (d.shift() * 100).to_numpy(),
                         "rf_days": days.to_numpy(), "rf": rf.to_numpy()})


def sample_of(dates):
    out = pd.Series("", index=dates.index)
    for name, (a, b) in SAMPLES.items():
        out[(dates >= a) & (dates <= b)] = name
    return out


def build(source="yfinance"):
    px = load_prices(source)
    px = px[px["date"] <= END]
    df = flag_panel(px).drop(columns=["prev_close"])

    # rf for each ETF from its previous session to today
    cal = NYSE.sessions_in_range(df["date"].min() - pd.Timedelta(days=10), END)
    df = df.merge(daily_rf(cal), on="date", how="left")
    df.loc[df["flag_first_row"], ["rf", "rf_days"]] = np.nan

    # all T-bill interest goes to the overnight leg, an intraday-only position
    # sits in cash overnight and earns it there
    df["rf_overnight"] = df["rf"]
    df["rf_intraday"] = np.where(df["flag_first_row"], np.nan, 0.0)
    df["overnight_xs"] = df["overnight"] - df["rf_overnight"]
    df["intraday_xs"] = df["intraday"] - df["rf_intraday"]

    df["sample"] = sample_of(df["date"])
    return df


# Verification

# (1 + overnight)(1 + intraday) should equal 1 + ret every day
def identity_check(df):
    d = df[~df["flag_first_row"]]
    err = ((1 + d["overnight"]) * (1 + d["intraday"]) - (1 + d["ret"])).abs()
    return d.assign(err=err).groupby("ticker")["err"].agg(["size", "max"]).rename(
        columns={"size": "days", "max": "max_abs_error"})


# Compounded legs should give the compounded total return
# Only the errors are shown, the growth levels are results for later
def cumulative_check(df):
    rows = []
    for (t, smp), g in df[~df["flag_first_row"]].groupby(["ticker", "sample"]):
        on, intra, tot = (1 + g["overnight"]).prod(), (1 + g["intraday"]).prod(), (1 + g["ret"]).prod()
        rows.append({"ticker": t, "sample": smp, "days": len(g), "rel_error": on * intra / tot - 1})
    return pd.DataFrame(rows).pivot(index="ticker", columns="sample", values="rel_error")[
        ["train", "validation", "test"]]


# With no dividend or split, overnight must be open / previous close - 1
def raw_gap_check(df):
    d = df.copy()
    d["prev_close"] = d.groupby("ticker")["close"].shift()
    plain = ~d["flag_first_row"] & ~d["flag_ex_div"] & ~d["flag_split"]
    err = (d["overnight"] - (d["open"] / d["prev_close"] - 1))[plain].abs()
    return d[plain].assign(err=err).groupby("ticker")["err"].agg(["size", "median", "max"]).rename(
        columns={"size": "days"})


# Recompute a few dividend and split days by hand from unadjusted prices
def hand_checks(df):
    d = df.copy()
    d["prev_close"] = d.groupby("ticker")["close"].shift()
    picks = []
    for t in ["SPY", "QQQ", "IWM"]:
        ex = d[(d["ticker"] == t) & d["flag_ex_div"] & (d["sample"] != "test")]
        picks += [ex.index[0], ex.index[len(ex) // 2], ex.index[-1]]
    picks += list(d.index[d["flag_split"]])

    h = d.loc[picks, ["date", "ticker", "prev_close", "open", "close", "dividend", "split_factor",
                      "ret", "overnight", "intraday"]].copy()

    h["ret_by_hand"] = (h["close"] + h["dividend"]) * h["split_factor"] / h["prev_close"] - 1
    h["overnight_price_only"] = h["open"] * h["split_factor"] / h["prev_close"] - 1
    h["overnight_by_hand"] = (1 + h["ret_by_hand"]) / (h["close"] / h["open"]) - 1
    h["div_yield"] = h["dividend"] * h["split_factor"] / h["prev_close"]
    h["diff_ret"] = h["ret"] - h["ret_by_hand"]
    h["diff_overnight"] = h["overnight"] - h["overnight_by_hand"]
    h["date"] = h["date"].dt.date
    return h


def first_day_check(df):
    first = df.sort_values("date").groupby("ticker").head(1)
    return first[["ticker", "date", "ret", "overnight", "intraday", "rf"]].assign(date=lambda x: x["date"].dt.date)


def rf_check(df):
    rf = df.sort_values("date").drop_duplicates("date").set_index("date")
    s = rf[["dtb3_prev", "rf_days", "rf"]].dropna()
    weekend = s[s["rf_days"] == 3].iloc[-1:]
    ordinary = s[s["rf_days"] == 1].iloc[-1:]
    examples = pd.concat([ordinary, weekend]).reset_index()
    examples["by_hand"] = (1 / (1 - examples["dtb3_prev"] / 100 * 91 / 360)) ** (examples["rf_days"] / 91) - 1
    examples["date"] = examples["date"].dt.date

    annual = (1 + s["rf"]).groupby(s.index.year).prod() - 1
    mean_dtb3 = s["dtb3_prev"].groupby(s.index.year).mean()
    years = pd.DataFrame({"compounded_rf": annual, "mean_dtb3_pct": mean_dtb3}).loc[[1995, 2000, 2008, 2015, 2023]]
    years.index.name = "year"
    return {"missing": int(df.loc[~df["flag_first_row"], "rf"].isna().sum()),
            "examples": examples, "years": years}


def report(source="yfinance"):
    df = build(source)
    path = save_processed(df, f"returns_{source}")

    ident, cum, gap, hand = identity_check(df), cumulative_check(df), raw_gap_check(df), hand_checks(df)
    first, rf = first_day_check(df), rf_check(df)
    flags = [c for c in df.columns if c.startswith("flag_")]
    counts = df.groupby("sample")["ticker"].value_counts().unstack().loc[["train", "validation", "test"]]

    hand_cols1 = ["date", "ticker", "prev_close", "open", "close", "dividend", "split_factor"]
    hand_cols2 = ["date", "ticker", "ret", "ret_by_hand", "diff_ret", "overnight", "overnight_by_hand",
                  "diff_overnight", "overnight_price_only", "div_yield", "intraday"]

    out = [f"# Step 3: returns ({source})", "",
           f"Generated by `python src/returns.py {source}`. Output: `data/processed/{path.name}` "
           f"({len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}), "
           "saved with `data_layer.save_processed`. No summary statistics here (step 5, train sample only).", "",
           "## Definitions", "",
           "- `ret`: close-to-close total return (from Adj Close, dividends included)",
           "- `intraday = close / open - 1` (same-day unadjusted prices)",
           "- `overnight = (1 + ret) / (close / open) - 1`: previous close to open, so the dividend and any split adjustment sit here",
           "- `rf`: 3-month T-bill return from the previous session's close to today's close. DTB3 is annual % on a "
           "discount basis, so price = 1 - d·91/360, the 91-day return is 1/price - 1, compounded per calendar day "
           "(a Friday-to-Monday period earns 3 days). Uses the rate known at the previous close; "
           "bond-market holidays carry the last rate forward.",
           f"- Sample ends {END} (last day every source covers). `sample` column: train 1993-2012, validation 2013-2018, test 2019+.",
           f"- Step 2 flags carried through: {', '.join(f'`{f}`' for f in flags)}.", "",
           "Rows per sample:", "", md_table(counts.rename_axis("sample")), "",
           "## Checks", "",
           "**1. (1 + overnight)(1 + intraday) = 1 + ret every day**", "", md_table(ident, "{:.2e}"), "",
           "**2. Cumulative products match**: relative error of ∏(1+overnight)·∏(1+intraday) vs ∏(1+ret), per ETF and sample "
           "(growth levels deliberately not shown)", "", md_table(cum, "{:.6g}"), "",
           "**3. Overnight vs raw open / previous close on days with no dividend or split** "
           "(independent of `ret`; differences are Adj Close rounding)", "", md_table(gap, "{:.2e}"), "",
           "**4. Hand checks: first, middle and last ex-dividend day per ETF, and both splits**", "",
           "`ret_by_hand = (close + dividend) × split / prev_close - 1`; "
           "`overnight_price_only = open × split / prev_close - 1` (the move in the price alone).", "",
           "Picked from the train and validation samples only.", "",
           md_table(hand[hand_cols1], "{:.4f}"), "", md_table(hand[hand_cols2], "{:.5f}"), "",
           "**5. First day per ETF has no overnight return**", "", md_table(first, "{:.5f}"), "",
           "**6. Risk-free rate**", "",
           f"- Days after the first with no rf: {rf['missing']}",
           "- Hand check of the conversion:", "", md_table(rf["examples"], "{:.6g}"), "",
           "- Sense check, rf compounded over a calendar year vs that year's average DTB3:", "",
           md_table(rf["years"], "{:.4f}"), ""]
    out += [NOTES]

    p = ROOT / "reports" / "step3_returns.md"
    p.write_text("\n".join(out))
    print(f"wrote {p.relative_to(ROOT)} and {path.relative_to(ROOT)}")
    return df


NOTES = """## Notes and decisions

- Ex-dividend days: `overnight` includes the dividend (it is a total return), so on ex-dates it is roughly
  `overnight_price_only + div_yield`. The intraday leg never contains the dividend.
- Split days: `overnight_price_only` uses `open × split_factor`, so the 2:1 splits (QQQ 2000-03-20,
  IWM 2005-06-09) show ordinary small moves, not -50%.
- Yahoo's Adj Close follows its own convention (dividends deflate the past instead of being added to
  today's price), so `ret` differs from the CRSP-style hand calculation by up to about 2.5e-4 on ex-dates
  and by rounding noise elsewhere. CRSP `ret` will remove that once it is loaded.
- `rf` is per close-to-close period and is assigned entirely to the overnight leg (decided 16 Oct):
  `rf_overnight = rf`, `rf_intraday = 0`, and `overnight_xs = overnight - rf_overnight`,
  `intraday_xs = intraday` (arithmetic excess returns). The overnight period holds the weekends and
  holidays, and the ~6.5 trading hours earn no T-bill interest in this convention.
- Nothing is dropped: flagged days stay in the file and later steps choose whether to exclude them.
"""


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "yfinance")
