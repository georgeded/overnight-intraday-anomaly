# Step 3: build and verify overnight, intraday and risk-free returns

import sys

import numpy as np
import pandas as pd

from checks import NYSE, flag_panel, md_table
from config import END, ROOT, SAMPLES
from data_layer import load_prices, load_rf, save_processed


# Fama-French daily rf is already the return from the previous session to today
# rf_days is the calendar days covered, kept to tell weekend nights apart
def daily_rf(dates):
    sessions = pd.DatetimeIndex(sorted(set(dates)))
    prev = pd.Series(sessions, index=sessions).shift()
    days = (sessions.to_series() - prev).dt.days
    rf = load_rf().reindex(sessions)
    return pd.DataFrame({"date": sessions, "rf_days": days.to_numpy(), "rf": rf.to_numpy()})


def sample_of(dates):
    out = pd.Series("", index=dates.index)
    for name, (a, b) in SAMPLES.items():
        out[(dates >= a) & (dates <= b)] = name
    return out


def build(source="crsp"):
    px = load_prices(source)
    px = px[px["date"] <= END]
    df = flag_panel(px).drop(columns=["prev_close"])

    # rf for each ETF from its previous session to today
    cal = NYSE.sessions_in_range(df["date"].min() - pd.Timedelta(days=10), END)
    df = df.merge(daily_rf(cal), on="date", how="left")
    df.loc[df["flag_first_row"], ["rf", "rf_days"]] = np.nan

    # all of the day's rf goes to the overnight leg, an intraday-only position
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
    return d.assign(err=err).groupby("ticker")["err"].agg(["size", "count", "max"]).rename(
        columns={"size": "days", "count": "days_with_both_legs", "max": "max_abs_error"})


# Compounded legs should give the compounded total return, on days with both legs
# Only the errors are shown, the growth levels are results for later
def cumulative_check(df):
    rows = []
    both = ~df["flag_first_row"] & df["overnight"].notna() & df["intraday"].notna()
    for (t, smp), g in df[both].groupby(["ticker", "sample"]):
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
    return d[plain].assign(err=err).groupby("ticker")["err"].agg(["count", "median", "max"]).rename(
        columns={"count": "days"})


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


# rf in the panel must be the raw file value on every day, and all of it overnight
def rf_check(df):
    d = df[~df["flag_first_row"]]
    raw = load_rf().reindex(d["date"]).to_numpy()

    by_day = d.groupby(d["rf_days"].clip(upper=4))["rf"].agg(["size", "mean"])
    by_day.index = by_day.index.map({1: "1 (weekday)", 2: "2", 3: "3 (weekend)", 4: "4+ (long weekend)"})
    by_day["mean"] = by_day["mean"] * 1e4

    return {"missing": int(d["rf"].isna().sum()),
            "max_diff_raw": float(np.nanmax(np.abs(d["rf"].to_numpy() - raw))),
            "overnight_equals_rf": bool((d["rf_overnight"] == d["rf"]).all()),
            "intraday_zero": bool((d["rf_intraday"] == 0).all()),
            "xs_check": float((d["overnight_xs"] - (d["overnight"] - d["rf"])).abs().max()),
            "by_day": by_day.rename_axis("calendar days covered").rename(columns={"size": "days", "mean": "mean_rf_bp"})}


def report(source="crsp"):
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
           "- `ret`: CRSP `DlyRet`, close-to-close total return with dividends",
           "- `intraday = close / open - 1` (same-day CRSP `DlyOpen` and `DlyPrc`, unadjusted)",
           "- `overnight = (1 + ret) / (close / open) - 1`: previous close to open, so the dividend and any split adjustment sit here",
           "- `rf`: Fama-French daily risk-free return (`ff_rf_daily.csv`, decimal), used as is for the period from the "
           "previous session's close to today's close. No T-bill conversion.",
           f"- Sample ends {END}. `sample` column: train 1993-2012, validation 2013-2018, test 2019-2025.",
           f"- Step 2 flags carried through: {', '.join(f'`{f}`' for f in flags)}.", "",
           "Rows per sample:", "", md_table(counts.rename_axis("sample")), "",
           "## Checks", "",
           "**1. (1 + overnight)(1 + intraday) = 1 + ret every day**", "", md_table(ident, "{:.2e}"), "",
           "**2. Cumulative products match**: relative error of ∏(1+overnight)·∏(1+intraday) vs ∏(1+ret), per ETF and sample, "
           "on days with both legs (growth levels deliberately not shown)", "", md_table(cum, "{:.6g}"), "",
           "**3. Overnight vs raw open / previous close on days with no dividend or split** "
           "(independent of `ret`; CRSP rounds `ret` to 6 decimals, so errors up to about 5e-7 are expected)", "",
           md_table(gap, "{:.2e}"), "",
           "**4. Hand checks: first, middle and last ex-dividend day per ETF, and both splits**", "",
           "`ret_by_hand = (close + dividend) × split / prev_close - 1`; "
           "`overnight_price_only = open × split / prev_close - 1` (the move in the price alone).", "",
           "Picked from the train and validation samples only.", "",
           md_table(hand[hand_cols1], "{:.4f}"), "", md_table(hand[hand_cols2], "{:.5f}"), "",
           "**5. First day per ETF has no overnight return**", "", md_table(first, "{:.5f}"), "",
           "**6. Risk-free rate**", "",
           f"- Days after the first with no rf: {rf['missing']}",
           f"- Largest difference between `rf` in the panel and the raw file: {rf['max_diff_raw']:.1e}",
           f"- `rf_overnight` equals `rf` on every day: {rf['overnight_equals_rf']}; "
           f"`rf_intraday` is 0 on every day: {rf['intraday_zero']}",
           f"- Largest error in `overnight_xs = overnight - rf`: {rf['xs_check']:.1e}",
           "- Mean rf by calendar days covered (Fama-French rf is per trading day, so a weekend earns about "
           "the same as a weekday):", "", md_table(rf["by_day"], "{:.3f}"), "",
           "Annual levels and the 4-decimal rounding of the rf file are in the step 2 report (section 9).", ""]
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
- The dividend is backed out of CRSP's `ret - retx`, so `diff_ret` in check 4 really tests that `retx`
  matches the unadjusted prices and split factor.
- Days with no open (the first rows, SPY 1993-02-01, 1994-11-07 and the no-trade day 1997-03-31) have
  no overnight or intraday return, and the identity checks skip them.
- `rf` is per close-to-close period and is assigned entirely to the overnight leg, as before:
  `rf_overnight = rf`, `rf_intraday = 0`, and `overnight_xs = overnight - rf_overnight`,
  `intraday_xs = intraday` (arithmetic excess returns).
- The rf file is rounded to whole basis points per day (step 2), so annual rf is off by up to about
  1 percentage point in some years. This feeds every excess return built here.
- Nothing is dropped: flagged days stay in the file and later steps choose whether to exclude them.
"""


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "crsp")
