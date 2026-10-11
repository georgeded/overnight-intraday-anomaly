# Step 4: overnight signals, using only what is known at each close
#
#     python src/signals.py          # yfinance
#     python src/signals.py crsp
#
# A signal on day t decides whether to hold the ETF from t's close to the next open,
# so it gets paired with the overnight return on the next row (trade_date)
#
#     sig_a   intraday(t) < 0
#     sig_b   intraday(t) at or below the 20th percentile of the previous 252 days
#     sig_c   VIX(t) above its median over the previous 252 days
#     sig_ac  sig_a and sig_c
#     sig_bc  sig_b and sig_c
#
# Signals stay NaN until a full 252 day window exists

import sys

import numpy as np
import pandas as pd

from checks import md_table
from config import ROOT
from data_layer import load_processed, load_vix, save_processed
from returns import sample_of

WINDOW = 252
PCT = 0.20
SIGNALS = ["sig_a", "sig_b", "sig_c", "sig_ac", "sig_bc"]
# checked in the look-ahead tests, the signals plus their thresholds
TESTED = SIGNALS + ["intraday_q20", "vix_median"]


# Rolling statistic over the WINDOW values before each date
def _past(series, fn, lag=1):
    return fn(series.shift(lag).rolling(WINDOW, min_periods=WINDOW))


# lag=1 is the real signal
# lag=-1 peeks one day ahead on purpose, to show the tests catch leaks
def compute_signals(ret, vix, lag=1):
    out = []
    v = vix.sort_index()
    vix_med = _past(v, lambda r: r.median(), lag)

    for t, g in ret.sort_values("date").groupby("ticker"):
        g = g[["date", "ticker", "intraday"]].copy()
        intra = g["intraday"]

        g["intraday_q20"] = _past(intra, lambda r: r.quantile(PCT), lag).to_numpy()
        g["vix"] = v.reindex(g["date"]).to_numpy()
        g["vix_median"] = vix_med.reindex(g["date"]).to_numpy()
        g["trade_date"] = g["date"].shift(-1)  # night from date's close to trade_date's open

        g["sig_a"] = _bool(intra < 0, intra.notna())
        g["sig_b"] = _bool(intra <= g["intraday_q20"], g["intraday_q20"].notna() & intra.notna())
        g["sig_c"] = _bool(g["vix"] > g["vix_median"], g["vix_median"].notna() & g["vix"].notna())
        # NA if either side is NA, pandas would give False & NA = False
        g["sig_ac"] = _bool(g["sig_a"].fillna(False) & g["sig_c"].fillna(False), g["sig_a"].notna() & g["sig_c"].notna())
        g["sig_bc"] = _bool(g["sig_b"].fillna(False) & g["sig_c"].fillna(False), g["sig_b"].notna() & g["sig_c"].notna())
        out.append(g)

    return pd.concat(out, ignore_index=True)


def _bool(cond, valid):
    return cond.astype("boolean").where(valid, pd.NA)


# Look-ahead tests

# Element-wise equal, with NA == NA counted as equal
def _same(a, b):
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    return bool(((a == b).fillna(False) | (a.isna() & b.isna())).all())


# Cut the data at T and recompute, signals up to T must not change
def truncation_test(ret, vix, cutoffs, lag=1):
    full = compute_signals(ret, vix, lag).set_index(["date", "ticker"]).sort_index()
    bad = []
    for T in cutoffs:
        part = compute_signals(ret[ret["date"] <= T], vix[vix.index <= T], lag).set_index(["date", "ticker"]).sort_index()
        ref = full.loc[part.index]
        if not all(_same(part[s], ref[s]) for s in TESTED):
            bad.append(T.date())
    return bad


# Replace everything after T with noise, signals up to T must not change
def perturbation_test(ret, vix, cutoffs, seed=0, lag=1):
    rng = np.random.default_rng(seed)
    full = compute_signals(ret, vix, lag)
    bad = []
    for T in cutoffs:
        r2, v2 = ret.copy(), vix.copy()
        after = r2["date"] > T
        r2.loc[after, "intraday"] = rng.normal(0, 0.05, after.sum())
        v2[v2.index > T] = rng.uniform(5, 80, (v2.index > T).sum())
        alt = compute_signals(r2, v2, lag)
        keep = full["date"] <= T
        if not all(_same(alt.loc[keep, s], full.loc[keep, s]) for s in TESTED):
            bad.append(T.date())
    return bad


# trade_date is the next session and windows hold exactly 252 past days
def timing_checks(sig, ret):
    nxt = ret.sort_values("date").groupby("ticker")["date"].shift(-1)
    nxt.index = ret.sort_values("date").index
    merged = sig.merge(ret[["date", "ticker"]].assign(next_session=nxt), on=["date", "ticker"])

    ok_next = (merged["trade_date"].isna() & merged["next_session"].isna()) | (merged["trade_date"] == merged["next_session"])
    first_b = sig[sig["sig_b"].notna()].groupby("ticker")["date"].min()
    first_rows = ret.groupby("ticker")["date"].apply(lambda d: d.sort_values().iloc[WINDOW])

    return {
        "trade_after_signal": bool((sig["trade_date"].dropna() > sig.loc[sig["trade_date"].notna(), "date"]).all()),
        "trade_is_next_session": bool(ok_next.all()),
        "sig_b_starts_on_row_253": bool((first_b == first_rows).all()),
        "first_sig_b": ", ".join(f"{t} {d.date()}" for t, d in first_b.items()),
        "first_sig_c": ", ".join(f"{t} {d.date()}" for t, d in sig[sig["sig_c"].notna()].groupby("ticker")["date"].min().items()),
    }


# Recompute one day by hand from the raw windows
def manual_check(sig, ret, vix, ticker="SPY", date="2008-10-15"):
    g = ret[ret["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    i = g.index[g["date"] == date][0]
    win = g["intraday"].iloc[i - WINDOW:i]

    vi = vix.index.get_loc(pd.Timestamp(date))
    vwin = vix.iloc[vi - WINDOW:vi]

    row = sig[(sig["ticker"] == ticker) & (sig["date"] == date)].iloc[0]
    return pd.DataFrame([
        {"item": "intraday(t)", "by_hand": g["intraday"].iloc[i], "stored": row["intraday"]},
        {"item": "q20 of previous 252 intraday", "by_hand": win.quantile(PCT), "stored": row["intraday_q20"]},
        {"item": "window first / last date", "by_hand": f"{g['date'].iloc[i - WINDOW].date()} / {g['date'].iloc[i - 1].date()}", "stored": ""},
        {"item": "VIX(t)", "by_hand": vix.iloc[vi], "stored": row["vix"]},
        {"item": "median of previous 252 VIX", "by_hand": vwin.median(), "stored": row["vix_median"]},
        {"item": "sig_a / sig_b / sig_c", "by_hand": f"{g['intraday'].iloc[i] < 0} / {g['intraday'].iloc[i] <= win.quantile(PCT)} / {vix.iloc[vi] > vwin.median()}",
         "stored": f"{row['sig_a']} / {row['sig_b']} / {row['sig_c']}"},
    ])


# How often signals fire

# Share of nights each signal fires, by year
def frequency(sig):
    s = sig.assign(year=sig["date"].dt.year)
    share = s.groupby(["year", "ticker"])[SIGNALS].mean().unstack("ticker")
    share.columns = [f"{t} {c[4:]}" for c, t in share.columns]
    order = [f"{t} {x}" for t in ["SPY", "QQQ", "IWM"] for x in ["a", "b", "c", "ac", "bc"]]
    return share[order]


def counts_by_sample(sig):
    rows = []
    for (smp, t), g in sig.groupby(["sample", "ticker"]):
        r = {"sample": smp, "ticker": t, "nights": len(g)}
        for c in SIGNALS:
            r[f"{c} fired"] = int(g[c].sum())
            r[f"{c} share"] = g[c].mean()
        rows.append(r)

    order = {"train": 0, "validation": 1, "test": 2}
    return pd.DataFrame(rows).sort_values(["sample", "ticker"], key=lambda c: c.map(order) if c.name == "sample" else c)


# Report

def report(source="yfinance"):
    ret = load_processed(f"returns_{source}")
    vix = load_vix()
    vix = vix[vix.index <= ret["date"].max()]
    sig = compute_signals(ret, vix)
    sig = sig[sig["trade_date"].notna()].reset_index(drop=True)  # last day has no next night
    sig["sample"] = sample_of(sig["trade_date"])  # sample of the night traded
    path = save_processed(sig, f"signals_{source}")

    rng = np.random.default_rng(42)
    dates = ret["date"].drop_duplicates().sort_values()
    dates = dates[dates > dates.iloc[WINDOW + 30]]
    cutoffs = sorted(set(pd.to_datetime(rng.choice(dates.to_numpy(), 20, replace=False)))
                     | {pd.Timestamp("2012-12-31"), pd.Timestamp("2018-12-31")})

    trunc = truncation_test(ret, vix, cutoffs)
    pert = perturbation_test(ret, vix, cutoffs)
    leak_trunc = truncation_test(ret, vix, cutoffs, lag=-1)
    leak_pert = perturbation_test(ret, vix, cutoffs, lag=-1)

    timing = timing_checks(sig, ret)
    hand = manual_check(sig, ret, vix)
    freq = frequency(sig)
    counts = counts_by_sample(sig)
    nsig = sig[SIGNALS].notna()
    overlap = pd.DataFrame({
        "Nights with sig_b on but sig_a off (possible only if the 20th percentile is positive)": [int((sig["sig_b"].fillna(False) & ~sig["sig_a"].fillna(False)).sum())],
    })

    out = [f"# Step 4: signals ({source})", "",
           f"Generated by `python src/signals.py {source}`. Output: `data/processed/{path.name}` "
           f"({len(sig)} signal nights), saved with `data_layer.save_processed`. No returns analysis here.", "",
           "## Definitions", "",
           "Each signal is dated t, computed from data available at t's close, and decides whether to hold "
           "the ETF over the night from t's close to the next session's open (`trade_date`). "
           "In step 5, `signal(t)` is paired with `overnight` on the `trade_date` row.", "",
           "- `sig_a`: intraday(t) < 0",
           "- `sig_b`: intraday(t) ≤ 20th percentile (linear interpolation) of the same ETF's intraday returns over "
           "the previous 252 sessions, t-252 to t-1 (today is excluded from its own window)",
           "- `sig_c`: VIX close(t) > median VIX close over the previous 252 NYSE sessions, t-252 to t-1. VIX is the "
           "CBOE close per NYSE session, gaps filled from FRED (see `data_layer.load_vix`); the same series for all ETFs",
           "- `sig_ac` = `sig_a` AND `sig_c`; `sig_bc` = `sig_b` AND `sig_c`. These five are the only candidates, "
           "fixed before any returns were examined. A combination is NA if either part is NA.",
           "- Until a full 252-day window exists the signal is missing (NA), not False. "
           f"First `sig_b`: {timing['first_sig_b']}. First `sig_c`: {timing['first_sig_c']}.",
           "- `sample` is the sample of the night traded (`trade_date`), so a signal on 2012-12-31 belongs to validation.",
           "- The last day in the sample (2026-10-02) has no next night inside the sample and is left out.",
           "- Step 3 now assigns all T-bill interest to the overnight leg (`rf_overnight = rf`, `rf_intraday = 0`, "
           "`overnight_xs`, `intraday_xs`).", "",
           "## Look-ahead tests", "",
           f"Run at {len(cutoffs)} cut-off dates (20 random plus the train/validation and validation/test boundaries).", "",
           "Each test compares the five signals and the two thresholds behind them (`intraday_q20`, `vix_median`) exactly.", "",
           "| Test | Real signals | Leaky control (window shifted one day into the future) |",
           "|---|---|---|",
           f"| Truncation: recompute with data up to T only; signals on or before T unchanged | "
           f"{'pass' if not trunc else 'FAIL at ' + str(trunc)} | "
           f"{'caught at ' + str(len(leak_trunc)) + ' of ' + str(len(cutoffs)) + ' cut-offs' if leak_trunc else 'NOT caught'} |",
           f"| Perturbation: replace all data after T with noise; signals on or before T unchanged | "
           f"{'pass' if not pert else 'FAIL at ' + str(pert)} | "
           f"{'caught at ' + str(len(leak_pert)) + ' of ' + str(len(cutoffs)) + ' cut-offs' if leak_pert else 'NOT caught'} |", "",
           "The leaky control is missed at a few cut-offs in the perturbation test only when the one future value "
           "it peeks at does not move the 20th percentile or the median; truncation catches it every time.", "",
           f"- `trade_date` is after the signal date on every row: {timing['trade_after_signal']}; "
           f"it is the ETF's next session: {timing['trade_is_next_session']}",
           f"- `sig_b` first appears on each ETF's 253rd row (a full 252-day window before it): {timing['sig_b_starts_on_row_253']}",
           f"- {overlap.columns[0]}: {overlap.iloc[0, 0]}", "",
           "Hand check, SPY on 2008-10-15:", "", md_table(hand, "{:.6f}"), "",
           "## How often each signal fires", "",
           "Share of nights with the signal on, by year of the signal date (NA nights excluded):", "",
           md_table(freq.rename_axis("year"), "{:.2f}"), "",
           "By sample:", "", md_table(counts, "{:.3f}"), "",
           NOTES]

    p = ROOT / "reports" / "step4_signals.md"
    p.write_text("\n".join(out))
    print(f"wrote {p.relative_to(ROOT)} and {path.relative_to(ROOT)}")


NOTES = """## Notes and open points

- `sig_a` fires on roughly half of nights by construction; `sig_b` on about 20% on average, but more in calm years
  after a volatile year and less in the reverse case, because its threshold comes from the past year; `sig_c`
  clusters in long high-volatility stretches (it can stay on or off for months).
- Timing at the close: `intraday(t)` needs the closing price, and the official VIX close is struck at 16:15 ET, after
  the 16:00 ETF close. A trader placing a market-on-close order (cut-off 15:50) knows neither exactly. With daily data
  this is the standard approximation; the practical fix is a 15:50 price, or using VIX(t-1) as a conservative
  variant of `sig_c`. Flagged for the implementation and cost step rather than changed here.
- Flagged days from step 2 are not excluded here. On a stale-open day the intraday return is the whole day's price
  move, so `sig_a`/`sig_b` still reflect a real fall in price; step 5 can test excluding them.
- Frequencies for 2019+ are shown because they describe the rule, not its returns; no test-period returns were looked at.
"""


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "yfinance")
