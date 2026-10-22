import sys

import numpy as np
import pandas as pd

from config import md_table, write_report
from data_layer import load_processed, load_vix, save_processed
from returns import sample_of

WINDOW = 252
MIN_INTRADAY_OBS = 250  # tolerate the few CRSP days with no open
PCT = 0.20
SIGNALS = ["sig_a", "sig_b", "sig_c", "sig_ac", "sig_bc"]
TESTED = SIGNALS + ["intraday_q20", "vix_median"]
EXEC_LAG = 1  # the decision on day t only uses data up to t-1's close

def _past(series, fn, lag=1, min_periods=WINDOW):
    return fn(series.shift(lag).rolling(WINDOW, min_periods=min_periods))

def _bool(cond, valid):
    return cond.astype("boolean").where(valid, pd.NA)

# lag=1 is the real signal, a negative lag peeks ahead on purpose to check the tests catch it
def compute_signals(ret, vix, lag=1):
    out = []
    v = vix.sort_index()
    vix_med = _past(v, lambda r: r.median(), lag)

    for t, g in ret.sort_values("date").groupby("ticker"):
        g = g[["date", "ticker", "intraday"]].copy()
        intra = g["intraday"]
        q20 = _past(intra, lambda r: r.quantile(PCT), lag, min_periods=MIN_INTRADAY_OBS)
        q20.iloc[:WINDOW] = np.nan  # never shorten the first window

        g["intraday_q20"] = q20.to_numpy()
        g["vix"] = v.reindex(g["date"]).to_numpy()
        g["vix_median"] = vix_med.reindex(g["date"]).to_numpy()
        g["trade_date"] = g["date"].shift(-1)

        g["sig_a"] = _bool(intra < 0, intra.notna())
        g["sig_b"] = _bool(intra <= g["intraday_q20"], g["intraday_q20"].notna() & intra.notna())
        g["sig_c"] = _bool(g["vix"] > g["vix_median"], g["vix_median"].notna() & g["vix"].notna())
        for a in ["a", "b"]:
            x, c = g[f"sig_{a}"], g["sig_c"]
            g[f"sig_{a}c"] = _bool(x.fillna(False) & c.fillna(False), x.notna() & c.notna())

        inputs = ["intraday", "intraday_q20", "vix", "vix_median"] + SIGNALS
        g[inputs] = g[inputs].shift(EXEC_LAG)
        out.append(g)

    return pd.concat(out, ignore_index=True)

def _same(a, b):
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    return bool(((a == b).fillna(False) | (a.isna() & b.isna())).all())

# Cut the data at T and recompute, nothing up to T may change
def truncation_test(ret, vix, cutoffs, lag=1):
    full = compute_signals(ret, vix, lag).set_index(["date", "ticker"]).sort_index()
    bad = []
    for T in cutoffs:
        part = compute_signals(ret[ret["date"] <= T], vix[vix.index <= T], lag).set_index(["date", "ticker"]).sort_index()
        if not all(_same(part[s], full.loc[part.index, s]) for s in TESTED):
            bad.append(T.date())
    return bad

# Replace everything after T with noise, nothing up to T may change
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

def manual_check(sig, ret, vix, ticker="SPY", date="2008-10-15"):
    g = ret[ret["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    i = g.index[g["date"] == date][0] - EXEC_LAG
    vi = vix.index.get_loc(pd.Timestamp(date)) - EXEC_LAG
    win, vwin = g["intraday"].iloc[i - WINDOW:i], vix.iloc[vi - WINDOW:vi]
    row = sig[(sig["ticker"] == ticker) & (sig["date"] == date)].iloc[0]
    return pd.DataFrame([
        {"item": "intraday(t-1)", "by_hand": g["intraday"].iloc[i], "stored": row["intraday"]},
        {"item": "q20 of the 252 days before", "by_hand": win.quantile(PCT), "stored": row["intraday_q20"]},
        {"item": "VIX(t-1)", "by_hand": vix.iloc[vi], "stored": row["vix"]},
        {"item": "median VIX of the 252 days before", "by_hand": vwin.median(), "stored": row["vix_median"]},
    ])

def report(source="crsp"):
    ret = load_processed(f"returns_{source}")
    vix = load_vix()
    vix = vix[vix.index <= ret["date"].max()]
    sig = compute_signals(ret, vix)
    sig = sig[sig["trade_date"].notna()].reset_index(drop=True)
    sig["sample"] = sample_of(sig["trade_date"])
    path = save_processed(sig, f"signals_{source}")

    rng = np.random.default_rng(42)
    dates = ret["date"].drop_duplicates().sort_values()
    dates = dates[dates > dates.iloc[WINDOW + 30]]
    cutoffs = sorted(set(pd.to_datetime(rng.choice(dates.to_numpy(), 20, replace=False)))
                     | {pd.Timestamp("2012-12-31"), pd.Timestamp("2018-12-31")})
    tests = pd.DataFrame({
        "test": ["truncation", "perturbation"],
        "real_signals_failures": [len(truncation_test(ret, vix, cutoffs)), len(perturbation_test(ret, vix, cutoffs))],
        "leaky_control_caught": [len(truncation_test(ret, vix, cutoffs, lag=-1 - EXEC_LAG)),
                                 len(perturbation_test(ret, vix, cutoffs, lag=-1 - EXEC_LAG))],
        "cutoffs": [len(cutoffs)] * 2})

    nxt = ret.sort_values("date").groupby("ticker")["date"].shift(-1)
    merged = sig.merge(ret[["date", "ticker"]].assign(next_session=nxt), on=["date", "ticker"])
    counts = sig.groupby(["sample", "ticker"])[SIGNALS].mean().reindex(["train", "validation", "test"], level=0)

    out = [f"# Step 4: signals ({source})", "",
           f"Generated by `python src/signals.py {source}`, {len(sig)} signal nights saved to `data/processed/{path.name}`.", "",
           "## Definitions", "",
           "A signal on day t decides whether to hold the ETF from t's close to the next open, and only uses data up "
           "to t-1's close so the order can be placed in time.", "",
           "- `sig_a`: intraday(t-1) < 0",
           "- `sig_b`: intraday(t-1) at or below the 20th percentile of the 252 days before it",
           "- `sig_c`: VIX(t-1) above its median over the 252 days before it",
           "- `sig_ac` = `sig_a` and `sig_c`, `sig_bc` = `sig_b` and `sig_c`", "",
           "## Look-ahead tests", "",
           "Truncation recomputes with data up to T only, perturbation replaces everything after T with noise. "
           "The leaky control peeks one day ahead and must be caught.", "",
           md_table(tests), "",
           f"`trade_date` is always the next session: {bool((merged['trade_date'] == merged['next_session']).all())}", "",
           "Hand check, SPY on 2008-10-15:", "",
           md_table(manual_check(sig, ret, vix), "{:.6f}"), "",
           "## Share of nights each signal is on", "",
           md_table(counts, "{:.3f}")]
    write_report("step4_signals.md", out)

if __name__ == "__main__":
    report(*sys.argv[1:2])
