# Step 8: final test on 2019-2025, run once

import pickle
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest import SETUP, TICKERS, backtest, check_engine, load_sample, md_table, _flat
from config import CHOSEN_SIGNAL, END, PROCESSED, ROOT, SAMPLES
from data_layer import load_processed
from hypothesis_tests import (BP, EXCLUDE_FLAGS, daily, h1, h2_main_table, h2_regression, load_train,
                              nights, nw_lags, stars)

SAMPLE = "test"
PERIODS = {"2019-2025": ("2019-01-01", END), "2019-2020": ("2019-01-01", "2020-12-31"), "2021+": ("2021-01-01", END)}
H4_PERIODS = {"2019-2020": PERIODS["2019-2020"], "2021+": PERIODS["2021+"]}
RESULTS = PROCESSED / "step8_results.pkl"
STRATS = ["buy_hold", "intraday", "overnight", f"on_{CHOSEN_SIGNAL}"]


# Returns and signals for one sample, with the step 5 flag column
def load_sample_rows(source, sample, periods):
    ret = load_processed(f"returns_{source}")
    ret = ret[ret["sample"] == sample].copy()
    sig = load_processed(f"signals_{source}")
    sig = sig[sig["sample"] == sample].copy()
    ret["flagged"] = ret[EXCLUDE_FLAGS].any(axis=1)

    lo = min(pd.Timestamp(a) for a, _ in periods.values())
    hi = max(pd.Timestamp(b) for _, b in periods.values())
    assert ret["date"].min() >= lo and ret["date"].max() <= hi
    return ret, sig


# H4

# Gap in each period minus the train gap, Newey-West on the stacked series
def h1_change_vs_train(train, test, exclude, periods):
    rows = []
    for t in TICKERS:
        a = daily(train, exclude)
        b = daily(test, exclude)
        g = pd.concat([a[a["ticker"] == t], b[b["ticker"] == t]]).sort_values("date")

        # skip days with no open (three SPY train days), otherwise the whole regression is NaN
        g = g[(g["overnight_xs"] - g["intraday_xs"]).notna()]
        y = (g["overnight_xs"] - g["intraday_xs"]).to_numpy()

        dummies = np.column_stack([((g["date"] >= p0) & (g["date"] <= p1)).to_numpy(float)
                                   for p0, p1 in periods.values()])
        X = sm.add_constant(dummies)
        r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(y))})

        row = {"ticker": t, "train_gap_bp": r.params[0] * BP}
        for i, name in enumerate(periods, start=1):
            row[f"{name} minus train_bp"] = r.params[i] * BP
            row[f"t ({name})"] = r.tvalues[i]
            row[f"p ({name})"] = r.pvalues[i]

        # 2021+ against 2019-2020
        c = np.zeros(len(r.params))
        c[1], c[2] = -1, 1
        w = r.t_test(c)
        row["2021+ minus 2019-2020_bp"] = float(np.squeeze(w.effect)) * BP
        row["p (2021+ vs 2019-2020)"] = float(np.squeeze(w.pvalue))
        rows.append(row)
    return pd.DataFrame(rows)


def h2_by_period(n_train, n_test, periods):
    rows = []
    for name, n in [("train 1993-2012", n_train)] + [
            (p, n_test[(n_test["trade_date"] >= a) & (n_test["trade_date"] <= b)]) for p, (a, b) in periods.items()]:
        tab, _ = h2_regression(n)
        rows.append(h2_main_table(tab).assign(period=name))

    out = pd.concat(rows, ignore_index=True)
    out["ticker"] = pd.Categorical(out["ticker"], TICKERS, ordered=True)
    out = out.sort_values("ticker", kind="stable")  # per ETF, train first
    out["ticker"] = out["ticker"].astype(str)
    return out[["period"] + [c for c in out.columns if c != "period"]]


# Pooled train plus period regression, Wald test that the slopes match train
def h2_change_vs_train(n_train, n_test, periods):
    regs = ["intraday", "sig_b", "vix"]
    rows = []
    for t in TICKERS:
        for name, (a, b) in periods.items():
            p = n_test[(n_test["ticker"] == t) & (n_test["trade_date"] >= a) & (n_test["trade_date"] <= b)]
            g = pd.concat([n_train[n_train["ticker"] == t].assign(post=0.0), p.assign(post=1.0)])
            g = g.dropna(subset=["overnight_xs"] + regs).sort_values("trade_date")

            X = np.column_stack([np.ones(len(g))] + [g[r] for r in regs] + [g["post"]] + [g["post"] * g[r] for r in regs])
            r = sm.OLS(g["overnight_xs"].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(g))})

            R = np.zeros((3, 8))
            R[[0, 1, 2], [5, 6, 7]] = 1
            w = r.wald_test(R, scalar=True)
            rows.append({"ticker": t, "period": name,
                         "d_intraday (bp per 1%)": r.params[5] * BP / 100, "t_intraday": r.tvalues[5],
                         "d_sig_b_bp": r.params[6] * BP, "t_sig_b": r.tvalues[6],
                         "d_vix (bp per point)": r.params[7] * BP, "t_vix": r.tvalues[7],
                         "p (3 slopes = train)": float(w.pvalue)})

    out = pd.DataFrame(rows)
    out["sig"] = out["p (3 slopes = train)"].map(stars)
    return out


# Run once

def compute(source="crsp", sample=SAMPLE, periods=PERIODS, h4_periods=H4_PERIODS, out=RESULTS):
    res = {"source": source, "sample": sample, "signal": CHOSEN_SIGNAL, "periods": periods,
           "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                    cwd=ROOT).stdout.strip()}

    ret, sig = load_sample(source, sample)
    res["engine_check"] = check_engine(ret, sig, sample)
    res["backtest"] = {p: backtest(source, sample, [CHOSEN_SIGNAL], w) for p, w in periods.items()}

    train_ret, train_sig = load_train(source)
    test_ret, test_sig = load_sample_rows(source, sample, periods)
    h4 = {}
    for exclude in [False, True]:
        n_train, n_test = nights(train_ret, train_sig, exclude), nights(test_ret, test_sig, exclude)
        h4[exclude] = {
            "h1": pd.concat([h1(train_ret, exclude), h1(test_ret, exclude, h4_periods)], ignore_index=True),
            "h1_change": h1_change_vs_train(train_ret, test_ret, exclude, h4_periods),
            "h2": h2_by_period(n_train, n_test, h4_periods),
            "h2_change": h2_change_vs_train(n_train, n_test, h4_periods),
        }
    res["h4"] = h4

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(res, f)
    print(f"saved {out}")
    return res


# Report

def bt_section(perf, be, plc, win):
    cols = ["strategy", "cum_ret_%", "ann_ret_%", "sharpe", "max_dd_%", "nights_in_%", "trades_per_yr"]
    out = [md_table(win, "{:.2f}"), ""]
    for t in TICKERS:
        out += [f"**{t}**", ""]
        rows = []
        for c in ["0 bp", "1 bp", "2 bp", "5 bp", "10 bp", "spread est."]:
            p = perf[(perf["ticker"] == t) & (perf["cost"] == c)].set_index("strategy").loc[STRATS].reset_index()
            rows.append(p[cols].assign(cost=c))
        tab = pd.concat(rows)
        out += [md_table(tab[["cost"] + cols], "{:.2f}"), ""]

    be_tab = be.pivot(index="strategy", columns="ticker",
                      values=["breakeven_vs_cash_bp", "breakeven_vs_buy_hold_bp"]).pipe(_flat).loc[STRATS[1:]]
    out += ["Breakeven cost per trade (bp; `-` = below the benchmark even at zero cost):", "",
            md_table(be_tab.round(2).astype(object).where(be_tab.notna(), "-"), "{:.2f}"), "",
            f"Placebo for `{CHOSEN_SIGNAL}` (1,000 random sets of the same number of nights, 0 bp):", "",
            md_table(plc, "{:.2f}"), ""]
    return out


def h4_section(h, label):
    return [f"### {label}", "",
            "**H1 overnight vs intraday excess return** (bp per day, Newey-West; train rows from step 5):", "",
            md_table(h["h1"], "{:.2f}"), "",
            "Change in the overnight-intraday gap vs train (stacked regression, Newey-West):", "",
            md_table(h["h1_change"], "{:.2f}"), "",
            "**H2 regression** `overnight_xs(after t) = a + b1 intraday(t-1) + b2 sig_b(t) + b3 VIX(t-1)` by period:", "",
            md_table(h["h2"], "{:.2f}"), "",
            "Change in the H2 slopes vs train (pooled regression with period interactions, Newey-West; "
            "`p` tests all three slopes equal to train):", "",
            md_table(h["h2_change"], "{:.3f}"), ""]


def render(path=RESULTS, report=ROOT / "reports" / "step8_test.md", notes=""):
    with open(path, "rb") as f:
        res = pickle.load(f)

    a, b = res["periods"][next(iter(res["periods"]))]
    out = [f"# Step 8: final test ({res['source']}, {a} to {b})", "",
           f"Computed by `python src/test_period.py run` at {res['run_utc']} UTC on commit `{res['commit']}`; "
           "this report is rendered from the saved results (`python src/test_period.py report`). Everything was "
           f"frozen beforehand: signal `{res['signal']}` (`config.CHOSEN_SIGNAL`, step 7), the step 6 engine and "
           "cost grid, and the step 5 test code. Nothing was changed after the run.", ""]
    if notes:
        out += [notes, ""]
    out += SETUP
    out += ["Engine checks (SPY, whole test period):", "", md_table(res["engine_check"], "{:.8f}"), ""]
    out += ["## Backtests", ""]
    for p, (perf, be, plc, win, _) in res["backtest"].items():
        out += [f"### {p}", ""] + bt_section(perf, be, plc, win)
    out += ["## H4: has the overnight effect changed?", "",
            "H1 and the H2 regression repeated on 2019-2020 and 2021+, compared with the train sample (1993-2012). "
            f"Run on all days and excluding step 2 flagged days ({', '.join(f'`{f}`' for f in EXCLUDE_FLAGS)}). "
            "Significance: * 10%, ** 5%, *** 1%.", ""]
    out += h4_section(res["h4"][False], "All days")
    out += h4_section(res["h4"][True], "Excluding flagged days")

    report.write_text("\n".join(out))
    print(f"wrote {report.relative_to(ROOT) if report.is_relative_to(ROOT) else report}")


# The Yahoo results and their summary are in the git history of reports/step8_test.md
NOTES = """## Note on this run: not a clean out-of-sample test

This is the third CRSP run of the final test, and the method was changed after the earlier results had been seen.

1. **First run** (2026-10-19 14:35 UTC, `97b2d35`, sha256 `0b642f3c2a453a90...`): signals measured at the same close they
   trade at, validation picked `sig_b`. The SPY row of the H4 change-vs-train table was empty because of a bug.
2. **Second run** (2026-10-19 15:52 UTC, `f35fa86`, sha256 `f6c7cd21e7f305b8...`): the same, with only that bug fixed.
   Every other number was identical to the first run.
3. **This run**, after two changes to the method:
   - signals use data up to the previous close, so the order can be placed before the close, and
   - the "excluding flagged days" runs drop only data problems (stale opens, bad prices, OHLC errors, half-days,
     splits), no longer days flagged as extreme moves or reversals, which are judged from the returns themselves.

   With the same selection rule, validation now picks `sig_c`. Both changes fix problems a reviewer raised, but
   they were made after the 2019-2025 results for `sig_b` were known, so these results are a re-analysis of the
   test period, not an untouched holdout. The earlier results files are kept as `step8_results_run1.pkl` and
   `step8_results_run2.pkl`, and the earlier reports are in the git history.

Before all of this the test was run once on Yahoo data (2019-01-01 to 2026-10-02, 2026-10-16).
"""


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"

    # the test runs once, refuse to overwrite saved results
    if cmd == "run":
        if RESULTS.exists():
            raise SystemExit(f"{RESULTS.name} already exists: the test is run once. Delete it only on purpose.")
        compute()
    render(notes=NOTES)
