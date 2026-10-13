# Step 8: final test on 2019-2026, run once

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
PERIODS = {"2019-2026": ("2019-01-01", END), "2019-2020": ("2019-01-01", "2020-12-31"), "2021+": ("2021-01-01", END)}
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

def compute(source="yfinance", sample=SAMPLE, periods=PERIODS, h4_periods=H4_PERIODS, out=RESULTS):
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
            "**H2 regression** `overnight_xs(t+1) = a + b1 intraday(t) + b2 sig_b(t) + b3 VIX(t)` by period:", "",
            md_table(h["h2"], "{:.2f}"), "",
            "Change in the H2 slopes vs train (pooled regression with period interactions, Newey-West; "
            "`p` tests all three slopes equal to train):", "",
            md_table(h["h2_change"], "{:.3f}"), ""]


def render(path=RESULTS, report=ROOT / "reports" / "step8_test.md", notes=""):
    with open(path, "rb") as f:
        res = pickle.load(f)

    a, b = res["periods"][next(iter(res["periods"]))]
    out = [f"# Step 8: final test ({res['source']}, {a} to {b})", "",
           f"Computed once by `python src/test_period.py run` at {res['run_utc']} UTC on commit `{res['commit']}`; "
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


NOTES = """## Results in brief (written after the single run; nothing was recomputed)

**Backtests, 2019-01-02 to 2026-10-02, before costs (0 bp)**

| | Buy-and-hold | Always overnight | Always intraday | `sig_a` | `sig_a` placebo percentile |
|---|---|---|---|---|---|
| SPY | 17.4% / 0.78 / -34% | 10.1% / 0.60 / -29% | 9.6% / 0.52 / -14% | 7.1% / 0.50 / -13% | 66th |
| QQQ | 23.4% / 0.88 / -35% | 13.2% / 0.72 / -27% | 12.2% / 0.56 / -24% | 7.7% / 0.49 / -17% | 55th |
| IWM | 11.5% / 0.45 / -41% | 17.6% / 0.92 / -29% | -2.6% / -0.19 / -48% | 10.5% / 0.69 / -18% | 55th |

(annual return / Sharpe / max drawdown; placebo percentile on annual return)

- **The chosen signal has no edge out of sample.** `sig_a` sits at the 55th-66th percentile of 1,000 random sets
  of nights, with lower return and Sharpe than always-overnight before costs on every ETF. Because it trades
  less than half as often, it loses less than always-overnight once costs are 2 bp or more, but it never beats
  buy-and-hold, and it stops beating cash at 1.8-2.8 bp per trade. It was 2019-2020 that went worst for it (27th
  percentile on QQQ); in 2021+ it was at the 74th-75th percentile for SPY and QQQ, still well short of significant.
- **Always-overnight:** for SPY and QQQ the overnight and intraday legs earned about the same in 2019-2026, so
  the overnight-only strategy lost to buy-and-hold even before costs. IWM is the exception: overnight 17.6% a
  year against -2.6% intraday and 11.5% buy-and-hold, with twice the buy-and-hold Sharpe, but the edge over
  buy-and-hold disappears at about 1.1 bp per trade and over cash at 2.7 bp.
- **Costs dominate**, as in train and validation: at 2 bp per trade every overnight strategy is at or below cash
  on SPY and QQQ; at 5 bp every active strategy loses money. The Corwin-Schultz numbers are again a pessimistic
  bound, not a realistic cost (step 6).

**H4: has the overnight effect changed?**

- **The overnight-intraday gap is no longer significant for SPY and QQQ.** Daily gap in train vs 2019-2020 vs
  2021+: SPY 3.3 / 1.9 / -0.5 bp, QQQ 7.9 / 0.1 / 0.2 bp (all days); none of the test-period gaps is significant.
  IWM keeps a large gap (11.7 bp in 2019-2020, 5.7 bp in 2021+), significant at 10% / 5% in some
  specifications. But the tests of a *change* versus train are not significant (smallest p = 0.09, QQQ 2021+), so
  the data are consistent with the effect having weakened for large caps, without proving that it has.
  Excluding flagged days gives the same picture.
- **H2 holds outside 2020.** The H2 regression is as uninformative in 2021+ as in train (slopes equal to train,
  p = 0.13-0.91). 2019-2020 is the exception: for SPY and QQQ a `sig_b` day was followed by a much worse night
  (SPY -57 bp, t = -3.2), so the slopes differ from train (SPY p = 0.004). That is the March 2020 crash, where
  large intraday falls were followed by further falls overnight, and it goes in the opposite direction to the
  "buy after a bad day" idea.

**Bottom line.** The overnight premium was large and significant in 1993-2007, weaker after, and in 2019-2026 it
survives only in small caps (IWM). None of the five conditional signals beat random nights in train, the one
picked on validation did not beat random nights in the test, and at realistic trading costs (2 bp or more per
trade, i.e. 4 bp per night) no overnight strategy beats buy-and-hold. All results are on Yahoo data and should be
re-run once CRSP is available, without changing anything else.
"""


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"

    # the test runs once, refuse to overwrite saved results
    if cmd == "run":
        if RESULTS.exists():
            raise SystemExit(f"{RESULTS.name} already exists: the test is run once. Delete it only on purpose.")
        compute()
    render(notes=NOTES)
