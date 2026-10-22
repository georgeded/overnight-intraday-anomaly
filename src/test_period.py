import pickle
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest import TICKERS, backtest, body, check_engine, load_sample
from config import CHOSEN_SIGNAL, END, PROCESSED, ROOT, md_table, write_report
from data_layer import load_processed
from hypothesis_tests import BP, EXCLUDE_FLAGS, daily, h1, h2_regression, load_train, nights, nw_lags, stars

PERIODS = {"2019-2025": ("2019-01-01", END), "2019-2020": ("2019-01-01", "2020-12-31"), "2021+": ("2021-01-01", END)}
H4_PERIODS = {"2019-2020": PERIODS["2019-2020"], "2021+": PERIODS["2021+"]}
RESULTS = PROCESSED / "step8_results.pkl"

NOTE = ("**Note:** this is the third CRSP run. The signal timing and the flagged-day rule were changed after earlier "
        "test results had been seen, so it is not a clean out-of-sample test. The earlier results are kept as "
        "`step8_results_run1.pkl` and `step8_results_run2.pkl`.")

def load_test(source):
    ret = load_processed(f"returns_{source}")
    sig = load_processed(f"signals_{source}")
    ret, sig = ret[ret["sample"] == "test"].copy(), sig[sig["sample"] == "test"].copy()
    ret["flagged"] = ret[EXCLUDE_FLAGS].any(axis=1)
    return ret, sig

# H4: the gap in each test period minus the train gap, Newey-West on the stacked series
def h1_change_vs_train(train, test, exclude, periods):
    rows = []
    for t in TICKERS:
        a, b = daily(train, exclude), daily(test, exclude)
        g = pd.concat([a[a["ticker"] == t], b[b["ticker"] == t]]).sort_values("date")
        g = g[(g["overnight_xs"] - g["intraday_xs"]).notna()]
        y = (g["overnight_xs"] - g["intraday_xs"]).to_numpy()
        X = sm.add_constant(np.column_stack([((g["date"] >= p0) & (g["date"] <= p1)).to_numpy(float) for p0, p1 in periods.values()]))
        r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(y))})

        row = {"ticker": t, "train_gap_bp": r.params[0] * BP}
        for i, name in enumerate(periods, start=1):
            row.update({f"{name} minus train_bp": r.params[i] * BP, f"t ({name})": r.tvalues[i], f"p ({name})": r.pvalues[i]})
        w = r.t_test(np.array([0, -1, 1]))
        row.update({"2021+ minus 2019-2020_bp": float(np.squeeze(w.effect)) * BP, "p (2021+ vs 2019-2020)": float(np.squeeze(w.pvalue))})
        rows.append(row)
    return pd.DataFrame(rows)

def h2_by_period(n_train, n_test, periods):
    parts = [("train 1993-2012", n_train)] + [(p, n_test[n_test["trade_date"].between(a, b)]) for p, (a, b) in periods.items()]
    out = pd.concat([h2_regression(n).assign(period=name) for name, n in parts], ignore_index=True)
    out = out.iloc[out["ticker"].map(TICKERS.index).argsort(kind="stable")]
    return out[["period"] + [c for c in out.columns if c != "period"]]

# Pooled train and test regression, Wald test that the three slopes match train
def h2_change_vs_train(n_train, n_test, periods):
    regs = ["intraday", "sig_b", "vix"]
    rows = []
    for t in TICKERS:
        for name, (a, b) in periods.items():
            p = n_test[(n_test["ticker"] == t) & n_test["trade_date"].between(a, b)]
            g = pd.concat([n_train[n_train["ticker"] == t].assign(post=0.0), p.assign(post=1.0)])
            g = g.dropna(subset=["overnight_xs"] + regs).sort_values("trade_date")
            X = np.column_stack([np.ones(len(g))] + [g[r] for r in regs] + [g["post"]] + [g["post"] * g[r] for r in regs])
            r = sm.OLS(g["overnight_xs"].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(g))})
            R = np.zeros((3, 8))
            R[[0, 1, 2], [5, 6, 7]] = 1
            rows.append({"ticker": t, "period": name, "d_intraday (bp per 1%)": r.params[5] * BP / 100, "t_intraday": r.tvalues[5],
                         "d_sig_b_bp": r.params[6] * BP, "t_sig_b": r.tvalues[6], "d_vix (bp per point)": r.params[7] * BP,
                         "t_vix": r.tvalues[7], "p (3 slopes = train)": float(r.wald_test(R, scalar=True).pvalue)})
    out = pd.DataFrame(rows)
    out["sig"] = out["p (3 slopes = train)"].map(stars)
    return out

def h4(train, test, exclude):
    n_train, n_test = nights(*train, exclude), nights(*test, exclude)
    return {"h1": pd.concat([h1(train[0], exclude), h1(test[0], exclude, H4_PERIODS)], ignore_index=True),
            "h1_change": h1_change_vs_train(train[0], test[0], exclude, H4_PERIODS),
            "h2": h2_by_period(n_train, n_test, H4_PERIODS),
            "h2_change": h2_change_vs_train(n_train, n_test, H4_PERIODS)}

def compute(source="crsp"):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    ret, sig = load_sample(source, "test")
    train, test = load_train(source), load_test(source)
    res = {"source": source, "signal": CHOSEN_SIGNAL, "periods": PERIODS, "commit": commit,
           "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
           "engine_check": check_engine(ret, sig, "test"),
           "backtest": {p: backtest(source, "test", [CHOSEN_SIGNAL], w) for p, w in PERIODS.items()},
           "h4": {exclude: h4(train, test, exclude) for exclude in [False, True]}}
    with open(RESULTS, "wb") as f:
        pickle.dump(res, f)
    print(f"saved {RESULTS}")

# Older saved results use the previous column names
def _rename(perf, be):
    return (perf.assign(cost=perf["cost"].replace({"spread est.": "spread"})),
            be.rename(columns={"breakeven_vs_cash_bp": "vs_cash", "breakeven_vs_buy_hold_bp": "vs_buy_hold"}))

def render():
    with open(RESULTS, "rb") as f:
        res = pickle.load(f)
    names = ["buy_hold", "intraday", "overnight", f"on_{res['signal']}"]
    a, b = res["periods"]["2019-2025"]

    out = [f"# Step 8: final test ({res['source']}, {a} to {b})", "",
           f"Computed at {res['run_utc']} UTC on commit `{res['commit']}` with signal `{res['signal']}`, and rendered "
           "from the saved results.", "", NOTE, ""]
    for p, (perf, be, plc, win, _) in res["backtest"].items():
        perf, be = _rename(perf, be)
        out += [f"## Backtest {p}", ""] + body(perf, be, plc, win, res["engine_check"], names)[:-1]

    out += ["## H4: has the overnight effect changed?", "",
            "H1 and the H2 regression on 2019-2020 and 2021+ against train (1993-2012). Significance: * 10%, ** 5%, *** 1%.", ""]
    for exclude, label in [(False, "All days"), (True, "Without flagged days")]:
        h = res["h4"][exclude]
        out += [f"### {label}", "",
                "Overnight vs intraday excess return, bp a day:", "", md_table(h["h1"], "{:.2f}"), "",
                "Change in the gap against train:", "", md_table(h["h1_change"], "{:.2f}"), "",
                "H2 regression by period:", "", md_table(h["h2"], "{:.2f}"), "",
                "Change in the H2 slopes against train:", "", md_table(h["h2_change"], "{:.3f}"), ""]
    write_report("step8_test.md", out)

if __name__ == "__main__":
    if sys.argv[1:] == ["run"]:
        if RESULTS.exists():
            raise SystemExit(f"{RESULTS.name} already exists, the test is run once")
        compute()
    render()
