# Step 5: hypothesis tests on the train sample only (1993-2012)
#
#     python src/hypothesis_tests.py          # yfinance
#     python src/hypothesis_tests.py crsp
#
# H1  overnight vs intraday excess return, Newey-West t-test on the difference
# H2  does today's intraday return, sig_b or VIX predict the next night
# H3  SPY vs IWM, stacked regression with Driscoll-Kraay errors
# H4  left for the test step
#
# Everything runs twice, with and without the step 2 flagged days

import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from checks import md_table
from config import ROOT, SAMPLES
from data_layer import load_processed
from signals import SIGNALS

TRAIN_END = pd.Timestamp(SAMPLES["train"][1])
SUBPERIODS = {"1993-2007": ("1993-01-01", "2007-12-31"), "2008-2012": ("2008-01-01", "2012-12-31")}
BP = 1e4

# Flags dropped in the clean run
# Ex-dividend days stay since the dividend is part of the overnight return
EXCLUDE_FLAGS = ["flag_stale_open", "flag_extreme", "flag_reversal", "flag_ohlc",
                 "flag_bad_price", "flag_half_day", "flag_split"]


# Newey-West (1994) rule of thumb
def nw_lags(n):
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def ols_nw(y, X):
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(y))})


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


# Data, train only

def load_train(source):
    ret = load_processed(f"returns_{source}")
    ret = ret[ret["date"] <= TRAIN_END].copy()
    sig = load_processed(f"signals_{source}")
    sig = sig[sig["trade_date"] <= TRAIN_END].copy()

    # stop if anything after train gets through
    assert ret["date"].max() <= TRAIN_END and sig["trade_date"].max() <= TRAIN_END
    assert set(ret["sample"]) == {"train"} and set(sig["sample"]) == {"train"}

    ret["flagged"] = ret[EXCLUDE_FLAGS].any(axis=1)
    return ret, sig


def daily(ret, exclude):
    d = ret[~ret["flag_first_row"]]
    return d[~d["flagged"]] if exclude else d


# One row per signal night, regressors from day t and the overnight return of t+1
def nights(ret, sig, exclude):
    keep = ["date", "ticker", "overnight_xs", "overnight", "flagged"]
    nxt = ret[keep].rename(columns={"date": "trade_date", "flagged": "flagged_next"})
    today = ret[["date", "ticker", "flagged"]]
    n = sig.merge(nxt, on=["trade_date", "ticker"], how="inner").merge(today, on=["date", "ticker"], how="left")

    for c in SIGNALS:
        n[c] = n[c].astype("Float64").astype(float)  # NA to NaN, dropped per regression

    if exclude:
        # a signal day before the loaded rows has no flag, count it as clean
        n = n[~(n["flagged"].astype("boolean").fillna(False).astype(bool) | n["flagged_next"])]
    return n


# H1

def h1(ret, exclude, periods=None):
    rows = []
    d = daily(ret, exclude)
    periods = periods or {"1993-2012": ("1993-01-01", "2012-12-31"), **SUBPERIODS}

    for t in ["SPY", "QQQ", "IWM"]:
        for name, (a, b) in periods.items():
            g = d[(d["ticker"] == t) & (d["date"] >= a) & (d["date"] <= b)]
            if len(g) < 100:
                continue

            on, intra = g["overnight_xs"], g["intraday_xs"]
            res = {k: ols_nw(s.to_numpy(), np.empty((len(s), 0))) for k, s in
                   [("on", on), ("in", intra), ("diff", on - intra)]}
            rows.append({
                "ticker": t, "period": name, "days": len(g),
                "overnight_bp": on.mean() * BP, "t_overnight": res["on"].tvalues[0],
                "intraday_bp": intra.mean() * BP, "t_intraday": res["in"].tvalues[0],
                "diff_bp": (on - intra).mean() * BP, "t_diff": res["diff"].tvalues[0],
                "p_diff": res["diff"].pvalues[0],
                "diff_ann_pct": (on - intra).mean() * 252 * 100,
            })

    out = pd.DataFrame(rows)
    out["sig"] = out["p_diff"].map(stars)
    return out


# Is the gap different in 2008-2012 than in 1993-2007?
def h1_subperiod_change(ret, exclude):
    d = daily(ret, exclude)
    rows = []
    for t in ["SPY", "QQQ", "IWM"]:
        g = d[d["ticker"] == t]
        y = (g["overnight_xs"] - g["intraday_xs"]).to_numpy()
        late = (g["date"] >= "2008-01-01").astype(float).to_numpy()
        r = ols_nw(y, late.reshape(-1, 1))
        rows.append({"ticker": t, "change_bp": r.params[1] * BP, "t": r.tvalues[1], "p": r.pvalues[1],
                     "sig": stars(r.pvalues[1])})
    return pd.DataFrame(rows)


# H2

def h2_regression(n, dummy="sig_b"):
    rows, fits = [], {}
    for t in ["SPY", "QQQ", "IWM"]:
        g = n[n["ticker"] == t].dropna(subset=["overnight_xs", "intraday", dummy, "vix"])
        X = g[["intraday", dummy, "vix"]].to_numpy()
        r = ols_nw(g["overnight_xs"].to_numpy(), X)
        fits[t] = r

        names = ["const", "intraday", dummy, "vix"]
        row = {"ticker": t, "nights": int(r.nobs), "r2_pct": r.rsquared * 100}
        for i, k in enumerate(names):
            row[f"{k}"] = r.params[i]
            row[f"t_{k}"] = r.tvalues[i]
        rows.append(row)
    return pd.DataFrame(rows), fits


# Readable units: bp, bp per 1% intraday move, bp per VIX point
def h2_main_table(tab, dummy="sig_b"):
    out = pd.DataFrame({
        "ticker": tab["ticker"], "nights": tab["nights"],
        "const_bp": tab["const"] * BP, "t_const": tab["t_const"],
        "intraday (bp per 1%)": tab["intraday"] * BP / 100, "t_intraday": tab["t_intraday"],
        f"{dummy}_bp": tab[dummy] * BP, f"t_{dummy}": tab[f"t_{dummy}"],
        "vix (bp per point)": tab["vix"] * BP, "t_vix": tab["t_vix"],
        "R2_%": tab["r2_pct"],
    })
    return out


# Coefficient on each candidate dummy, with intraday and VIX as controls
def h2_dummies(n):
    rows = []
    for s in SIGNALS:
        tab, _ = h2_regression(n, s)
        for _, r in tab.iterrows():
            rows.append({"signal": s, "ticker": r["ticker"], "nights": r["nights"],
                         "dummy_bp": r[s] * BP, "t": r[f"t_{s}"]})
    return pd.DataFrame(rows)


def h2_on_off(n):
    rows = []
    for s in SIGNALS:
        for t in ["SPY", "QQQ", "IWM"]:
            g = n[n["ticker"] == t].dropna(subset=[s, "overnight_xs"])
            on = g[s] == 1
            r = ols_nw(g["overnight_xs"].to_numpy(), g[[s]].to_numpy())

            rows.append({"signal": s, "ticker": t, "nights_on": int(on.sum()), "nights_off": int((~on).sum()),
                         "mean_on_bp": g.loc[on, "overnight_xs"].mean() * BP,
                         "mean_off_bp": g.loc[~on, "overnight_xs"].mean() * BP,
                         "diff_bp": r.params[1] * BP, "t": r.tvalues[1], "p": r.pvalues[1]})

    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p"], method="holm")[1]  # over all 15 tests
    out["sig"] = out["p"].map(stars)
    out["sig_holm"] = out["p_holm"].map(stars)
    return out


# H3

# Stack SPY and IWM on common nights, the IWM terms give the differences
# Driscoll-Kraay errors allow both ETFs to move together on the same night
def stacked_dk(n, regressors):
    g = n[n["ticker"].isin(["SPY", "IWM"])].dropna(subset=["overnight_xs"] + regressors)
    common = g.groupby("date")["ticker"].nunique()
    g = g[g["date"].isin(common[common == 2].index)].sort_values(["date", "ticker"])

    iwm = (g["ticker"] == "IWM").astype(float).to_numpy()
    X = np.column_stack([np.ones(len(g))] + [g[r].to_numpy() for r in regressors]
                        + [iwm] + [iwm * g[r].to_numpy() for r in regressors])
    names = ["const"] + regressors + ["IWM"] + [f"IWM x {r}" for r in regressors]

    time = pd.factorize(g["date"])[0]
    lags = nw_lags(len(common[common == 2]))
    r = sm.OLS(g["overnight_xs"].to_numpy(), X).fit(cov_type="hac-groupsum",
                                                   cov_kwds={"time": time, "maxlags": lags})
    return r, names, len(common[common == 2])


def h3(n):
    rows = []

    # the H2 regression
    regs = ["intraday", "sig_b", "vix"]
    r, names, nights_common = stacked_dk(n, regs)
    k = len(regs) + 1
    scale = {"IWM": BP, "IWM x intraday": BP / 100, "IWM x sig_b": BP, "IWM x vix": BP}
    for i in range(k, 2 * k):
        rows.append({"test": "H2 regression", "coefficient": names[i].replace("IWM x ", "").replace("IWM", "const"),
                     "SPY": r.params[i - k] * scale[names[i]], "IWM - SPY": r.params[i] * scale[names[i]],
                     "t": r.tvalues[i], "p": r.pvalues[i], "nights": nights_common})

    # joint Wald tests on the three slopes, then slopes plus intercept
    # the four-way test mostly picks up IWM's higher average return
    for label, idx in [("3 slopes jointly (Wald)", range(k + 1, 2 * k)), ("all 4 jointly (Wald)", range(k, 2 * k))]:
        R = np.zeros((len(idx), 2 * k))
        R[np.arange(len(idx)), list(idx)] = 1
        joint = r.wald_test(R, scalar=True)
        rows.append({"test": "H2 regression", "coefficient": label, "SPY": np.nan, "IWM - SPY": np.nan,
                     "t": np.nan, "p": float(joint.pvalue), "nights": nights_common})

    # on/off gap for every candidate
    for s in SIGNALS:
        r, names, nc = stacked_dk(n, [s])
        rows.append({"test": "on/off gap", "coefficient": s, "SPY": r.params[1] * BP,
                     "IWM - SPY": r.params[3] * BP, "t": r.tvalues[3], "p": r.pvalues[3], "nights": nc})

    out = pd.DataFrame(rows)
    out["sig"] = out["p"].map(stars)
    return out


# Report

def run(ret, sig, exclude):
    n = nights(ret, sig, exclude)
    tab, _ = h2_regression(n)
    return {
        "h1": h1(ret, exclude), "h1_change": h1_subperiod_change(ret, exclude),
        "h2": h2_main_table(tab), "h2_dummies": h2_dummies(n), "h2_onoff": h2_on_off(n), "h3": h3(n),
        "days": len(daily(ret, exclude)), "nights": len(n),
    }


def section(res, label):
    onoff_cols = ["signal", "ticker", "nights_on", "nights_off", "mean_on_bp", "mean_off_bp", "diff_bp", "t", "p",
                  "sig", "p_holm", "sig_holm"]
    return [
        f"## {label}", "",
        f"{res['days']} ETF-days for H1, {res['nights']} signal nights for H2/H3.", "",
        "### H1: overnight vs intraday excess return", "",
        "Mean daily excess returns in basis points, Newey-West t-statistics; `diff` = overnight - intraday, "
        "`diff_ann_pct` = 252 × daily diff in %. The 1993-2012 rows start at each ETF's first day "
        "(QQQ 1999, IWM 2000).", "",
        md_table(res["h1"], "{:.2f}"), "",
        "Change in the overnight-intraday gap, 2008-2012 minus 1993-2007 (Newey-West):", "",
        md_table(res["h1_change"], "{:.2f}"), "",
        "### H2: what predicts the next night's overnight excess return", "",
        "`overnight_xs(t+1) = a + b1 intraday(t) + b2 sig_b(t) + b3 VIX(t)`, Newey-West t-statistics.", "",
        md_table(res["h2"], "{:.2f}"), "",
        "Same regression with each candidate dummy in place of `sig_b` (dummy coefficient only, bp):", "",
        md_table(res["h2_dummies"], "{:.2f}"), "",
        "Mean overnight excess return with the signal on vs off (bp per night). `p_holm` adjusts for the "
        "15 tests (5 signals × 3 ETFs):", "",
        md_table(res["h2_onoff"][onoff_cols], "{:.3f}"), "",
        "### H3: SPY vs IWM", "",
        "Stacked regression on nights both ETFs trade, Driscoll-Kraay standard errors. `SPY` is SPY's coefficient, "
        "`IWM - SPY` the difference (bp; intraday per 1% move, VIX per point).", "",
        md_table(res["h3"].astype(object).where(res["h3"].notna(), ""), "{:.3f}"), "",
    ]


def report(source="yfinance"):
    ret, sig = load_train(source)
    full, clean = run(ret, sig, False), run(ret, sig, True)
    lags = nw_lags(len(ret[ret["ticker"] == "SPY"]))

    out = [f"# Step 5: hypothesis tests ({source}, train sample only)", "",
           f"Generated by `python src/hypothesis_tests.py {source}`. Only rows dated up to {TRAIN_END.date()} are "
           "loaded (for H2 the night traded must also end by then); validation and test rows are never touched. "
           "H4 is deferred to the test step.", "",
           "## Set-up", "",
           "- Returns are excess returns: `overnight_xs = overnight - rf` (all T-bill interest on the overnight leg), "
           "`intraday_xs = intraday`.",
           f"- Newey-West lags: floor(4 (n/100)^(2/9)), e.g. {lags} for SPY's full train sample.",
           "- H2 pairs signal(t) and regressors from day t with the overnight return from t's close to the next open.",
           "- Rows with a missing regressor (before the 252-day windows fill) are dropped per regression.",
           f"- Excluding flagged days drops any day with {', '.join(f'`{f}`' for f in EXCLUDE_FLAGS)}; for H2 a night "
           "is dropped if either the signal day or the night's day is flagged. Ex-dividend days are kept, because "
           "the dividend is part of the overnight return.",
           "- Significance: * 10%, ** 5%, *** 1%.", ""]
    out += section(full, "A. All train days")
    out += section(clean, "B. Excluding step 2 flagged days")
    out += [NOTES]

    p = ROOT / "reports" / "step5_tests.md"
    p.write_text("\n".join(out))
    print(f"wrote {p.relative_to(ROOT)}")
    return full, clean


NOTES = """## What the train sample says (yfinance snapshot)

- **H1, overnight beats intraday: supported on the full train sample.** All three ETFs earn a positive overnight
  excess return and a slightly negative intraday one; the gap is significant at 5% for SPY and IWM and 1% for QQQ,
  and at 1% for all three once flagged days are excluded. It comes from 1993-2007. In 2008-2012 the gap is small
  and not significant (except IWM at 10% without flagged days), but the change between the two periods is not
  significant either, so a fall in the gap is suggested, not shown.
- **H2, the signals do not pick better nights.** None of the 15 signal x ETF on/off gaps survives the Holm
  correction. The raw gaps are mostly negative: when the day was bad and/or VIX was high, the next night earned
  *less* on average (QQQ `sig_bc` -16 bp, p = 0.015 before correction). In the regression, today's intraday
  return, `sig_b` and VIX level add almost nothing (R2 below 0.4%). QQQ's positive `sig_a` coefficient appears
  only when the intraday return is also in the regression (the two are nearly collinear) and should not be read
  as a standalone effect.
- **H3, SPY vs IWM: no robust difference in how they respond.** The three slopes are jointly
  insignificant (p = 0.84 on all days, 0.11 without flagged days). The four-coefficient test rejects because IWM's
  average overnight return is higher than SPY's on the same nights, not because it reacts differently to the
  signals. The one individually significant gap (`sig_ac`, p = 0.04-0.05) is one of many tests.
- **Flagged days** do not change the conclusions; excluding them strengthens H1 and leaves H2/H3 null.

## Notes

- No parameters were tuned here: the signal definitions are the ones fixed in step 4, and the 5 candidates were
  fixed before any return was looked at.
- Yahoo data. The step 2 caveats (stale opens in early SPY years, a possibly not-fully-ex-dividend open) apply and
  will be re-checked on CRSP.
"""


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "yfinance")
