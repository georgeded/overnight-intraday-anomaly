import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from config import SAMPLES, TICKERS, md_table, write_report
from data_layer import load_processed
from signals import SIGNALS

TRAIN_END = pd.Timestamp(SAMPLES["train"][1])
PERIODS = {"1993-2012": ("1993-01-01", "2012-12-31"), "1993-2007": ("1993-01-01", "2007-12-31"), "2008-2012": ("2008-01-01", "2012-12-31")}
BP = 1e4

# Data problems only, never the size of the returns. Ex-dividend days stay since the dividend is overnight return
EXCLUDE_FLAGS = ["flag_stale_open", "flag_ohlc", "flag_bad_price", "flag_half_day", "flag_split"]

def nw_lags(n):
    return int(np.floor(4 * (n / 100) ** (2 / 9)))

def ols_nw(y, X):
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(y))})

def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""

def load_train(source):
    ret = load_processed(f"returns_{source}")
    ret = ret[ret["date"] <= TRAIN_END].copy()
    sig = load_processed(f"signals_{source}")
    sig = sig[sig["trade_date"] <= TRAIN_END].copy()
    assert set(ret["sample"]) == {"train"} and set(sig["sample"]) == {"train"}
    ret["flagged"] = ret[EXCLUDE_FLAGS].any(axis=1)
    return ret, sig

def daily(ret, exclude):
    d = ret[~ret["flag_first_row"]]
    return d[~d["flagged"]] if exclude else d

def nights(ret, sig, exclude):
    nxt = ret[["date", "ticker", "overnight_xs", "overnight", "flagged"]].rename(columns={"date": "trade_date", "flagged": "flagged_next"})
    n = sig.merge(nxt, on=["trade_date", "ticker"]).merge(ret[["date", "ticker", "flagged"]], on=["date", "ticker"], how="left")
    for c in SIGNALS:
        n[c] = n[c].astype("Float64").astype(float)
    if exclude:
        n = n[~(n["flagged"].astype("boolean").fillna(False).astype(bool) | n["flagged_next"])]
    return n

# H1: mean overnight vs intraday excess return, Newey-West t-stats
def h1(ret, exclude, periods=PERIODS):
    d = daily(ret, exclude)
    rows = []
    for t in TICKERS:
        for name, (a, b) in periods.items():
            g = d[(d["ticker"] == t) & (d["date"] >= a) & (d["date"] <= b)]
            if len(g) < 100:
                continue
            on, intra = g["overnight_xs"], g["intraday_xs"]
            res = {k: ols_nw(s.to_numpy(), np.empty((len(s), 0))) for k, s in [("on", on), ("in", intra), ("diff", on - intra)]}
            rows.append({"ticker": t, "period": name, "days": len(g),
                         "overnight_bp": on.mean() * BP, "t_overnight": res["on"].tvalues[0],
                         "intraday_bp": intra.mean() * BP, "t_intraday": res["in"].tvalues[0],
                         "diff_bp": (on - intra).mean() * BP, "t_diff": res["diff"].tvalues[0], "p_diff": res["diff"].pvalues[0]})
    out = pd.DataFrame(rows)
    out["sig"] = out["p_diff"].map(stars)
    return out

# Change in the gap from 1993-2007 to 2008-2012
def h1_change(ret, exclude):
    d = daily(ret, exclude)
    rows = []
    for t in TICKERS:
        g = d[d["ticker"] == t]
        r = ols_nw((g["overnight_xs"] - g["intraday_xs"]).to_numpy(), (g["date"] >= "2008-01-01").astype(float).to_numpy().reshape(-1, 1))
        rows.append({"ticker": t, "change_bp": r.params[1] * BP, "t": r.tvalues[1], "p": r.pvalues[1], "sig": stars(r.pvalues[1])})
    return pd.DataFrame(rows)

# H2: next night's overnight return on the previous intraday return, sig_b and VIX
def h2_regression(n, dummy="sig_b"):
    rows = []
    for t in TICKERS:
        g = n[n["ticker"] == t].dropna(subset=["overnight_xs", "intraday", dummy, "vix"])
        r = ols_nw(g["overnight_xs"].to_numpy(), g[["intraday", dummy, "vix"]].to_numpy())
        rows.append({"ticker": t, "nights": int(r.nobs), "const_bp": r.params[0] * BP, "t_const": r.tvalues[0],
                     "intraday (bp per 1%)": r.params[1] * BP / 100, "t_intraday": r.tvalues[1],
                     f"{dummy}_bp": r.params[2] * BP, f"t_{dummy}": r.tvalues[2],
                     "vix (bp per point)": r.params[3] * BP, "t_vix": r.tvalues[3], "R2_%": r.rsquared * 100})
    return pd.DataFrame(rows)

# Overnight return with each signal on vs off, Holm-adjusted over all 15 tests
def h2_on_off(n):
    rows = []
    for s in SIGNALS:
        for t in TICKERS:
            g = n[n["ticker"] == t].dropna(subset=[s, "overnight_xs"])
            on = g[s] == 1
            r = ols_nw(g["overnight_xs"].to_numpy(), g[[s]].to_numpy())
            rows.append({"signal": s, "ticker": t, "nights_on": int(on.sum()),
                         "mean_on_bp": g.loc[on, "overnight_xs"].mean() * BP, "mean_off_bp": g.loc[~on, "overnight_xs"].mean() * BP,
                         "diff_bp": r.params[1] * BP, "t": r.tvalues[1], "p": r.pvalues[1]})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    out["sig_holm"] = out["p_holm"].map(stars)
    return out

# H3: SPY and IWM stacked on common nights, Driscoll-Kraay errors since both move together
def stacked_dk(n, regressors):
    g = n[n["ticker"].isin(["SPY", "IWM"])].dropna(subset=["overnight_xs"] + regressors)
    common = g.groupby("date")["ticker"].nunique()
    common = common[common == 2].index
    g = g[g["date"].isin(common)].sort_values(["date", "ticker"])
    iwm = (g["ticker"] == "IWM").astype(float).to_numpy()
    X = np.column_stack([np.ones(len(g))] + [g[r].to_numpy() for r in regressors] + [iwm] + [iwm * g[r].to_numpy() for r in regressors])
    r = sm.OLS(g["overnight_xs"].to_numpy(), X).fit(cov_type="hac-groupsum",
                                                   cov_kwds={"time": pd.factorize(g["date"])[0], "maxlags": nw_lags(len(common))})
    return r, len(common)

def h3(n):
    regs = ["intraday", "sig_b", "vix"]
    r, nc = stacked_dk(n, regs)
    rows = []
    for i, (name, scale) in enumerate(zip(["const"] + regs, [BP, BP / 100, BP, BP]), start=4):
        rows.append({"test": "H2 regression", "coefficient": name, "SPY": r.params[i - 4] * scale,
                     "IWM - SPY": r.params[i] * scale, "t": r.tvalues[i], "p": r.pvalues[i], "nights": nc})

    # the three slopes jointly, then with the intercept, which mostly picks up IWM's higher mean
    for label, idx in [("3 slopes jointly (Wald)", range(5, 8)), ("all 4 jointly (Wald)", range(4, 8))]:
        R = np.zeros((len(idx), 8))
        R[np.arange(len(idx)), list(idx)] = 1
        rows.append({"test": "H2 regression", "coefficient": label, "p": float(r.wald_test(R, scalar=True).pvalue), "nights": nc})

    for s in SIGNALS:
        r, nc = stacked_dk(n, [s])
        rows.append({"test": "on/off gap", "coefficient": s, "SPY": r.params[1] * BP, "IWM - SPY": r.params[3] * BP,
                     "t": r.tvalues[3], "p": r.pvalues[3], "nights": nc})
    out = pd.DataFrame(rows)
    out["sig"] = out["p"].map(stars)
    return out

def run(ret, sig, exclude):
    n = nights(ret, sig, exclude)
    return {"h1": h1(ret, exclude), "h1_change": h1_change(ret, exclude), "h2": h2_regression(n),
            "h2_onoff": h2_on_off(n), "h3": h3(n), "days": len(daily(ret, exclude)), "nights": len(n)}

def verdict(p_all, p_clean):
    if p_all < 0.05 and p_clean < 0.05:
        return "supported"
    return "mixed" if min(p_all, p_clean) < 0.05 else "not supported"

def summary(full, clean):
    rows = []
    def add(hypothesis, ticker, estimate, p_all, p_clean):
        rows.append({"hypothesis": hypothesis, "ticker": ticker, "estimate": estimate,
                     "p_all_days": p_all, "p_clean": p_clean, "verdict": verdict(p_all, p_clean)})
    for t in TICKERS:
        a, b = (r["h1"][(r["h1"]["ticker"] == t) & (r["h1"]["period"] == "1993-2012")].iloc[0] for r in (full, clean))
        add("H1 overnight beats intraday", t, f"{a['diff_bp']:.2f} bp a day", a["p_diff"], b["p_diff"])
    for t in TICKERS:
        a, b = (r["h1_change"].set_index("ticker").loc[t] for r in (full, clean))
        add("H1 gap smaller after 2008", t, f"{a['change_bp']:.2f} bp a day", a["p"], b["p"])
    for t in TICKERS:
        a, b = (r["h2_onoff"][r["h2_onoff"]["ticker"] == t] for r in (full, clean))
        add("H2 signal nights earn more", t, f"{int((a['p_holm'] < 0.05).sum())} of 5 signals after Holm",
            a["p_holm"].min(), b["p_holm"].min())
    a, b = (r["h3"].set_index("coefficient").loc["3 slopes jointly (Wald)"] for r in (full, clean))
    add("H3 IWM reacts differently from SPY", "SPY vs IWM", "3 slopes jointly", a["p"], b["p"])
    return pd.DataFrame(rows)

def section(res, label):
    return [f"## {label}", "",
            f"{res['days']} ETF-days for H1, {res['nights']} signal nights for H2 and H3.", "",
            "**H1** mean daily excess returns in bp, Newey-West t-stats:", "",
            md_table(res["h1"], "{:.2f}"), "",
            "Change in the gap, 2008-2012 minus 1993-2007:", "",
            md_table(res["h1_change"], "{:.2f}"), "",
            "**H2** `overnight_xs = a + b1 intraday(t-1) + b2 sig_b + b3 VIX(t-1)`:", "",
            md_table(res["h2"], "{:.2f}"), "",
            "Signal on vs off, bp per night:", "",
            md_table(res["h2_onoff"], "{:.3f}"), "",
            "**H3** SPY vs IWM, `IWM - SPY` is the difference:", "",
            md_table(res["h3"].astype(object).where(res["h3"].notna(), ""), "{:.3f}"), ""]

def report(source="crsp"):
    ret, sig = load_train(source)
    full, clean = run(ret, sig, False), run(ret, sig, True)

    out = [f"# Step 5: hypothesis tests ({source}, train sample only)", "",
           f"Generated by `python src/hypothesis_tests.py {source}` on rows up to {TRAIN_END.date()}. Significance: "
           f"* 10%, ** 5%, *** 1%. The clean runs drop days with {', '.join(f'`{f}`' for f in EXCLUDE_FLAGS)}.", "",
           "## Summary", "",
           "Supported means p < 0.05 both on all days and without flagged days.", "",
           md_table(summary(full, clean), "{:.3f}"), ""]
    write_report("step5_tests.md", out + section(full, "All train days") + section(clean, "Without flagged days"))
    return full, clean

if __name__ == "__main__":
    report(*sys.argv[1:2])
