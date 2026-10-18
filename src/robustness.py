# Step 9: robustness checks and cumulative-return charts

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest import TICKERS, legs, load_sample, metrics, placebo, run, years_of
from checks import md_table
from config import CHOSEN_SIGNAL, END, ROOT, SAMPLES
from hypothesis_tests import BP, EXCLUDE_FLAGS, nw_lags, stars
from signals import SIGNALS

SIG = CHOSEN_SIGNAL
STRATS = ["buy_hold", "overnight", "intraday", f"on_{SIG}"]
LABELS = {"buy_hold": "Buy and hold", "overnight": "Always overnight", "intraday": "Always intraday",
          f"on_{SIG}": f"Signal ({SIG})"}
ERAS = {"1993-2007": ("1993-01-01", "2007-12-31"), "2008-2020": ("2008-01-01", "2020-12-31"),
        "2021+": ("2021-01-01", END)}
SPLITS = {k: tuple(v) for k, v in SAMPLES.items()}
SPLITS["test"] = (SAMPLES["test"][0], END)
COSTS = [0, 1]
VOL_WINDOW, VOL_TARGET_WINDOW = 21, 252
N_BOOT, BLOCK = 2000, 21
FIG_DIR = ROOT / "reports" / "figures"

# colour-blind safe palette, lines are also labelled directly
COLORS = {"buy_hold": "#2a78d6", "overnight": "#eb6834", "intraday": "#1baf7a", f"on_{SIG}": "#eda100"}


# Helpers

def positions(g, lag=0):
    n = len(g)
    one, zero = np.ones(n), np.zeros(n)
    s = g[SIG].astype(float)

    if lag:
        s = s.shift(lag).fillna(0.0)  # lag=-1 gives the same-close version of the signal

    return {"buy_hold": (one, one), "overnight": (one, zero), "intraday": (zero, one),
            f"on_{SIG}": (s.to_numpy(), zero)}


def window(g, a, b):
    return g[(g["date"] >= a) & (g["date"] <= b)].reset_index(drop=True)


def evaluate(g, p_on, p_in, cost_bp):
    d, trades = run(g, p_on, p_in, cost_bp / 1e4)
    return metrics(d, g["rf"].to_numpy(), trades, np.mean(p_on), years_of(g)), d


def short(m):
    return {"ann_%": m["ann_ret_%"], "sharpe": m["sharpe"], "max_dd_%": m["max_dd_%"]}


def all_legs(source):
    ret, sig = load_sample(source, "all")
    return {t: legs(ret, sig, t, "all")[0] for t in TICKERS}, ret


# 1. Eras

def eras(G, rng):
    perf, gap, plc = [], [], []
    for t, g0 in G.items():
        for era, (a, b) in ERAS.items():
            g = window(g0, a, b)

            for name, (p_on, p_in) in positions(g).items():
                for c in COSTS:
                    m, _ = evaluate(g, p_on, p_in, c)
                    perf.append({"ticker": t, "era": era, "strategy": name, "cost_bp": c, **short(m)})

            # overnight minus intraday gap
            d = (g["overnight"] - g["rf_overnight"] - g["intraday"]).to_numpy()
            r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(d))})
            gap.append({"ticker": t, "era": era, "days": len(g), "first_day": g["date"].min().date(),
                        "overnight_minus_intraday_bp": r.params[0] * BP, "t": r.tvalues[0], "sig": stars(r.pvalues[0])})

            p_ann, _ = placebo(g, SIG, 0.0, rng)
            real = (np.prod(1 + run(g, *positions(g)[f"on_{SIG}"], 0.0)[0]) ** (1 / years_of(g)) - 1)
            plc.append({"ticker": t, "era": era, "placebo_percentile": (p_ann < real).mean() * 100})

    return pd.DataFrame(perf), pd.DataFrame(gap), pd.DataFrame(plc)


def pivot_perf(perf, by, cost, value):
    p = perf[perf["cost_bp"] == cost].pivot_table(index=["ticker", by], columns="strategy", values=value, sort=False)
    return p[STRATS].rename(columns=LABELS)


# 2. Drop the best 1% of days

# Swap each strategy's best 1% of invested days for cash and recompute at 0 bp
def drop_best(G):
    rows = []
    for t, g0 in G.items():
        for split, (a, b) in [("all", (SPLITS["train"][0], END)), *SPLITS.items()]:
            g = window(g0, a, b)
            rf = g["rf"].to_numpy()
            for name, (p_on, p_in) in positions(g).items():
                d, _ = run(g, p_on, p_in, 0.0)
                invested = (p_on + p_in) > 0
                k = max(1, int(round(0.01 * invested.sum())))
                top = np.argsort(np.where(invested, d, -np.inf))[-k:]

                d2 = d.copy()
                cash_day = np.where(p_on[top] > 0, g["rf_overnight"].to_numpy()[top], 0.0) if name != "buy_hold" \
                    else g["rf_overnight"].to_numpy()[top]
                d2[top] = cash_day

                yrs = years_of(g)
                ann = lambda x: (np.prod(1 + x) ** (1 / yrs) - 1) * 100
                sh = lambda x: (x - rf).mean() / (x - rf).std(ddof=1) * np.sqrt(252)
                rows.append({"ticker": t, "sample": split, "strategy": LABELS[name], "days_dropped": k,
                             "ann_%": ann(d), "ann_without_best_1%": ann(d2),
                             "sharpe": sh(d), "sharpe_without_best_1%": sh(d2)})
    return pd.DataFrame(rows)


# 3. Weekend vs weekday nights

def weekend(G):
    rows = []
    for t, g0 in G.items():
        for split, (a, b) in [("all", (SPLITS["train"][0], END)), *SPLITS.items()]:
            g = window(g0, a, b)
            g = g[g["rf_days"].notna()]

            y = (g["overnight"] - g["rf_overnight"]).to_numpy()
            long_gap = (g["rf_days"] > 1).to_numpy(float)
            r = sm.OLS(y, sm.add_constant(long_gap)).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(y))})

            wk, we = y[long_gap == 0], y[long_gap == 1]
            share = g.loc[g[SIG].astype(bool), "rf_days"].gt(1).mean() * 100

            rows.append({"ticker": t, "sample": split, "weekday_nights": len(wk), "weekend_holiday_nights": len(we),
                         "weekday_bp": wk.mean() * BP, "weekend_bp": we.mean() * BP,
                         "weekend_per_calendar_day_bp": we.mean() / g.loc[g["rf_days"] > 1, "rf_days"].mean() * BP,
                         "diff_bp": r.params[1] * BP, "t": r.tvalues[1], "sig": stars(r.pvalues[1]),
                         f"{SIG}_share_weekend_%": share,
                         "all_nights_share_weekend_%": long_gap.mean() * 100})
    return pd.DataFrame(rows)


# 4. Walk-forward re-pick

# Each year from 2019, re-pick the signal with the step 7 rule on the last 5 years
def walk_forward(G):
    picks, start_year = [], int(SPLITS["test"][0][:4])
    end_year = pd.Timestamp(END).year

    for y in range(start_year, end_year + 1):
        a, b = f"{y - 5}-01-01", f"{y - 1}-12-31"
        rows = []
        for s in SIGNALS:
            sh, tr = [], []
            for t, g0 in G.items():
                g = window(g0, a, b)
                m, _ = evaluate(g, g[s].to_numpy().astype(float), np.zeros(len(g)), 1)
                sh.append(m["sharpe"])
                tr.append(m["trades_per_yr"])
            rows.append({"signal": s, "mean_sharpe": np.mean(sh), "trades": np.mean(tr)})

        r = pd.DataFrame(rows).assign(r2=lambda x: x["mean_sharpe"].round(2))
        best = r.sort_values(["r2", "trades"], ascending=[False, True]).iloc[0]
        picks.append({"year": y, "window": f"{a[:4]}-{b[:4]}", "picked": best["signal"],
                      "picked_mean_sharpe": best["mean_sharpe"],
                      f"{SIG}_mean_sharpe": r.set_index("signal").loc[SIG, "mean_sharpe"]})
    picks = pd.DataFrame(picks)

    # trade each year's pick through the test period
    perf = []
    for t, g0 in G.items():
        g = window(g0, *SPLITS["test"])
        chosen = g["date"].dt.year.map(picks.set_index("year")["picked"])
        p_wf = np.array([g.at[i, s] for i, s in zip(g.index, chosen)], dtype=float)
        zero = np.zeros(len(g))

        for name, pos in [("walk-forward re-pick", p_wf), (f"frozen {SIG}", g[SIG].to_numpy(float)),
                          ("always overnight", np.ones(len(g)))]:
            for c in COSTS:
                m, _ = evaluate(g, pos, zero, c)
                perf.append({"ticker": t, "strategy": name, "cost_bp": c, **short(m),
                             "trades_per_yr": m["trades_per_yr"]})
    return picks, pd.DataFrame(perf)


# 5. Bootstrap CIs

# Circular block bootstrap indices, blocks of BLOCK days
def block_indices(n, rng):
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=(N_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n
    return idx.reshape(N_BOOT, -1)[:, :n]


def sharpe_rows(x):
    return x.mean(axis=1) / x.std(axis=1, ddof=1) * np.sqrt(252)


def bootstrap(G, rng, cost_bp=1):
    rows = []
    for t, g0 in G.items():
        for split, (a, b) in SPLITS.items():
            g = window(g0, a, b)
            rf = g["rf"].to_numpy()
            xs = {name: run(g, p_on, p_in, cost_bp / 1e4)[0] - rf for name, (p_on, p_in) in positions(g).items()}

            idx = block_indices(len(g), rng)
            boot = {k: sharpe_rows(v[idx]) for k, v in xs.items()}
            point = {k: v.mean() / v.std(ddof=1) * np.sqrt(252) for k, v in xs.items()}

            for k in STRATS:
                lo, hi = np.percentile(boot[k], [2.5, 97.5])
                rows.append({"ticker": t, "sample": split, "item": LABELS[k], "sharpe": point[k],
                             "ci_low": lo, "ci_high": hi})

            # paired differences use the same draws
            for k1, k2 in [("overnight", "buy_hold"), (f"on_{SIG}", "overnight"), (f"on_{SIG}", "buy_hold")]:
                diff = boot[k1] - boot[k2]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                rows.append({"ticker": t, "sample": split, "item": f"{LABELS[k1]} minus {LABELS[k2]}",
                             "sharpe": point[k1] - point[k2], "ci_low": lo, "ci_high": hi})

    out = pd.DataFrame(rows)
    out["excludes_0"] = np.where((out["ci_low"] > 0) | (out["ci_high"] < 0), "yes", "")
    return out


# 6. Flagged days

def flagged(G):
    rows = []
    for t, g0 in G.items():
        for split, (a, b) in SPLITS.items():
            g = window(g0, a, b)
            pos = positions(g)
            keep = ~g[EXCLUDE_FLAGS].any(axis=1).to_numpy()
            gk = g[keep].reset_index(drop=True)

            for name, (p_on, p_in) in pos.items():
                for c in COSTS:
                    m_all, _ = evaluate(g, p_on, p_in, c)
                    m_ex, _ = evaluate(gk, p_on[keep], p_in[keep], c)
                    rows.append({"ticker": t, "sample": split, "strategy": LABELS[name], "cost_bp": c,
                                 "days_dropped": int((~keep).sum()),
                                 "ann_%": m_all["ann_ret_%"], "ann_excl_%": m_ex["ann_ret_%"],
                                 "sharpe": m_all["sharpe"], "sharpe_excl": m_ex["sharpe"]})
    return pd.DataFrame(rows)


# 7. Lagged signal

def lagged(G, rng):
    rows = []
    for t, g0 in G.items():
        for split, (a, b) in SPLITS.items():
            g = window(g0, a, b)
            for lag in [0, -1]:
                p_on, p_in = positions(g, lag)[f"on_{SIG}"]
                row = {"ticker": t, "sample": split,
                       "signal": f"{SIG} from t-1 (used)" if lag == 0 else f"{SIG} from t (same close)"}
                for c in COSTS:
                    m, _ = evaluate(g, p_on, p_in, c)
                    row[f"ann_% {c}bp"] = m["ann_ret_%"]
                    row[f"sharpe {c}bp"] = m["sharpe"]

                gg = g.assign(_lag=p_on.astype(bool))
                p_ann, _ = placebo(gg, "_lag", 0.0, rng)
                real = np.prod(1 + run(g, p_on, p_in, 0.0)[0]) ** (1 / years_of(g)) - 1
                row["placebo_percentile"] = (p_ann < real).mean() * 100
                rows.append(row)
    return pd.DataFrame(rows)


# 8. Volatility-scaled sizing

# run() with fractional positions, the rest sits in cash and costs apply to weight changes
def run_weighted(g, w_on, w_in, cost):
    n = len(g)
    seq = np.empty(2 * n)
    seq[0::2], seq[1::2] = w_on, w_in
    ch = np.abs(np.diff(np.concatenate([[0.0], seq, [0.0]])))

    on = w_on * g["overnight"].to_numpy() + (1 - w_on) * g["rf_overnight"].to_numpy()
    intra = w_in * g["intraday"].to_numpy() + (1 - w_in) * g["rf_intraday"].to_numpy()
    gross = (1 - cost * ch[0:2 * n:2]) * (1 + on) * (1 - cost * ch[1:2 * n:2]) * (1 + intra)
    gross[-1] *= 1 - cost * ch[-1]

    return gross - 1, ch.sum()


# Weight is target / trailing vol, capped at 1 so there is no leverage
# Vol is the 21 day std up to yesterday, target its 252 day median
def vol_weights(ret, t):
    r = ret[ret["ticker"] == t].sort_values("date").set_index("date")

    out = {}
    for leg, col in [("on", "overnight"), ("in", "intraday"), ("cc", "ret")]:
        vol = r[col].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().shift(1)
        target = vol.rolling(VOL_TARGET_WINDOW, min_periods=VOL_TARGET_WINDOW).median()
        out[leg] = (target / vol).clip(upper=1.0)
    return pd.DataFrame(out)


def vol_scaled(G, ret):
    rows = []
    for t, g0 in G.items():
        w = vol_weights(ret, t)
        for split, (a, b) in SPLITS.items():
            g = window(g0, a, b)
            ww = w.reindex(g["date"]).fillna(1.0).to_numpy()
            pos = positions(g)

            scaled = {"buy_hold": (ww[:, 2], ww[:, 2]), "overnight": (ww[:, 0], np.zeros(len(g))),
                      "intraday": (np.zeros(len(g)), ww[:, 1]),
                      f"on_{SIG}": (pos[f"on_{SIG}"][0] * ww[:, 0], np.zeros(len(g)))}

            for name in STRATS:
                for c in COSTS:
                    m0, _ = evaluate(g, *pos[name], c)
                    d, ch = run_weighted(g, *scaled[name], c / 1e4)
                    m1 = metrics(d, g["rf"].to_numpy(), ch, np.mean(scaled[name][0]), years_of(g))
                    rows.append({"ticker": t, "sample": split, "strategy": LABELS[name], "cost_bp": c,
                                 "ann_%": m0["ann_ret_%"], "ann_scaled_%": m1["ann_ret_%"],
                                 "sharpe": m0["sharpe"], "sharpe_scaled": m1["sharpe"],
                                 "max_dd_%": m0["max_dd_%"], "max_dd_scaled_%": m1["max_dd_%"],
                                 "avg_exposure": np.mean([x for x in scaled[name] if x.any()][0])})
    return pd.DataFrame(rows)


# With 0/1 weights run_weighted must match run
def check_weighted(G):
    g = G["SPY"]
    pos = positions(g)[f"on_{SIG}"]

    a, _ = run(g, *pos, 2e-4)
    b, _ = run_weighted(g, *pos, 2e-4)
    return float(np.max(np.abs(a - b)))


# Chart

def chart(G, cost_bp, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    for ax, t in zip(axes, TICKERS):
        g = G[t]

        for name, (p_on, p_in) in positions(g).items():
            d, _ = run(g, p_on, p_in, cost_bp / 1e4)
            eq = np.cumprod(1 + d)
            ax.plot(g["date"], eq, color=COLORS[name], lw=2, label=LABELS[name])
            ax.annotate(f"{LABELS[name]}  {eq[-1]:.1f}x", (g["date"].iloc[-1], eq[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color="#0b0b0b")

        ax.set_yscale("log")

        # mark where validation and test start
        for x, lab in [(SPLITS["validation"][0], "validation"), (SPLITS["test"][0], "test")]:
            ax.axvline(pd.Timestamp(x), color="#8a8984", lw=1, ls=":")
            ax.text(pd.Timestamp(x), ax.get_ylim()[1], f" {lab}", va="top", fontsize=8, color="#52514e")

        ax.axhline(1, color="#c3c2b7", lw=0.8)
        ax.set_title(f"{t}: growth of $1, {g['date'].min().year}-{g['date'].max().year}, "
                     f"{cost_bp} bp per trade", loc="left", fontsize=11, color="#0b0b0b")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)

        # room on the right for the labels, ticks every 4 years
        ax.set_xlim(g["date"].min(), g["date"].max() + pd.Timedelta(days=1400))
        first, last = g["date"].min().year, g["date"].max().year
        ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(first + (4 - first % 4) % 4, last + 1, 4)])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))

    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# Report

def report(source="crsp"):
    rng = np.random.default_rng(9)
    G, ret = all_legs(source)

    era_perf, era_gap, era_plc = eras(G, rng)
    drop = drop_best(G)
    wk = weekend(G)
    picks, wf = walk_forward(G)
    boot = bootstrap(G, rng)
    flg = flagged(G)
    lag = lagged(G, rng)
    vol = vol_scaled(G, ret)
    wcheck = check_weighted(G)

    chart(G, 0, FIG_DIR / "step9_cumulative_0bp.png")
    chart(G, 1, FIG_DIR / "step9_cumulative_1bp.png")

    win = pd.DataFrame([{"ticker": t, "first_day": g["date"].min().date(), "last_day": g["date"].max().date(),
                         "days": len(g)} for t, g in G.items()])

    out = [f"# Step 9: robustness ({source})", "",
           f"Generated by `python src/robustness.py {source}`. Descriptive only: no frozen choice was changed "
           f"(signal `{SIG}`, step 6 engine and costs), the step 8 test stands as run, and nothing here feeds a "
           "decision. These checks use the whole data set, including the test period.", "",
           f"Strategies: buy-and-hold, always overnight, always intraday and the frozen signal (`{SIG}`), as in step 6. "
           "Windows start at the first night all five signals exist:", "", md_table(win), "",
           "Sharpe is annualised over the T-bill; annual return compounded; costs are per trade (one-way).", "",
           NOTES, "",
           "## Cumulative returns, full sample", "",
           "![Growth of $1 before costs](figures/step9_cumulative_0bp.png)", "",
           "![Growth of $1 at 1 bp per trade](figures/step9_cumulative_1bp.png)", "",
           "Log scale; dotted lines mark the start of validation (2013) and test (2019).", "",
           "## 1. By era", "",
           "Overnight minus intraday excess return per day (Newey-West):", "", md_table(era_gap, "{:.2f}"), ""]
    for c in COSTS:
        out += [f"Annual return % at {c} bp:", "", md_table(pivot_perf(era_perf, "era", c, "ann_%"), "{:.2f}"), "",
                f"Sharpe at {c} bp:", "", md_table(pivot_perf(era_perf, "era", c, "sharpe"), "{:.2f}"), ""]
    out += ["Max drawdown % at 0 bp:", "", md_table(pivot_perf(era_perf, "era", 0, "max_dd_%"), "{:.1f}"), "",
            f"`{SIG}` placebo percentile (annual return, 0 bp, 1,000 random sets of nights):", "",
            md_table(era_plc.pivot(index="era", columns="ticker", values="placebo_percentile")[TICKERS]
                     .loc[list(ERAS)], "{:.1f}"), "",
            "## 2. Dropping the best 1% of days", "",
            "Each strategy's best 1% of invested days are replaced by cash and the metrics recomputed (0 bp).", "",
            md_table(drop, "{:.2f}"), "",
            "## 3. Weekend vs weekday nights", "",
            "Overnight excess return per night: weekday nights (1 calendar day) vs weekend and holiday nights "
            "(2+ days). `diff` = weekend minus weekday (Newey-West).", "",
            md_table(wk, "{:.2f}"), "",
            "## 4. Walk-forward re-pick from 2019", "",
            "Each year the step 7 rule (highest mean Sharpe at 1 bp over the three ETFs, ties at 2 dp to fewer trades) "
            "is re-applied to the previous five years and the winner is traded for that year. Shown only to see how "
            f"stable the pick is; the frozen choice remains `{SIG}`.", "",
            md_table(picks, "{:.3f}"), "",
            f"{SPLITS['test'][0][:4]}-{END[:4]} performance:", "", md_table(wf, "{:.2f}"), "",
            "## 5. Bootstrap 95% confidence intervals for Sharpe (1 bp per trade)", "",
            f"Circular block bootstrap, blocks of {BLOCK} days, {N_BOOT} draws; differences use the same draws for "
            "both strategies (paired).", "", md_table(boot, "{:.2f}"), "",
            "## 6. With vs without flagged days", "",
            f"Days with any of {', '.join(f'`{f}`' for f in EXCLUDE_FLAGS)} removed from every strategy.", "",
            md_table(flg, "{:.2f}"), "",
            "## 7. Signal timing", "",
            f"The signals use data up to the previous close, so the order can be placed before the close. For "
            f"comparison, `{SIG} from t (same close)` uses the same day's close and VIX, which a market-on-close "
            "order could not know in time.",
            "", md_table(lag, "{:.2f}"), "",
            "## 8. Volatility-scaled sizing", "",
            f"Weight = min(1, target / trailing vol): vol is the std of the leg's own returns over the previous "
            f"{VOL_WINDOW} days (close-to-close for buy-and-hold), target is the median of that vol over the previous "
            f"{VOL_TARGET_WINDOW} days; the rest of the position is in cash; no leverage. Costs are charged on every "
            f"change in weight. Engine check: with 0/1 weights it reproduces the step 6 engine (max difference "
            f"{wcheck:.1e}).", "", md_table(vol, "{:.2f}"), ""]

    p = ROOT / "reports" / "step9_robustness.md"
    p.write_text("\n".join(out))
    print(f"wrote {p.relative_to(ROOT)}")


NOTES = """## Summary

- **Eras.** The overnight-minus-intraday gap is significant at 5% only in 1993-2007 for SPY (4.3 bp a day) and QQQ
  (9.5 bp), with IWM at 5.8 bp (10%). In 2008-2020 it shrinks (SPY 1.2 bp, QQQ 1.8 bp, IWM 4.0 bp at 10%), and in
  2021+ it is about zero for SPY and QQQ (-1.0 and -0.1 bp) while IWM keeps 6.1 bp (t = 1.6). Before costs,
  always-overnight has the best Sharpe in 7 of the 9 era/ETF cells, but at 1 bp per trade buy-and-hold has the best
  Sharpe in every era from 2008 for SPY and QQQ. `sig_b` against random nights: 0th-84th percentile by era, with
  no consistent edge.
- **Best 1% of days.** Every strategy depends heavily on its best days. Without its best 1% of nights,
  always-overnight still earns 1-3% a year over the full sample (Sharpe 0.00-0.16), while buy-and-hold turns
  negative because it loses its best 1% of whole days, which are larger.
- **Weekends.** No evidence that weekend nights earn more. Over the full sample the weekend return per night is
  lower than on weekday nights (not significant), and per calendar day it is much lower. In the test period SPY
  and QQQ weekend nights were significantly *worse* than weekday nights (-11 and -14 bp, t = -2.1 and -2.3).
  `sig_b` holds weekend nights in roughly the same proportion as all nights.
- **Walk-forward re-pick.** The rule picks `sig_b` for 2019-2020 and then `sig_a` (2021, 2022, 2024), `sig_bc`
  (2023) and `sig_c` (2025). Against frozen `sig_b` at 0 bp, the re-picked strategy is worse for SPY (Sharpe 0.20
  vs 0.29) and better for QQQ (0.15 vs 0.13) and IWM (0.45 vs 0.31). The choice of signal is unstable.
- **Bootstrap.** At 1 bp per trade most Sharpe CIs include zero. Buy-and-hold's Sharpe is significantly positive in
  validation and test for SPY and QQQ. `sig_b` is significantly positive only for IWM in validation, the sample it
  was chosen on. Significant differences: always-overnight below buy-and-hold for SPY in the test, `sig_b` below
  buy-and-hold for QQQ in the test, and `sig_b` below always-overnight for QQQ in train. No CI shows `sig_b` or any
  overnight strategy beating buy-and-hold in the test.
- **Flagged days.** Excluding them raises always-overnight returns in 7 of 9 sample/ETF cells. These flags include
  `flag_extreme` and `flag_reversal`, which are based on the returns themselves, so this check is only indicative.
- **Lagged signal.** Using yesterday's `sig_b` is better in train for all three ETFs but worse in validation and
  test for SPY and IWM, and slightly better for QQQ. The validation edge that picked `sig_b` (91st and 98th
  placebo percentiles for SPY and IWM) falls to the 68th and 42nd when the signal is lagged one day, so the pick
  depends on acting at the same close the signal is measured on.
- **Volatility scaling** cuts drawdowns (in the test at 1 bp, SPY always-overnight from -30% to -20%, `sig_b` from
  -14% to -8%) and raises Sharpe for always-overnight on SPY and QQQ, but it lowers `sig_b`'s Sharpe on SPY and
  IWM and does not change any ranking against buy-and-hold.

**Overall:** the overnight premium in these three ETFs is concentrated in the early years and in a minority of
nights, has faded for SPY and QQQ, and is too small to survive costs of about 1-2 bp per trade. IWM keeps a gap
that is not significant. The frozen signal `sig_b` adds nothing reliable out of sample, and its validation edge
depends on same-close timing.
"""


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "crsp")
