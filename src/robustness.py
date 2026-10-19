# Step 9: robustness checks and cumulative-return charts, descriptive only

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest import TICKERS, ann_return, legs, load_sample, metrics, placebo, run, years_of
from checks import md_table
from config import CHOSEN_SIGNAL, END, ROOT, SAMPLES
from hypothesis_tests import BP, EXCLUDE_FLAGS, nw_lags, stars
from signals import SIGNALS

SIG = CHOSEN_SIGNAL
STRATS = ["buy_hold", "overnight", "intraday", f"on_{SIG}"]
LABELS = {"buy_hold": "Buy and hold", "overnight": "Always overnight", "intraday": "Always intraday", f"on_{SIG}": f"Signal ({SIG})"}
COLORS = {"buy_hold": "#2a78d6", "overnight": "#eb6834", "intraday": "#1baf7a", f"on_{SIG}": "#eda100"}
ERAS = {"1993-2007": ("1993-01-01", "2007-12-31"), "2008-2020": ("2008-01-01", "2020-12-31"), "2021+": ("2021-01-01", END)}
ALL = {"all": (SAMPLES["train"][0], END), **SAMPLES}
COSTS = [0, 1]
N_BOOT, BLOCK = 2000, 21
VOL_WINDOW, VOL_TARGET_WINDOW = 21, 252

# lag=-1 gives the same-close version of the signal, which could not be traded in time
def positions(g, lag=0):
    one, zero = np.ones(len(g)), np.zeros(len(g))
    s = g[SIG].astype(float)
    if lag:
        s = s.shift(lag).fillna(0.0)
    return {"buy_hold": (one, one), "overnight": (one, zero), "intraday": (zero, one), f"on_{SIG}": (s.to_numpy(), zero)}

def window(g, a, b):
    return g[(g["date"] >= a) & (g["date"] <= b)].reset_index(drop=True)

def evaluate(g, p_on, p_in, cost_bp):
    d, trades = run(g, p_on, p_in, cost_bp / 1e4)
    return metrics(d, g["rf"].to_numpy(), trades, np.mean(p_on), years_of(g))

def sharpe(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(252)

def eras(G, rng):
    perf, gap = [], []
    for t, g0 in G.items():
        for era, (a, b) in ERAS.items():
            g = window(g0, a, b)
            for name, (p_on, p_in) in positions(g).items():
                for c in COSTS:
                    m = evaluate(g, p_on, p_in, c)
                    perf.append({"ticker": t, "era": era, "strategy": LABELS[name], "cost_bp": c, "ann_%": m["ann_ret_%"], "sharpe": m["sharpe"]})

            d = (g["overnight"] - g["rf_overnight"] - g["intraday"]).to_numpy()
            r = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(d))})
            p_ann, _ = placebo(g, SIG, 0.0, rng)
            real = ann_return(run(g, *positions(g)[f"on_{SIG}"], 0.0)[0], years_of(g))
            gap.append({"ticker": t, "era": era, "overnight_minus_intraday_bp": r.params[0] * BP, "t": r.tvalues[0],
                        "sig": stars(r.pvalues[0]), f"{SIG}_placebo_percentile": (p_ann < real).mean() * 100})
    return pd.DataFrame(perf), pd.DataFrame(gap)

# Each strategy's best 1% of invested days swapped for cash, at 0 bp
def drop_best(G):
    rows = []
    for t, g0 in G.items():
        g = window(g0, *ALL["all"])
        rf = g["rf"].to_numpy()
        for name, (p_on, p_in) in positions(g).items():
            d, _ = run(g, p_on, p_in, 0.0)
            invested = (p_on + p_in) > 0
            top = np.argsort(np.where(invested, d, -np.inf))[-max(1, int(round(0.01 * invested.sum()))):]
            d2 = d.copy()
            d2[top] = g["rf_overnight"].to_numpy()[top] if name == "buy_hold" else np.where(p_on[top] > 0, g["rf_overnight"].to_numpy()[top], 0.0)
            rows.append({"ticker": t, "strategy": LABELS[name], "ann_%": ann_return(d, years_of(g)) * 100,
                         "ann_without_best_1%": ann_return(d2, years_of(g)) * 100,
                         "sharpe": sharpe(d - rf), "sharpe_without_best_1%": sharpe(d2 - rf)})
    return pd.DataFrame(rows)

def weekend(G):
    rows = []
    for t, g0 in G.items():
        for split, (a, b) in ALL.items():
            g = window(g0, a, b)
            g = g[g["rf_days"].notna()]
            y = (g["overnight"] - g["rf_overnight"]).to_numpy()
            long_gap = (g["rf_days"] > 1).to_numpy(float)
            r = sm.OLS(y, sm.add_constant(long_gap)).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(y))})
            rows.append({"ticker": t, "sample": split, "weekday_bp": y[long_gap == 0].mean() * BP, "weekend_bp": y[long_gap == 1].mean() * BP,
                         "diff_bp": r.params[1] * BP, "t": r.tvalues[1], "sig": stars(r.pvalues[1])})
    return pd.DataFrame(rows)

# Each test year, re-pick the signal with the step 7 rule on the five years before
def walk_forward(G):
    picks = []
    for y in range(int(SAMPLES["test"][0][:4]), pd.Timestamp(END).year + 1):
        a, b = f"{y - 5}-01-01", f"{y - 1}-12-31"
        rows = []
        for s in SIGNALS:
            ms = [evaluate(window(g0, a, b), window(g0, a, b)[s].to_numpy().astype(float), np.zeros(len(window(g0, a, b))), 1) for g0 in G.values()]
            rows.append({"signal": s, "mean_sharpe": np.mean([m["sharpe"] for m in ms]), "trades": np.mean([m["trades_per_yr"] for m in ms])})
        r = pd.DataFrame(rows).assign(r2=lambda x: x["mean_sharpe"].round(2))
        best = r.sort_values(["r2", "trades"], ascending=[False, True]).iloc[0]
        picks.append({"year": y, "picked": best["signal"], "picked_mean_sharpe": best["mean_sharpe"]})
    picks = pd.DataFrame(picks)

    perf = []
    for t, g0 in G.items():
        g = window(g0, *SAMPLES["test"])
        chosen = g["date"].dt.year.map(picks.set_index("year")["picked"])
        p_wf = np.array([g.at[i, s] for i, s in zip(g.index, chosen)], dtype=float)
        for name, pos in [("walk-forward re-pick", p_wf), (f"frozen {SIG}", g[SIG].to_numpy(float))]:
            m = evaluate(g, pos, np.zeros(len(g)), 0)
            perf.append({"ticker": t, "strategy": name, "ann_%": m["ann_ret_%"], "sharpe": m["sharpe"]})
    return picks, pd.DataFrame(perf)

# Circular block bootstrap, differences use the same draws for both strategies
def bootstrap(G, rng, cost_bp=1):
    rows = []
    for t, g0 in G.items():
        for split, (a, b) in SAMPLES.items():
            g = window(g0, a, b)
            xs = {k: run(g, p_on, p_in, cost_bp / 1e4)[0] - g["rf"].to_numpy() for k, (p_on, p_in) in positions(g).items()}
            n = len(g)
            starts = rng.integers(0, n, size=(N_BOOT, int(np.ceil(n / BLOCK))))
            idx = ((starts[:, :, None] + np.arange(BLOCK)[None, None, :]) % n).reshape(N_BOOT, -1)[:, :n]
            boot = {k: v[idx].mean(axis=1) / v[idx].std(axis=1, ddof=1) * np.sqrt(252) for k, v in xs.items()}
            point = {k: sharpe(v) for k, v in xs.items()}

            items = [(LABELS[k], point[k], boot[k]) for k in STRATS]
            items += [(f"{LABELS[a_]} minus {LABELS[b_]}", point[a_] - point[b_], boot[a_] - boot[b_])
                      for a_, b_ in [("overnight", "buy_hold"), (f"on_{SIG}", "overnight"), (f"on_{SIG}", "buy_hold")]]
            for item, pt, dist in items:
                lo, hi = np.percentile(dist, [2.5, 97.5])
                rows.append({"ticker": t, "sample": split, "item": item, "sharpe": pt, "ci_low": lo, "ci_high": hi,
                             "excludes_0": "yes" if lo > 0 or hi < 0 else ""})
    return pd.DataFrame(rows)

def flagged(G):
    rows = []
    for t, g0 in G.items():
        for split, (a, b) in SAMPLES.items():
            g = window(g0, a, b)
            keep = ~g[EXCLUDE_FLAGS].any(axis=1).to_numpy()
            for name, (p_on, p_in) in positions(g).items():
                m_all, m_ex = evaluate(g, p_on, p_in, 0), evaluate(g[keep].reset_index(drop=True), p_on[keep], p_in[keep], 0)
                rows.append({"ticker": t, "sample": split, "strategy": LABELS[name], "days_dropped": int((~keep).sum()),
                             "ann_%": m_all["ann_ret_%"], "ann_excl_%": m_ex["ann_ret_%"]})
    return pd.DataFrame(rows)

def timing(G, rng):
    rows = []
    for t, g0 in G.items():
        for split, (a, b) in SAMPLES.items():
            g = window(g0, a, b)
            for lag, label in [(0, "previous close (used)"), (-1, "same close")]:
                p_on, p_in = positions(g, lag)[f"on_{SIG}"]
                p_ann, _ = placebo(g.assign(_lag=p_on.astype(bool)), "_lag", 0.0, rng)
                real = ann_return(run(g, p_on, p_in, 0.0)[0], years_of(g))
                m = evaluate(g, p_on, p_in, 1)
                rows.append({"ticker": t, "sample": split, "signal": label, "ann_% 1bp": m["ann_ret_%"],
                             "sharpe 1bp": m["sharpe"], "placebo_percentile": (p_ann < real).mean() * 100})
    return pd.DataFrame(rows)

# Weight is target / trailing vol capped at 1, vol is the 21-day std up to yesterday and the
# target its 252-day median. The rest sits in cash and costs apply to weight changes
def vol_scaled(G, ret):
    rows = []
    for t, g0 in G.items():
        r = ret[ret["ticker"] == t].sort_values("date").set_index("date")
        vol = r[["overnight", "intraday", "ret"]].rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().shift(1)
        w = (vol.rolling(VOL_TARGET_WINDOW, min_periods=VOL_TARGET_WINDOW).median() / vol).clip(upper=1.0)
        g = window(g0, *SAMPLES["test"])
        ww = w.reindex(g["date"]).fillna(1.0).to_numpy()
        zero, pos = np.zeros(len(g)), positions(g)
        scaled = {"buy_hold": (ww[:, 2], ww[:, 2]), "overnight": (ww[:, 0], zero), "intraday": (zero, ww[:, 1]),
                  f"on_{SIG}": (pos[f"on_{SIG}"][0] * ww[:, 0], zero)}
        for name in STRATS:
            w_on, w_in = scaled[name]
            seq = np.empty(2 * len(g))
            seq[0::2], seq[1::2] = w_on, w_in
            ch = np.abs(np.diff(np.concatenate([[0.0], seq, [0.0]])))
            on = w_on * g["overnight"].to_numpy() + (1 - w_on) * g["rf_overnight"].to_numpy()
            intra = w_in * g["intraday"].to_numpy() + (1 - w_in) * g["rf_intraday"].to_numpy()
            gross = (1 - 1e-4 * ch[0::2][:len(g)]) * (1 + on) * (1 - 1e-4 * ch[1::2][:len(g)]) * (1 + intra)
            gross[-1] *= 1 - 1e-4 * ch[-1]
            m0, m1 = evaluate(g, *pos[name], 1), metrics(gross - 1, g["rf"].to_numpy(), ch.sum(), np.mean(w_on), years_of(g))
            rows.append({"ticker": t, "strategy": LABELS[name], "sharpe": m0["sharpe"], "sharpe_scaled": m1["sharpe"],
                         "max_dd_%": m0["max_dd_%"], "max_dd_scaled_%": m1["max_dd_%"]})
    return pd.DataFrame(rows)

def chart(G, cost_bp, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, t in zip(axes, TICKERS):
        g = G[t]
        for name, (p_on, p_in) in positions(g).items():
            eq = np.cumprod(1 + run(g, p_on, p_in, cost_bp / 1e4)[0])
            ax.plot(g["date"], eq, color=COLORS[name], lw=2, label=LABELS[name])
            ax.annotate(f"{LABELS[name]}  {eq[-1]:.1f}x", (g["date"].iloc[-1], eq[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=8, color="#0b0b0b")
        ax.set_yscale("log")
        for x, lab in [(SAMPLES["validation"][0], "validation"), (SAMPLES["test"][0], "test")]:
            ax.axvline(pd.Timestamp(x), color="#8a8984", lw=1, ls=":")
            ax.text(pd.Timestamp(x), ax.get_ylim()[1], f" {lab}", va="top", fontsize=8, color="#52514e")
        ax.axhline(1, color="#c3c2b7", lw=0.8)
        ax.set_title(f"{t}: growth of $1, {g['date'].min().year}-{g['date'].max().year}, {cost_bp} bp per trade",
                     loc="left", fontsize=11, color="#0b0b0b")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.set_facecolor("#fcfcfb")

        # room on the right for the labels, a tick every 4 years
        first, last = g["date"].min().year, g["date"].max().year
        ax.set_xlim(g["date"].min(), g["date"].max() + pd.Timedelta(days=1400))
        ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(first + (4 - first % 4) % 4, last + 1, 4)])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)

def report(source="crsp"):
    rng = np.random.default_rng(9)
    ret, sig = load_sample(source, "all")
    G = {t: legs(ret, sig, t, "all") for t in TICKERS}

    era_perf, era_gap = eras(G, rng)
    picks, wf = walk_forward(G)
    boot = bootstrap(G, rng)
    tim = timing(G, rng)
    for c in COSTS:
        chart(G, c, ROOT / "reports" / "figures" / f"step9_cumulative_{c}bp.png")
    era_sharpe = era_perf.pivot_table(index=["ticker", "era", "cost_bp"], columns="strategy", values="sharpe", sort=False)

    out = [f"# Step 9: robustness ({source})", "",
           f"Generated by `python src/robustness.py {source}` on the whole sample including the test period. "
           f"Descriptive only, nothing here changes the frozen signal `{SIG}`.", "",
           "![Growth of $1 before costs](figures/step9_cumulative_0bp.png)", "",
           "![Growth of $1 at 1 bp per trade](figures/step9_cumulative_1bp.png)", "",
           "## 1. By era", "",
           "Overnight minus intraday excess return per day, and the signal against random nights:", "",
           md_table(era_gap, "{:.2f}"), "",
           "Sharpe by era:", "", md_table(era_sharpe[[LABELS[s] for s in STRATS]], "{:.2f}"), "",
           "## 2. Without each strategy's best 1% of days (0 bp)", "", md_table(drop_best(G), "{:.2f}"), "",
           "## 3. Weekend vs weekday nights", "", md_table(weekend(G), "{:.2f}"), "",
           "## 4. Walk-forward re-pick of the signal", "", md_table(picks, "{:.3f}"), "", md_table(wf, "{:.2f}"), "",
           f"## 5. Bootstrap 95% intervals for Sharpe at 1 bp ({BLOCK}-day blocks, {N_BOOT} draws)", "", md_table(boot, "{:.2f}"), "",
           "## 6. Without flagged days (0 bp)", "", md_table(flagged(G), "{:.2f}"), "",
           "## 7. Signal timing", "",
           "The same-close version uses the day's own close and VIX, which a market-on-close order cannot know in time.", "",
           md_table(tim, "{:.2f}"), "",
           "## 8. Volatility-scaled sizing in the test period (1 bp)", "", md_table(vol_scaled(G, ret), "{:.2f}")]

    p = ROOT / "reports" / "step9_robustness.md"
    p.write_text("\n".join(out))
    print(f"wrote {p.relative_to(ROOT)}")

if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "crsp")
