import sys

import numpy as np
import pandas as pd

from config import END, SAMPLES, TICKERS, md_table, write_report
from data_layer import load_processed
from signals import SIGNALS

COSTS_BP = [0, 1, 2, 5, 10]
N_PLACEBO = 1000

def _bounds(sample):
    return (SAMPLES["train"][0], END) if sample == "all" else SAMPLES[sample]

# Rows up to the end of the sample, earlier rows only warm up the spread
def load_sample(source, sample="train"):
    end = pd.Timestamp(_bounds(sample)[1])
    ret = load_processed(f"returns_{source}")
    sig = load_processed(f"signals_{source}")
    ret, sig = ret[ret["date"] <= end].copy(), sig[sig["trade_date"] <= end].copy()
    if sample not in ("test", "all"):
        assert "test" not in set(ret["sample"]) | set(sig["sample"])
    return ret, sig

# Half the closing quoted spread, known at the previous close, a crossed quote is not a cost
def quoted_half_spread(g):
    mid = (g["bid"] + g["ask"]) / 2
    return ((g["ask"] - g["bid"]) / 2 / mid).where((mid > 0) & (g["ask"] >= g["bid"])).shift(1)

def legs(ret, sig, ticker, sample="train", window=None):
    g = ret[ret["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    g["cost_spread"] = quoted_half_spread(g)
    s = sig[sig["ticker"] == ticker].set_index("trade_date")[SIGNALS]
    for c in SIGNALS:
        g[c] = s[c].reindex(g["date"]).to_numpy()

    start = max(g.loc[g[SIGNALS].notna().all(axis=1), "date"].min(), pd.Timestamp(_bounds(sample)[0]))
    g = g[g["date"] >= start]
    if window is not None:
        g = g[(g["date"] >= window[0]) & (g["date"] <= window[1])]
    assert sample == "all" or set(g["sample"]) == {sample}

    # days with no open can't be traded, a missing signal input means staying in cash
    g = g[g[["overnight", "intraday", "ret", "rf"]].notna().all(axis=1)].reset_index(drop=True)
    for c in SIGNALS:
        g[c] = g[c].astype("boolean").fillna(False).astype(bool)
    g["cost_spread"] = g["cost_spread"].ffill().bfill()
    return g

# Daily returns, pos_on and pos_in are 0/1 per day, cost is one-way per trade
def run(g, pos_on, pos_in, cost):
    pos_on, pos_in = np.asarray(pos_on, float), np.asarray(pos_in, float)
    n = len(g)
    cost = np.broadcast_to(np.asarray(cost, float), (n,))

    # legs in order on_0, in_0, on_1, in_1 ..., flat before and after
    seq = np.empty(2 * n)
    seq[0::2], seq[1::2] = pos_on, pos_in
    changes = np.abs(np.diff(np.concatenate([[0.0], seq, [0.0]])))
    trade_on, trade_in, exit_end = changes[0:2 * n:2], changes[1:2 * n:2], changes[-1]

    on = np.where(pos_on == 1, g["overnight"].to_numpy(), g["rf_overnight"].to_numpy())
    intra = np.where(pos_in == 1, g["intraday"].to_numpy(), g["rf_intraday"].to_numpy())
    gross = (1 - cost * trade_on) * (1 + on) * (1 - cost * trade_in) * (1 + intra)
    gross[-1] *= 1 - cost[-1] * exit_end
    return gross - 1, trade_on.sum() + trade_in.sum() + exit_end

# Faster run() for overnight-only positions, every held night is two trades
def run_overnight_only(pos, on, rf_on, cost):
    return np.where(pos == 1, (1 - cost) ** 2 * (1 + on), 1 + rf_on) - 1, 2 * pos.sum()

def years_of(g):
    return len(g) / 252

def sharpe(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(252)

def metrics(daily, rf, trades, nights_in, years):
    eq = np.cumprod(1 + daily)
    xs = daily - rf
    dd = eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1
    return {"cum_ret_%": (eq[-1] - 1) * 100, "ann_ret_%": (eq[-1] ** (1 / years) - 1) * 100,
            "sharpe": sharpe(xs) if xs.std() > 0 else np.nan,
            "max_dd_%": dd.min() * 100, "nights_in_%": nights_in * 100, "trades_per_yr": trades / years}

def strategies(g, signals=SIGNALS):
    one, zero = np.ones(len(g)), np.zeros(len(g))
    out = {"buy_hold": (one, one), "intraday": (zero, one), "overnight": (one, zero)}
    out.update({f"on_{s}": (g[s].to_numpy().astype(float), zero) for s in signals})
    return out

def ann_return(daily, years):
    return np.prod(1 + daily) ** (1 / years) - 1

# Cost per trade in bp where the annual return falls to target, by bisection
def breakeven(g, pos_on, pos_in, target):
    yrs = years_of(g)
    ann = lambda c: ann_return(run(g, pos_on, pos_in, c / 1e4)[0], yrs)
    if ann(0) <= target:
        return np.nan
    lo, hi = 0.0, 1.0
    while ann(hi) > target and hi < 1e4:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ann(mid) > target else (lo, mid)
    return (lo + hi) / 2

# Random sets of nights with the same count as the signal
def placebo(g, signal, cost, rng):
    n, k, yrs = len(g), int(g[signal].sum()), years_of(g)
    cost = np.broadcast_to(np.asarray(cost, float), (n,))
    on, rf_on, rf = g["overnight"].to_numpy(), g["rf_overnight"].to_numpy(), g["rf"].to_numpy()
    ann, sh = np.empty(N_PLACEBO), np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        pos = np.zeros(n)
        pos[rng.choice(n, k, replace=False)] = 1
        d, _ = run_overnight_only(pos, on, rf_on, cost)
        ann[i] = ann_return(d, yrs)
        sh[i] = sharpe(d - rf)
    return ann, sh

def backtest(source="crsp", sample="train", signals=SIGNALS, window=None):
    ret, sig = load_sample(source, sample)
    rng = np.random.default_rng(2026)
    perf, be, plc, windows, spread_by_year = [], [], [], [], []

    for t in TICKERS:
        g = legs(ret, sig, t, sample, window)
        yrs, rf = years_of(g), g["rf"].to_numpy()
        windows.append({"ticker": t, "first_day": g["date"].min().date(), "last_day": g["date"].max().date(),
                        "days": len(g), "median_spread_bp": g["cost_spread"].median() * 1e4})
        spread_by_year.append(g.groupby(g["date"].dt.year)["cost_spread"].mean().rename(t) * 1e4)

        strats = strategies(g, signals)
        for name, (p_on, p_in) in strats.items():
            for label, cost in [(f"{c} bp", c / 1e4) for c in COSTS_BP] + [("spread", g["cost_spread"].to_numpy())]:
                d, trades = run(g, p_on, p_in, cost)
                perf.append({"ticker": t, "strategy": name, "cost": label, **metrics(d, rf, trades, np.mean(p_on), yrs)})

        cash_ann = ann_return(rf, yrs)
        bh_ann = ann_return(run(g, *strats["buy_hold"], 0.0)[0], yrs)
        for name, (p_on, p_in) in strats.items():
            if name != "buy_hold":
                be.append({"ticker": t, "strategy": name, "vs_cash": breakeven(g, p_on, p_in, cash_ann),
                           "vs_buy_hold": breakeven(g, p_on, p_in, bh_ann)})

        # at zero cost, a flat cost hits every random draw the same way
        for s in signals:
            d, _ = run(g, g[s].to_numpy().astype(float), np.zeros(len(g)), 0.0)
            real_ann, real_sh = ann_return(d, yrs), sharpe(d - rf)
            p_ann, p_sh = placebo(g, s, 0.0, rng)
            plc.append({"ticker": t, "signal": s, "nights": int(g[s].sum()), "ann_ret_%": real_ann * 100,
                        "placebo_median_ann_%": np.median(p_ann) * 100, "ann_percentile": (p_ann < real_ann).mean() * 100,
                        "sharpe": real_sh, "placebo_median_sharpe": np.median(p_sh), "sharpe_percentile": (p_sh < real_sh).mean() * 100})

    return pd.DataFrame(perf), pd.DataFrame(be), pd.DataFrame(plc), pd.DataFrame(windows), pd.concat(spread_by_year, axis=1).rename_axis("year")

def check_engine(ret, sig, sample="train"):
    g = legs(ret, sig, "SPY", sample)
    one, zero = np.ones(len(g)), np.zeros(len(g))
    pos = g["sig_b"].to_numpy().astype(float)
    rows = [("buy and hold = compounded ret", run(g, one, one, 0.0)[0], g["ret"]),
            ("always overnight = compounded overnight", run(g, one, zero, 0.0)[0], g["overnight"]),
            ("all cash = compounded T-bill", run(g, zero, zero, 0.0)[0], g["rf"]),
            ("fast placebo path = engine at 5 bp", run(g, pos, zero, 5e-4)[0],
             run_overnight_only(pos, g["overnight"].to_numpy(), g["rf_overnight"].to_numpy(), np.full(len(g), 5e-4))[0])]
    return pd.DataFrame([{"check": c, "engine": np.prod(1 + a), "independent": np.prod(1 + np.asarray(b))} for c, a, b in rows])

def body(perf, be, plc, win, eng, names=None):
    names = names or ["buy_hold", "intraday", "overnight"] + [f"on_{s}" for s in SIGNALS]
    cols = ["cost", "strategy", "ann_ret_%", "sharpe", "max_dd_%", "nights_in_%", "trades_per_yr"]
    out = [md_table(win, "{:.2f}"), "", "Engine checks:", "", md_table(eng, "{:.8f}"), "", "## Performance", ""]
    for t in TICKERS:
        p = perf[(perf["ticker"] == t) & perf["strategy"].isin(names) & perf["cost"].isin(["0 bp", "1 bp", "2 bp", "spread"])]
        out += [f"**{t}**", "", md_table(p[cols], "{:.2f}"), ""]

    be_tab = be.pivot(index="strategy", columns="ticker", values=["vs_cash", "vs_buy_hold"])
    be_tab.columns = [f"{t} {v}" for v, t in be_tab.columns]
    be_tab = be_tab.loc[[n for n in names if n != "buy_hold"], [f"{t} {v}" for v in ["vs_cash", "vs_buy_hold"] for t in TICKERS]]
    out += ["## Breakeven cost per trade (bp)", "",
            "Cost at which the annual return falls to cash or to buy-and-hold, `-` means below it even at zero cost.", "",
            md_table(be_tab.round(2).astype(object).where(be_tab.notna(), "-"), "{:.2f}"), "",
            "## Placebo", "",
            f"Percentile of the signal against {N_PLACEBO} random sets of the same number of nights, at 0 bp.", "",
            md_table(plc, "{:.2f}"), ""]
    return out

def report(source="crsp"):
    ret, sig = load_sample(source, "train")
    perf, be, plc, win, _ = backtest(source)

    out = [f"# Step 6: backtest ({source}, train sample only)", "",
           f"Generated by `python src/backtest.py {source}`. Each leg is fully in the ETF or in cash, cash earns the "
           "T-bill overnight, every change of position is a trade and holding one night costs two trades. `spread` "
           "uses the previous closing quoted half-spread as the cost.", ""]
    write_report("step6_backtest.md", out + body(perf, be, plc, win, check_engine(ret, sig)))

if __name__ == "__main__":
    report(*sys.argv[1:2])
