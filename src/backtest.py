# Step 6: backtest engine, run on the train sample only (1993-2012)
#
#     python src/backtest.py          # yfinance
#     python src/backtest.py crsp
#
# Each day has an overnight leg and an intraday leg, and every leg is either
# fully in the ETF or in cash. Cash earns the T-bill rate overnight only.
# Every change of position is a trade, so holding one night costs two trades.
#
#     buy_hold      in on every leg
#     intraday      in on intraday legs, cash overnight
#     overnight     in on overnight legs, cash intraday
#     on_<signal>   hold the night when yesterday's signal is on
#     placebo       1000 random sets of nights, same count as the signal

import sys

import numpy as np
import pandas as pd

from checks import md_table
from config import END, ROOT, SAMPLES
from data_layer import load_processed
from signals import SIGNALS

TRAIN_END = pd.Timestamp(SAMPLES["train"][1])
COSTS_BP = [0, 1, 2, 5, 10]
N_PLACEBO = 1000
SPREAD_WINDOW = 21  # days averaged in the Corwin-Schultz estimate
TICKERS = ["SPY", "QQQ", "IWM"]


# Data

# "all" is the whole data set, only used for robustness checks
def _bounds(sample):
    return (SAMPLES["train"][0], END) if sample == "all" else SAMPLES[sample]


# Rows up to the end of the sample, nothing later is loaded
# Earlier rows only warm up the spread estimate
def load_sample(source, sample="train"):
    end = pd.Timestamp(_bounds(sample)[1])
    ret = load_processed(f"returns_{source}")
    ret = ret[ret["date"] <= end].copy()
    sig = load_processed(f"signals_{source}")
    sig = sig[sig["trade_date"] <= end].copy()

    assert ret["date"].max() <= end and sig["trade_date"].max() <= end
    if sample not in ("test", "all"):
        assert "test" not in set(ret["sample"]) | set(sig["sample"])
    return ret, sig


def load_train(source):
    ret, sig = load_sample(source, "train")
    assert set(ret["sample"]) == {"train"} and set(sig["sample"]) == {"train"}
    return ret, sig


# One-way cost per day, half the Corwin-Schultz (2012) high-low spread
# Negative estimates are set to zero and only past days are averaged
def corwin_schultz(g):
    h, l, c = g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy()
    split = g["split_factor"].to_numpy()
    ph, pl, pc = (np.roll(x, 1) / split for x in (h, l, c))

    # adjust today's range for the overnight gap
    up = np.where(l > pc, l - pc, 0.0)
    down = np.where(h < pc, pc - h, 0.0)
    h2, l2 = h - up + down, l - up + down

    k = 3 - 2 * np.sqrt(2)
    beta = np.log(h2 / l2) ** 2 + np.log(ph / pl) ** 2
    gamma = np.log(np.maximum(h2, ph) / np.minimum(l2, pl)) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = np.maximum(2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)), 0.0)
    s[0] = np.nan

    spread = pd.Series(s, index=g.index).rolling(SPREAD_WINDOW, min_periods=5).mean().shift(1)
    return spread / 2


# One-way cost from the closing bid and ask (CRSP), known at the previous close
def quoted_half_spread(g):
    mid = (g["bid"] + g["ask"]) / 2
    return ((g["ask"] - g["bid"]) / 2 / mid).where(mid > 0).shift(1)


# Daily frame for one ETF over its common window inside the sample
# window can cut it further, the spread is computed before cutting
def legs(ret, sig, ticker, sample="train", window=None):
    g = ret[ret["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    has_bidask = g["bid"].notna().any() and g["ask"].notna().any()
    g["cost_spread"] = quoted_half_spread(g) if has_bidask else corwin_schultz(g)
    spread_source = "closing bid/ask" if has_bidask else "Corwin-Schultz"

    s = sig[sig["ticker"] == ticker].set_index("trade_date")[SIGNALS]
    for c in SIGNALS:
        g[c] = s[c].reindex(g["date"]).to_numpy()  # signal from the previous close

    start = max(g.loc[g[SIGNALS].notna().all(axis=1), "date"].min(), pd.Timestamp(_bounds(sample)[0]))
    g = g[g["date"] >= start].reset_index(drop=True)
    if window is not None:
        g = g[(g["date"] >= window[0]) & (g["date"] <= window[1])].reset_index(drop=True)

    assert sample == "all" or set(g["sample"]) == {sample}
    assert g[SIGNALS].notna().all().all()

    for c in SIGNALS:
        g[c] = g[c].astype(bool)
    g["cost_spread"] = g["cost_spread"].ffill().bfill()
    return g, spread_source


# Engine

# Daily returns of a strategy
# pos_on and pos_in are 0/1 per day, cost is one-way per trade (number or daily array)
def run(g, pos_on, pos_in, cost):
    pos_on, pos_in = np.asarray(pos_on, float), np.asarray(pos_in, float)
    n = len(g)
    cost = np.broadcast_to(np.asarray(cost, float), (n,))

    # legs in order on_0, in_0, on_1, in_1 ... starting and ending flat
    seq = np.empty(2 * n)
    seq[0::2], seq[1::2] = pos_on, pos_in
    changes = np.abs(np.diff(np.concatenate([[0.0], seq, [0.0]])))  # before each leg plus the final exit
    trade_on, trade_in = changes[0:2 * n:2], changes[1:2 * n:2]
    exit_end = changes[-1]

    on = np.where(pos_on == 1, g["overnight"].to_numpy(), g["rf_overnight"].to_numpy())
    intra = np.where(pos_in == 1, g["intraday"].to_numpy(), g["rf_intraday"].to_numpy())
    gross = (1 - cost * trade_on) * (1 + on) * (1 - cost * trade_in) * (1 + intra)
    gross[-1] *= 1 - cost[-1] * exit_end

    trades = trade_on.sum() + trade_in.sum() + exit_end
    return gross - 1, trades


def metrics(daily, rf, trades, nights_in, years):
    eq = np.cumprod(1 + daily)
    xs = daily - rf
    dd = eq / np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:] - 1

    return {
        "cum_ret_%": (eq[-1] - 1) * 100,
        "ann_ret_%": (eq[-1] ** (1 / years) - 1) * 100,
        "sharpe": xs.mean() / xs.std(ddof=1) * np.sqrt(252) if xs.std() > 0 else np.nan,
        "max_dd_%": dd.min() * 100,
        "nights_in_%": nights_in * 100,
        "trades_per_yr": trades / years,
    }


def strategies(g, signals=SIGNALS):
    n = len(g)
    one, zero = np.ones(n), np.zeros(n)
    out = {"buy_hold": (one, one), "intraday": (zero, one), "overnight": (one, zero)}

    for s in signals:
        out[f"on_{s}"] = (g[s].to_numpy().astype(float), zero)
    return out


def years_of(g):
    return len(g) / 252


def evaluate(g, name, pos_on, pos_in, cost, cost_label):
    daily, trades = run(g, pos_on, pos_in, cost)
    m = metrics(daily, g["rf"].to_numpy(), trades, np.mean(pos_on), years_of(g))
    return {"strategy": name, "cost": cost_label, **m}


# Cost per trade (bp) where the annual return falls to target_ann, by bisection
def breakeven(g, pos_on, pos_in, target_ann):
    yrs = years_of(g)

    def ann(c):
        d, _ = run(g, pos_on, pos_in, c / 1e4)
        return np.prod(1 + d) ** (1 / yrs) - 1

    if ann(0) <= target_ann:
        return np.nan  # worse than the benchmark even with no costs

    lo, hi = 0.0, 1.0
    while ann(hi) > target_ann and hi < 1e4:
        hi *= 2

    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ann(mid) > target_ann else (lo, mid)
    return (lo + hi) / 2


# Random sets of nights with the same count as the signal
def placebo(g, signal, cost, rng):
    n = len(g)
    k = int(g[signal].sum())
    cost = np.broadcast_to(np.asarray(cost, float), (n,))
    on, rf_on = g["overnight"].to_numpy(), g["rf_overnight"].to_numpy()
    rf = g["rf"].to_numpy()
    yrs = years_of(g)

    ann, sharpe = np.empty(N_PLACEBO), np.empty(N_PLACEBO)
    for i in range(N_PLACEBO):
        pos = np.zeros(n)
        pos[rng.choice(n, k, replace=False)] = 1
        d, _ = run_overnight_only(pos, on, rf_on, cost)
        xs = d - rf
        ann[i] = np.prod(1 + d) ** (1 / yrs) - 1
        sharpe[i] = xs.mean() / xs.std(ddof=1) * np.sqrt(252)
    return ann, sharpe


# Faster run() for overnight-only strategies, every held night is two trades
def run_overnight_only(pos, on, rf_on, cost):
    gross = np.where(pos == 1, (1 - cost) ** 2 * (1 + on), 1 + rf_on)
    return gross - 1, 2 * pos.sum()


# Report

# All strategies on one sample, or a window inside it
def backtest(source="yfinance", sample="train", signals=SIGNALS, window=None):
    ret, sig = load_sample(source, sample)
    rng = np.random.default_rng(2026)
    perf, be, plc, windows, spread_by_year = [], [], [], [], []

    for t in TICKERS:
        g, spread_src = legs(ret, sig, t, sample, window)
        yrs = years_of(g)

        windows.append({"ticker": t, "first_day": g["date"].min().date(), "last_day": g["date"].max().date(),
                        "days": len(g), "years": yrs,
                        "mean_spread_cost_bp": g["cost_spread"].mean() * 1e4,
                        "median_spread_cost_bp": g["cost_spread"].median() * 1e4, "spread_source": spread_src})
        strats = strategies(g, signals)
        for name, (p_on, p_in) in strats.items():
            for c in COSTS_BP:
                perf.append({"ticker": t, **evaluate(g, name, p_on, p_in, c / 1e4, f"{c} bp")})
            perf.append({"ticker": t, **evaluate(g, name, p_on, p_in, g["cost_spread"].to_numpy(), "spread est.")})

        # breakeven against cash and against buy-and-hold at zero cost
        cash_ann = np.prod(1 + g["rf"].to_numpy()) ** (1 / yrs) - 1
        bh_ann = np.prod(1 + run(g, *strats["buy_hold"], 0.0)[0]) ** (1 / yrs) - 1
        for name, (p_on, p_in) in strats.items():
            if name == "buy_hold":
                continue
            be.append({"ticker": t, "strategy": name,
                       "breakeven_vs_cash_bp": breakeven(g, p_on, p_in, cash_ann),
                       "breakeven_vs_buy_hold_bp": breakeven(g, p_on, p_in, bh_ann)})

        # placebo at zero cost, a flat cost hits every draw the same so the ranking holds
        spread_by_year.append(g.groupby(g["date"].dt.year)["cost_spread"].mean().rename(t) * 1e4)
        for s in signals:
            for cost, label in [(0.0, "0 bp")]:
                daily, _ = run(g, g[s].to_numpy().astype(float), np.zeros(len(g)), cost)
                xs = daily - g["rf"].to_numpy()
                real_ann = np.prod(1 + daily) ** (1 / yrs) - 1
                real_sh = xs.mean() / xs.std(ddof=1) * np.sqrt(252)
                p_ann, p_sh = placebo(g, s, cost, rng)
                plc.append({"ticker": t, "signal": s, "cost": label, "nights": int(g[s].sum()),
                            "ann_ret_%": real_ann * 100, "placebo_median_ann_%": np.median(p_ann) * 100,
                            "ann_percentile": (p_ann < real_ann).mean() * 100,
                            "sharpe": real_sh, "placebo_median_sharpe": np.median(p_sh),
                            "placebo_5_95_sharpe": f"{np.percentile(p_sh, 5):.2f} / {np.percentile(p_sh, 95):.2f}",
                            "sharpe_percentile": (p_sh < real_sh).mean() * 100})

    spread_tab = pd.concat(spread_by_year, axis=1).rename_axis("year")
    return pd.DataFrame(perf), pd.DataFrame(be), pd.DataFrame(plc), pd.DataFrame(windows), spread_tab


# Check the engine against numbers computed another way
def check_engine(ret, sig, sample="train"):
    g, _ = legs(ret, sig, "SPY", sample)
    rows = []

    d, tr = run(g, np.ones(len(g)), np.ones(len(g)), 0.0)
    rows.append({"check": "buy_hold at 0 bp = compounded ret",
                 "engine": np.prod(1 + d), "independent": np.prod(1 + g["ret"]), "trades": tr})

    d, tr = run(g, np.ones(len(g)), np.zeros(len(g)), 0.0)
    rows.append({"check": "overnight at 0 bp = prod(1+overnight) x prod(1+0) (intraday in cash)",
                 "engine": np.prod(1 + d), "independent": np.prod(1 + g["overnight"]), "trades": tr})

    d, tr = run(g, np.zeros(len(g)), np.zeros(len(g)), 0.0)
    rows.append({"check": "all cash = compounded T-bill", "engine": np.prod(1 + d),
                 "independent": np.prod(1 + g["rf"]), "trades": tr})

    pos = g["sig_b"].to_numpy().astype(float)
    d1, t1 = run(g, pos, np.zeros(len(g)), 5e-4)
    d2, t2 = run_overnight_only(pos, g["overnight"].to_numpy(), g["rf_overnight"].to_numpy(),
                                np.full(len(g), 5e-4))
    rows.append({"check": "placebo fast path = engine (sig_b, 5 bp)", "engine": np.prod(1 + d1),
                 "independent": np.prod(1 + d2), "trades": t1 - t2})
    return pd.DataFrame(rows)


SETUP = [
    "## Set-up", "",
    "- Each day is two legs, overnight (previous close to open) and intraday (open to close). Each leg is either "
    "fully in the ETF or fully in cash; no leverage.",
    "- Cash earns the T-bill rate on the overnight leg (where all T-bill interest sits, step 4); cash intraday "
    "earns 0. ETF legs earn `overnight` (total return, dividends included) and `intraday`.",
    "- A trade is any change of position between consecutive legs, plus the entry at the start and the exit at "
    "the end. Holding one night = buy at the close, sell at the next open = 2 trades. Cost is per trade, "
    "one-way, as a fraction of the position.",
    "- `on_<signal>` holds the overnight leg of day d when the signal computed at the close of d-1 is on.",
    "- Spread estimate: half of the Corwin-Schultz (2012) high-low spread, averaged over the previous 21 days, "
    "used as the one-way cost of every trade (closing bid/ask is used instead when the source has it, i.e. CRSP).",
    "- Sharpe: annualised mean daily return in excess of the T-bill over its standard deviation, × √252. "
    "Annual return: compounded, over years = days / 252. Max drawdown on the daily equity curve.",
    "- Breakeven cost: per-trade cost at which the strategy's annual return falls to that of cash (T-bill) or "
    "of buy-and-hold at zero cost.",
    f"- Placebo: {N_PLACEBO} random sets of overnight legs in the same window with the same number of nights as "
    "the signal; percentile = share of placebo draws below the signal. Costs are per night, so with flat costs "
    "the percentile is the same at every cost level; it is shown at 0 bp.", "",
]


# Shared markdown for steps 6 and 7
def body(perf, be, plc, win, spread_tab, eng):
    names = ["buy_hold", "intraday", "overnight"] + [f"on_{s}" for s in SIGNALS]
    cols = ["strategy", "cum_ret_%", "ann_ret_%", "sharpe", "max_dd_%", "nights_in_%", "trades_per_yr"]

    out = ["Common window per ETF:", "",
           md_table(win, "{:.2f}"), "",
           "Corwin-Schultz one-way cost by year (bp, mean of the daily estimate):", "",
           md_table(spread_tab.round(1).astype(object).where(spread_tab.notna(), ""), "{:.1f}"), "",
           "Engine checks (SPY window):", "", md_table(eng, "{:.8f}"), ""]

    out += ["## Performance", ""]
    for t in TICKERS:
        out += [f"### {t}", ""]
        for c in ["0 bp", "1 bp", "2 bp", "5 bp", "10 bp", "spread est."]:
            p = perf[(perf["ticker"] == t) & (perf["cost"] == c)].set_index("strategy").loc[names].reset_index()
            out += [f"**Cost {c} per trade**", "", md_table(p[cols], "{:.2f}"), ""]

    be_tab = be.pivot(index="strategy", columns="ticker",
                      values=["breakeven_vs_cash_bp", "breakeven_vs_buy_hold_bp"]).pipe(_flat).loc[names[1:]]
    out += ["## Breakeven cost per trade (bp)", "",
            "One-way cost per trade at which the strategy's annual return drops to the benchmark's. "
            "`-` = below the benchmark even at zero cost.", "",
            md_table(be_tab.round(2).astype(object).where(be_tab.notna(), "-"), "{:.2f}"), "",
            "## Placebo", "", md_table(plc, "{:.2f}"), ""]
    return out


def report(source="yfinance"):
    ret, sig = load_train(source)
    perf, be, plc, win, spread_tab = backtest(source)
    eng = check_engine(ret, sig)

    out = [f"# Step 6: backtest ({source}, train sample only)", "",
           f"Generated by `python src/backtest.py {source}`. Only rows up to {TRAIN_END.date()} are loaded (asserted); "
           "nothing on validation or test rows.", ""]
    out += SETUP
    out += ["Each window starts at the first night on which all five signals exist.", ""]
    out += body(perf, be, plc, win, spread_tab, eng) + [NOTES]

    p = ROOT / "reports" / "step6_backtest.md"
    p.write_text("\n".join(out))
    print(f"wrote {p.relative_to(ROOT)}")
    return perf, be, plc, win


def _flat(df):
    df.columns = [f"{t} vs {'cash' if 'cash' in v else 'B&H'}" for v, t in df.columns]
    return df[[f"{t} vs {b}" for b in ["cash", "B&H"] for t in TICKERS]]


NOTES = """## What the train backtest says (yfinance snapshot)

- **Before costs, always-overnight beats buy-and-hold on every ETF**, with a higher return, about twice the
  Sharpe ratio and a smaller drawdown (SPY 10.2% a year, Sharpe 0.68, max drawdown -33%, against 7.9%, 0.33,
  -55%). Always-intraday is the worst strategy on all three ETFs.
- **But it trades 504 times a year, so costs decide.** It stops beating cash at about 1-1.7 bp per trade and stops
  beating buy-and-hold at 0.4 bp (SPY) to 2.2 bp (QQQ, whose buy-and-hold starts at the March 2000 peak). At 2 bp
  per trade it is already at or below cash.
- **None of the five signals adds value.** At zero cost, every conditional strategy has a lower Sharpe ratio than
  always-overnight, and against the placebo of random nights the signals sit at or below the median: no signal
  beats more than about two thirds of the random draws (best: SPY `sig_c`, 65th percentile on annual return but
  43rd on Sharpe). `sig_bc` is in the bottom 3% on annual return for all three ETFs, and `sig_b` and `sig_ac` for
  QQQ and `sig_ac` for IWM are in the bottom 6%.
  So the nights the signals pick are, if anything, worse than random, matching step 5.
- **The spread estimate is not usable as a realistic cost for these ETFs.** Corwin-Schultz gives one-way costs
  of 8-35 bp for SPY, which is close to the tick-size spread of the mid-1990s but far above SPY's quoted spread
  after decimalisation (2001), where it is well under 1 bp one-way. The estimator picks up intraday volatility
  (it peaks in 2008) and is known to overstate spreads for very liquid securities. Results with it are shown for
  completeness but are a pessimistic bound; the flat 0-10 bp grid and the breakeven costs are the ones to read.
  CRSP's closing bid/ask will replace it automatically once loaded.
- Placebo percentiles are shown at 0 bp only. With a flat cost every draw pays the same per night, so the ranking
  is identical at 1-10 bp. With the spread estimate the ranking would mostly reflect that signal nights fall on
  high-volatility days, where the estimate is larger, rather than the signal itself.
"""


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "yfinance")
