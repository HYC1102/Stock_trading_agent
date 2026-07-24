"""
Portfolio engine for the Donchian + ATR strategy.

basket.py sizes each name in isolation and averages the return streams, which
assumes infinite capital and no concentration limits. This engine instead runs
ONE capital-constrained book: the per-name breakout signals compete for a shared
allocation under portfolio-level risk control.

Allocation each day (decided at the close of t, applied at t+1 -- no look-ahead):

  1. Take the ATR risk-weight of every name currently long (0 if flat). These
     are proportional to 1/volatility, so calmer names get more risk.
  2. Capacity: if more than `max_positions` names signal, keep the top K by
     `mom_window`-day momentum (favour the strongest trends).
  3. Direction: normalise the kept risk-weights to sum to 1 (r_norm).
  4. Vol target: estimate the book's trailing volatility (r_norm on the last
     `vol_window` days of returns) and set gross exposure so annualised vol
     hits `target_vol`, clipped to `max_gross` (long-only, no leverage).
  5. Concentration: cap any single name at `max_name`, then re-cap gross.

Usage:
    python portfolio.py --plot
    python portfolio.py --target-vol 0.20 --max-gross 1.5 --max-positions 8
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import donchian as dc
import atr_strategy as at
from basket import UNIVERSE

PPY = 252


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def curve_stats(ret: pd.Series) -> dict:
    r = ret.dropna()
    if r.empty:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan, total=np.nan)
    eq = (1.0 + r).cumprod()
    n_years = len(r) / PPY
    return dict(
        cagr=eq.iloc[-1] ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan,
        vol=r.std() * np.sqrt(PPY),
        sharpe=r.mean() / r.std() * np.sqrt(PPY) if r.std() > 0 else np.nan,
        maxdd=(eq / eq.cummax() - 1.0).min(),
        total=eq.iloc[-1] - 1.0,
    )


# --------------------------------------------------------------------------- #
# Per-name signal panels
# --------------------------------------------------------------------------- #
def build_panels(tickers, start, end, cfg):
    """Return aligned (target, open_ret, close_ret, close) DataFrames + name list.

    ``target`` holds each name's ATR risk-weight on days it is long (decided at
    the close, UNSHIFTED -- the single execution shift happens at portfolio
    level). ``max_weight`` is set high so the per-name ATR sizing is pure
    1/volatility and never clipped before portfolio scaling.
    """
    tgt, oret, cret, close = {}, {}, {}, {}
    for t in tickers:
        try:
            raw = dc.load_data(t, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {t}: skipped ({exc})")
            continue
        p = at.AtrParams(entry=cfg["entry"], exit=cfg["exit"],
                         risk_frac=0.01, max_weight=10.0)
        df = at.prepare(raw, p)
        target, _ = at.build_weights(df, p)
        tgt[t] = target
        oret[t] = raw["Open"].pct_change()
        cret[t] = raw["Close"].pct_change()
        close[t] = raw["Close"]

    names = list(tgt)
    idx = sorted(set().union(*[s.index for s in tgt.values()]))
    TGT = pd.DataFrame(tgt).reindex(idx)
    OR = pd.DataFrame(oret).reindex(idx)
    CR = pd.DataFrame(cret).reindex(idx)
    CL = pd.DataFrame(close).reindex(idx)
    return TGT, OR, CR, CL, names


# --------------------------------------------------------------------------- #
# The allocation loop
# --------------------------------------------------------------------------- #
def allocate(TGT, CR, CL, cfg):
    """Produce the (unshifted) portfolio weight matrix, decided at each close."""
    names = list(TGT.columns)
    dates = TGT.index
    tgt = TGT.to_numpy()
    cr = CR.to_numpy()
    mom = (CL / CL.shift(cfg["mom_window"]) - 1.0).to_numpy()
    n, m = tgt.shape

    K = cfg["max_positions"]
    vw = cfg["vol_window"]
    tv = cfg["target_vol"]
    mg = cfg["max_gross"]
    cap = cfg["max_name"]

    W = np.zeros((n, m))
    for t in range(n):
        row = tgt[t]
        longs = np.where(np.nan_to_num(row) > 0)[0]
        if longs.size == 0 or t < vw:
            continue
        # capacity: keep top-K by momentum
        if longs.size > K:
            mv = np.nan_to_num(mom[t, longs], nan=-np.inf)
            longs = longs[np.argsort(mv)[::-1][:K]]
        r = row[longs]
        if not np.isfinite(r).all() or r.sum() <= 0:
            continue
        r_norm = r / r.sum()
        # trailing vol of the book in the r_norm direction
        win = np.nan_to_num(cr[t - vw + 1:t + 1][:, longs])
        book_hist = win.dot(r_norm)
        sd = book_hist.std()
        if not np.isfinite(sd) or sd <= 0:
            continue
        est_vol = sd * np.sqrt(PPY)
        gross = min(mg, tv / est_vol)
        w = gross * r_norm
        w = np.minimum(w, cap)          # concentration cap
        s = w.sum()
        if s > mg:                      # re-cap gross after clipping
            w *= mg / s
        W[t, longs] = w
    return pd.DataFrame(W, index=dates, columns=names)


# --------------------------------------------------------------------------- #
# Backtest the book
# --------------------------------------------------------------------------- #
def run_portfolio(TGT, OR, CR, CL, cfg):
    W = allocate(TGT, CR, CL, cfg)
    held = W.shift(1).fillna(0.0)                       # weight decided at prior close

    # Fill at the next open, so weights earn the FOLLOWING open-to-open move; using
    # OR directly would credit the signal bar's own move (one-day look-ahead).
    oret = OR.shift(-1).fillna(0.0)
    gross_ret = (held * oret).sum(axis=1)
    turnover = held.diff().abs().fillna(held.abs()).sum(axis=1)
    cost = turnover * (cfg["cost_bps"] / 10_000.0)
    port_ret = gross_ret - cost

    exposure = held.sum(axis=1)
    n_pos = (held > 0).sum(axis=1)
    contrib = (held * oret).sum(axis=0)                 # per-name additive P&L
    return dict(weights=held, port_ret=port_ret, exposure=exposure,
                n_pos=n_pos, contrib=contrib, turnover=turnover)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(res, OR, cfg, bench_ret):
    port = res["port_ret"]
    ps = curve_stats(port)
    ew = curve_stats(OR.mean(axis=1))                   # equal-weight buy & hold
    active = res["exposure"][res["exposure"] > 0]

    print("\n" + "=" * 66)
    print("  PORTFOLIO ENGINE  (vol-target + caps, long-only)")
    print("=" * 66)
    print(f"  Universe / positions : {OR.shape[1]} names, max {cfg['max_positions']} held")
    print(f"  Target vol / gross   : {cfg['target_vol']*100:.0f}%  /  {cfg['max_gross']*100:.0f}% max")
    print(f"  Per-name cap         : {cfg['max_name']*100:.0f}%")
    print("-" * 66)
    print(f"  {'':<14}{'Portfolio':>12}{'SPY B&H':>12}{'EW basket B&H':>15}")
    bs = curve_stats(bench_ret) if bench_ret is not None else None
    def col(v, pct=True):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return f"{'n/a':>12}"
        return f"{v*100:>11.1f}%" if pct else f"{v:>12.2f}"
    rows = [("Total return", "total"), ("CAGR", "cagr"),
            ("Volatility", "vol"), ("Max drawdown", "maxdd")]
    for label, key in rows:
        line = f"  {label:<14}{col(ps[key])}"
        line += col(bs[key]) if bs else f"{'n/a':>12}"
        line += col(ew[key])
        print(line)
    line = f"  {'Sharpe':<14}{col(ps['sharpe'], False)}"
    line += col(bs['sharpe'], False) if bs else f"{'n/a':>12}"
    line += col(ew['sharpe'], False)
    print(line)
    print("-" * 66)
    print(f"  Realised vol vs target : {ps['vol']*100:.1f}%  (target {cfg['target_vol']*100:.0f}%)")
    print(f"  Avg gross exposure     : {active.mean()*100:.0f}%   "
          f"(invested {(res['exposure']>0).mean()*100:.0f}% of days)")
    print(f"  Avg # positions        : {res['n_pos'][res['n_pos']>0].mean():.1f}")
    print(f"  Turnover / year        : {res['turnover'].sum()/(len(port)/PPY):.1f}x")
    print("=" * 66)
    top = res["contrib"].sort_values(ascending=False)
    print("  Top contributors  :", ", ".join(f"{k} {v*100:+.0f}%" for k, v in top.head(5).items()))
    print("  Worst contributors:", ", ".join(f"{k} {v*100:+.0f}%" for k, v in top.tail(3).items()))
    print("=" * 66 + "\n")


def plot_portfolio(res, OR, bench_ret, path):
    import matplotlib.pyplot as plt

    port = res["port_ret"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    # (0,0) equity curves
    a = ax[0, 0]
    a.plot((1 + port).cumprod(), color="tab:blue", lw=1.5, label="Portfolio")
    if bench_ret is not None:
        a.plot((1 + bench_ret).cumprod(), color="tab:red", lw=1.1, ls="--", label="SPY B&H")
    a.plot((1 + OR.mean(axis=1)).cumprod(), color="gray", lw=1.1, ls=":", label="EW basket B&H")
    a.set_yscale("log"); a.set_ylabel("Growth of $1 (log)")
    a.set_title("Portfolio vs benchmarks"); a.legend(loc="upper left"); a.grid(alpha=0.3, which="both")

    # (0,1) drawdown
    a = ax[0, 1]
    eq = (1 + port).cumprod(); dd = eq / eq.cummax() - 1
    a.fill_between(dd.index, dd * 100, 0, color="tab:blue", alpha=0.4)
    a.set_title("Portfolio drawdown"); a.set_ylabel("%"); a.grid(alpha=0.3)

    # (1,0) exposure + position count
    a = ax[1, 0]
    a.fill_between(res["exposure"].index, res["exposure"] * 100, 0,
                   color="tab:green", alpha=0.3, label="Gross exposure %")
    a.set_ylabel("Gross exposure %"); a.set_ylim(0, 110)
    a2 = a.twinx(); a2.plot(res["n_pos"].index, res["n_pos"], color="tab:purple", lw=0.8)
    a2.set_ylabel("# positions", color="tab:purple")
    a.set_title("Exposure & position count"); a.grid(alpha=0.3)

    # (1,1) per-name contribution
    a = ax[1, 1]
    c = res["contrib"].sort_values() * 100
    a.barh(c.index, c.values, color=["tab:red" if x < 0 else "tab:blue" for x in c.values])
    a.set_title("Per-name P&L contribution (%)"); a.grid(alpha=0.3, axis="x")
    a.tick_params(labelsize=7)

    fig.suptitle("Donchian + ATR portfolio engine", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Chart saved to {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Donchian + ATR portfolio engine.")
    p.add_argument("--tickers", default=None, help="Comma-separated; default = 30-name universe.")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--entry", type=int, default=20)
    p.add_argument("--exit", type=int, default=10)
    p.add_argument("--target-vol", type=float, default=0.15)
    p.add_argument("--max-gross", type=float, default=1.0)
    p.add_argument("--max-positions", type=int, default=10)
    p.add_argument("--max-name", type=float, default=0.20)
    p.add_argument("--mom-window", type=int, default=60)
    p.add_argument("--vol-window", type=int, default=60)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-path", default="portfolio.png")
    return p.parse_args()


def main():
    args = parse_args()
    tickers = ([t.strip().upper() for t in args.tickers.split(",")]
               if args.tickers else list(UNIVERSE))
    cfg = dict(entry=args.entry, exit=args.exit, target_vol=args.target_vol,
               max_gross=args.max_gross, max_positions=args.max_positions,
               max_name=args.max_name, mom_window=args.mom_window,
               vol_window=args.vol_window, cost_bps=args.cost_bps)

    print(f"Building portfolio over {len(tickers)} names...")
    TGT, OR, CR, CL, names = build_panels(tickers, args.start, args.end, cfg)
    res = run_portfolio(TGT, OR, CR, CL, cfg)

    bench_ret = None
    if args.benchmark:
        try:
            b = dc.load_data(args.benchmark, args.start, args.end)
            bench_ret = b["Open"].pct_change().reindex(OR.index).fillna(0.0)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! benchmark {args.benchmark}: {exc}")

    print_report(res, OR, cfg, bench_ret)
    if args.plot:
        plot_portfolio(res, OR, bench_ret, args.plot_path)


if __name__ == "__main__":
    main()
