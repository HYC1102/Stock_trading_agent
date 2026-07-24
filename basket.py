"""
Basket / cross-sectional test for the Donchian breakout strategy.

The sweep and walk-forward both look impressive on TSLA -- but TSLA is a 100x
trending outlier, exactly the stock a breakout system is built to love. The
only way to tell a real edge from a story about one ticker is to run the SAME,
UN-OPTIMISED default (20/10) across a diverse universe and see whether it still
helps: trending winners, choppy cyclicals, outright decliners, and indices.

For each name we compare the strategy against simply buying and holding it, and
we build an equal-weight basket curve for the portfolio-level view. No parameter
is tuned per name -- that is the whole point (no data-snooping here).

Note on survivorship: yfinance only serves still-listed tickers, so a truly
clean test would also include delisted names (this understates how bad the
"loser" bucket really was). We include several large drawdown names as a proxy.

Usage:
    python basket.py --plot
    python basket.py --engine atr --plot
    python basket.py --tickers SPY,QQQ,KO,XOM --entry 20 --exit 10
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import donchian as dc
import atr_strategy as at

# Universe grouped by character, so we can see WHERE the edge lives.
UNIVERSE = {
    # trending / momentum names
    "TSLA": "trend winner", "NVDA": "trend winner", "AVGO": "trend winner",
    "AMD": "volatile", "NFLX": "volatile",
    # mega-cap tech
    "AAPL": "mega tech", "MSFT": "mega tech", "AMZN": "mega tech",
    "GOOGL": "mega tech", "META": "mega tech",
    # broad indices
    "SPY": "index", "QQQ": "index", "IWM": "index",
    # defensives / staples
    "KO": "defensive", "PFE": "defensive", "JNJ": "defensive",
    "PG": "defensive", "WMT": "defensive",
    # cyclicals / energy / industrials
    "XOM": "cyclical", "CVX": "cyclical", "CAT": "cyclical", "F": "cyclical",
    # financials
    "JPM": "financial", "GS": "financial", "BAC": "financial",
    # laggards / blow-ups
    "INTC": "decliner", "T": "decliner", "BA": "decliner",
    "DIS": "decliner", "PYPL": "blow-up",
}
PERIODS_PER_YEAR = 252


def curve_stats(ret: pd.Series) -> dict:
    """CAGR / Sharpe / max drawdown / total return for a per-bar return series."""
    r = ret.dropna()
    if r.empty:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan, "total": np.nan}
    eq = (1.0 + r).cumprod()
    n_years = len(r) / PERIODS_PER_YEAR
    return {
        "cagr": eq.iloc[-1] ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan,
        "sharpe": r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR) if r.std() > 0 else np.nan,
        "maxdd": (eq / eq.cummax() - 1.0).min(),
        "total": eq.iloc[-1] - 1.0,
    }


def run_one(ticker: str, args) -> tuple[dict, pd.Series, pd.Series] | None:
    """Backtest one name; return (row, strategy_ret, market_ret) or None on failure."""
    try:
        raw = dc.load_data(ticker, args.start, args.end)
    except Exception as exc:  # noqa: BLE001 - want to skip and continue
        print(f"  ! {ticker}: skipped ({exc})")
        return None
    if len(raw) < args.entry + 30:
        print(f"  ! {ticker}: skipped (only {len(raw)} bars)")
        return None

    if args.engine == "atr":
        p = at.AtrParams(entry=args.entry, exit=args.exit, cost_bps=args.cost_bps,
                         risk_frac=args.risk_frac, max_weight=args.max_weight)
        d = at.backtest_atr(at.prepare(raw, p), p).data
    else:
        df = dc.donchian_channels(raw, args.entry, args.exit)
        df["position"] = dc.generate_positions(df)
        d = dc.backtest(df, cost_bps=args.cost_bps).data

    strat = d["strategy_ret"]
    market = d["market_ret"]
    s = curve_stats(strat)
    b = curve_stats(market)
    row = {
        "ticker": ticker,
        "type": UNIVERSE.get(ticker, "?"),
        "strat_cagr": s["cagr"], "bh_cagr": b["cagr"],
        "strat_sharpe": s["sharpe"], "bh_sharpe": b["sharpe"],
        "strat_maxdd": s["maxdd"], "bh_maxdd": b["maxdd"],
        "sharpe_edge": s["sharpe"] - b["sharpe"],
        "dd_reduction": s["maxdd"] - b["maxdd"],   # positive = shallower DD
    }
    return row, strat.rename(ticker), market.rename(ticker)


def print_report(res: pd.DataFrame, port: dict, bh_port: dict, args) -> None:
    show = res.copy().sort_values("sharpe_edge", ascending=False)
    fmt = show.copy()
    for c in ("strat_cagr", "bh_cagr", "strat_maxdd", "bh_maxdd", "dd_reduction"):
        fmt[c] = (fmt[c] * 100).round(1)
    for c in ("strat_sharpe", "bh_sharpe", "sharpe_edge"):
        fmt[c] = fmt[c].round(2)
    fmt = fmt.rename(columns={
        "strat_cagr": "S_cagr%", "bh_cagr": "BH_cagr%",
        "strat_sharpe": "S_shrp", "bh_sharpe": "BH_shrp",
        "strat_maxdd": "S_DD%", "bh_maxdd": "BH_DD%",
        "sharpe_edge": "shrp_edge", "dd_reduction": "DD_red%"})

    pd.set_option("display.width", 200)
    print(f"\nPer-name results — default {args.entry}/{args.exit} "
          f"({args.engine} engine), strategy vs buy & hold:")
    print(fmt[["ticker", "type", "S_shrp", "BH_shrp", "shrp_edge",
               "S_DD%", "BH_DD%", "DD_red%", "S_cagr%", "BH_cagr%"]].to_string(index=False))

    n = len(res)
    beat_sharpe = (res["strat_sharpe"] > res["bh_sharpe"]).sum()
    beat_ret = (res["strat_cagr"] > res["bh_cagr"]).sum()
    shallower = (res["strat_maxdd"] > res["bh_maxdd"]).sum()

    print("\n" + "=" * 70)
    print(f"  CROSS-SECTIONAL VERDICT  ({n} names)")
    print("=" * 70)
    print(f"  Median Sharpe   : strategy {res['strat_sharpe'].median():.2f}"
          f"   vs   buy & hold {res['bh_sharpe'].median():.2f}")
    print(f"  Median max DD   : strategy {res['strat_maxdd'].median()*100:.1f}%"
          f"  vs   buy & hold {res['bh_maxdd'].median()*100:.1f}%")
    print(f"  Beats B&H Sharpe: {beat_sharpe}/{n} names ({beat_sharpe/n*100:.0f}%)")
    print(f"  Beats B&H CAGR  : {beat_ret}/{n} names ({beat_ret/n*100:.0f}%)")
    print(f"  Shallower max DD: {shallower}/{n} names ({shallower/n*100:.0f}%)")
    print("-" * 70)
    print(f"  Equal-weight BASKET portfolio (daily-rebalanced):")
    print(f"    Strategy  : Sharpe {port['sharpe']:.2f}  CAGR {port['cagr']*100:5.1f}%"
          f"  maxDD {port['maxdd']*100:6.1f}%")
    print(f"    Buy & hold: Sharpe {bh_port['sharpe']:.2f}  CAGR {bh_port['cagr']*100:5.1f}%"
          f"  maxDD {bh_port['maxdd']*100:6.1f}%")
    print("=" * 70 + "\n")


def plot_basket(res: pd.DataFrame, port_ret, bh_ret, args, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # (1) per-name Sharpe scatter: strategy vs buy & hold.
    x, y = res["bh_sharpe"], res["strat_sharpe"]
    ax1.scatter(x, y, s=45, color="tab:blue", zorder=3)
    for _, r in res.iterrows():
        ax1.annotate(r["ticker"], (r["bh_sharpe"], r["strat_sharpe"]),
                     fontsize=8, xytext=(3, 3), textcoords="offset points")
    lo = float(min(x.min(), y.min())) - 0.2
    hi = float(max(x.max(), y.max())) + 0.2
    ax1.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1)
    ax1.fill_between([lo, hi], [lo, hi], hi, color="tab:green", alpha=0.06)
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Buy & hold Sharpe")
    ax1.set_ylabel("Strategy Sharpe")
    ax1.set_title("Per-name Sharpe (above line = strategy wins)")
    ax1.grid(alpha=0.3)

    # (2) equal-weight basket equity curves.
    ax2.plot((1 + port_ret).cumprod(), color="tab:blue", lw=1.5, label="Strategy basket")
    ax2.plot((1 + bh_ret).cumprod(), color="gray", lw=1.2, ls="--", label="Buy & hold basket")
    ax2.set_yscale("log")
    ax2.set_ylabel("Growth of $1 (log)")
    ax2.set_title(f"Equal-weight basket ({len(res)} names, {args.engine})")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle(f"Donchian {args.entry}/{args.exit} — cross-sectional test", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Chart saved to {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Donchian cross-sectional basket test.")
    p.add_argument("--tickers", default=None,
                   help="Comma-separated; defaults to the built-in universe.")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--entry", type=int, default=20)
    p.add_argument("--exit", type=int, default=10)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--engine", default="base", choices=["base", "atr"])
    p.add_argument("--risk-frac", type=float, default=0.02, help="ATR engine only.")
    p.add_argument("--max-weight", type=float, default=1.0, help="ATR engine only.")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-path", default="basket.png")
    p.add_argument("--csv", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = ([t.strip().upper() for t in args.tickers.split(",")]
               if args.tickers else list(UNIVERSE))

    print(f"Testing {len(tickers)} names with default {args.entry}/{args.exit} "
          f"({args.engine} engine)...")
    rows, strat_cols, mkt_cols = [], [], []
    for t in tickers:
        out = run_one(t, args)
        if out is None:
            continue
        row, strat, market = out
        rows.append(row)
        strat_cols.append(strat)
        mkt_cols.append(market)

    if not rows:
        raise SystemExit("No tickers succeeded.")
    res = pd.DataFrame(rows)

    # Equal-weight, daily-rebalanced basket: mean across whatever names are live.
    strat_mat = pd.concat(strat_cols, axis=1)
    mkt_mat = pd.concat(mkt_cols, axis=1)
    port_ret = strat_mat.mean(axis=1)
    bh_ret = mkt_mat.mean(axis=1)

    print_report(res, curve_stats(port_ret), curve_stats(bh_ret), args)

    if args.csv:
        res.to_csv(args.csv, index=False)
        print(f"Full results written to {args.csv}")
    if args.plot:
        plot_basket(res, port_ret, bh_ret, args, args.plot_path)


if __name__ == "__main__":
    main()
