"""
Walk-forward test for the Donchian breakout strategy.

The parameter sweep (sweep.py) optimises over the *whole* history, so its
"best" parameters are fitted with hindsight. Walk-forward analysis removes that
hindsight:

    1. Split history into consecutive folds.
    2. On each fold's IN-SAMPLE (train) window, pick the best parameters.
    3. Trade those parameters on the following OUT-OF-SAMPLE (test) window,
       which the optimiser never saw.
    4. Stitch the OOS windows together into one continuous equity curve.

If the stitched OOS curve is decent, the edge is more likely real. If it falls
apart versus a fixed default and versus buy & hold, the sweep was curve-fitting.

Usage:
    python walkforward.py --ticker TSLA --train 3 --test 1 --plot
    python walkforward.py --anchored --select cagr
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import donchian as dc

DEFAULT_ENTRIES = [10, 15, 20, 25, 30, 40, 55]
DEFAULT_EXITS = [5, 10, 15, 20, 25]
PERIODS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Metrics on a raw per-bar return series
# --------------------------------------------------------------------------- #
def series_metrics(returns: pd.Series) -> dict:
    """Summary stats for a per-bar strategy return series."""
    r = returns.dropna()
    if r.empty:
        return {"total_return": np.nan, "cagr": np.nan, "sharpe": np.nan,
                "max_drawdown": np.nan, "ann_vol": np.nan}
    eq = (1.0 + r).cumprod()
    n_years = len(r) / PERIODS_PER_YEAR
    total = eq.iloc[-1] - 1.0
    cagr = eq.iloc[-1] ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
    sharpe = (r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR)
              if r.std() > 0 else np.nan)
    dd = (eq / eq.cummax() - 1.0).min()
    vol = r.std() * np.sqrt(PERIODS_PER_YEAR)
    return {"total_return": total, "cagr": cagr, "sharpe": sharpe,
            "max_drawdown": dd, "ann_vol": vol}


# --------------------------------------------------------------------------- #
# Precompute per-parameter return streams (once) to keep folds cheap
# --------------------------------------------------------------------------- #
def precompute_returns(
    base: pd.DataFrame, entries: list[int], exits: list[int], cost_bps: float
) -> tuple[dict, pd.Series]:
    """Return {(entry, exit): strategy_ret series} and the market (B&H) series.

    Channels and positions only ever use *past* bars, so computing each stream
    over the full history and later slicing an OOS window introduces no
    look-ahead: the position entering a window reflects only prior prices.
    """
    streams: dict[tuple[int, int], pd.Series] = {}
    market: pd.Series | None = None
    for entry in entries:
        for exit_ in exits:
            if exit_ >= entry:
                continue
            df = dc.donchian_channels(base, entry, exit_)
            df["position"] = dc.generate_positions(df)
            res = dc.backtest(df, cost_bps=cost_bps)
            streams[(entry, exit_)] = res.data["strategy_ret"]
            if market is None:
                market = res.data["market_ret"]
    return streams, market


# --------------------------------------------------------------------------- #
# Fold construction
# --------------------------------------------------------------------------- #
def make_folds(
    index: pd.DatetimeIndex, train_years: int, test_years: int, anchored: bool
) -> list[dict]:
    """Build rolling (or anchored/expanding) train/test date boundaries."""
    start, end = index[0], index[-1]
    folds = []
    test_start = start + pd.DateOffset(years=train_years)
    while test_start < end:
        test_end = min(test_start + pd.DateOffset(years=test_years), end + pd.Timedelta(days=1))
        train_start = start if anchored else test_start - pd.DateOffset(years=train_years)
        folds.append({
            "train_start": train_start,
            "train_end": test_start,   # exclusive
            "test_start": test_start,
            "test_end": test_end,      # exclusive
        })
        test_start = test_start + pd.DateOffset(years=test_years)
    return folds


def _slice(s: pd.Series, lo, hi) -> pd.Series:
    """Half-open slice [lo, hi) on a DatetimeIndex series."""
    return s[(s.index >= lo) & (s.index < hi)]


# --------------------------------------------------------------------------- #
# Walk-forward
# --------------------------------------------------------------------------- #
def walk_forward(
    streams: dict,
    market: pd.Series,
    folds: list[dict],
    select: str,
    min_trades: int,
    default_params: tuple[int, int],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Run the walk-forward loop.

    Returns:
        summary  : per-fold table (chosen params + OOS metrics)
        oos_ret  : stitched out-of-sample strategy returns
        def_ret  : same OOS windows using the fixed default parameters
        mkt_ret  : buy & hold returns over the OOS span
    """
    rows, oos_parts, def_parts, mkt_parts = [], [], [], []

    for k, fold in enumerate(folds, 1):
        # --- pick best params on the training window ---
        best_key, best_score = None, -np.inf
        for key, stream in streams.items():
            train_r = _slice(stream, fold["train_start"], fold["train_end"])
            # Count position changes as a proxy for round-trips in-sample.
            n_moves = int((train_r != 0).astype(int).diff().abs().fillna(0).sum())
            if n_moves < min_trades:
                continue
            m = series_metrics(train_r)
            score = m[select]
            # For drawdown, "higher" (closer to 0) is better -> already handled
            # because max_drawdown is negative and we maximise it.
            if score is not None and not np.isnan(score) and score > best_score:
                best_score, best_key = score, key
        if best_key is None:
            best_key = default_params  # nothing qualified -> fall back

        # --- apply on the out-of-sample window ---
        oos_r = _slice(streams[best_key], fold["test_start"], fold["test_end"])
        def_r = _slice(streams[default_params], fold["test_start"], fold["test_end"])
        mkt_r = _slice(market, fold["test_start"], fold["test_end"])
        if oos_r.empty:
            continue
        oos_parts.append(oos_r)
        def_parts.append(def_r)
        mkt_parts.append(mkt_r)

        m = series_metrics(oos_r)
        rows.append({
            "fold": k,
            "train": f"{fold['train_start'].date()}→{fold['train_end'].date()}",
            "test": f"{fold['test_start'].date()}→{oos_r.index[-1].date()}",
            "entry": best_key[0],
            "exit": best_key[1],
            "oos_return": m["total_return"],
            "oos_cagr": m["cagr"],
            "oos_sharpe": m["sharpe"],
            "oos_maxDD": m["max_drawdown"],
            "bh_return": series_metrics(mkt_r)["total_return"],
        })

    summary = pd.DataFrame(rows)
    oos_ret = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    def_ret = pd.concat(def_parts) if def_parts else pd.Series(dtype=float)
    mkt_ret = pd.concat(mkt_parts) if mkt_parts else pd.Series(dtype=float)
    return summary, oos_ret, def_ret, mkt_ret


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(summary: pd.DataFrame, oos, deff, mkt, select: str) -> None:
    show = summary.copy()
    for c in ("oos_return", "oos_cagr", "oos_maxDD", "bh_return"):
        show[c] = (show[c] * 100).round(1)
    show["oos_sharpe"] = show["oos_sharpe"].round(2)
    show = show.rename(columns={
        "oos_return": "oos_ret_%", "oos_cagr": "oos_cagr_%",
        "oos_maxDD": "oos_maxDD_%", "bh_return": "bh_ret_%"})

    pd.set_option("display.width", 200)
    print(f"\nPer-fold results (params chosen in-sample by {select}, "
          f"traded out-of-sample):")
    print(show.to_string(index=False))

    wf, wf_def, wf_bh = (series_metrics(oos), series_metrics(deff),
                         series_metrics(mkt))
    beat = (summary["oos_return"] > summary["bh_return"]).mean()

    def line(name, m):
        return (f"  {name:<26}: return {m['total_return']*100:9.1f}%   "
                f"CAGR {m['cagr']*100:6.1f}%   Sharpe {m['sharpe']:5.2f}   "
                f"maxDD {m['max_drawdown']*100:6.1f}%")

    print("\n" + "=" * 74)
    print("  STITCHED OUT-OF-SAMPLE PERFORMANCE (the honest number)")
    print("=" * 74)
    print(line("Walk-forward (adaptive)", wf))
    print(line("Fixed default 20/10", wf_def))
    print(line("Buy & hold", wf_bh))
    print("-" * 74)
    print(f"  Folds where WF beat B&H : {beat*100:.0f}%  ({int(beat*len(summary))}/{len(summary)})")
    most = summary.groupby(['entry', 'exit']).size().sort_values(ascending=False)
    print(f"  Most-chosen params      : "
          f"{most.index[0][0]}/{most.index[0][1]}  (picked {most.iloc[0]}/{len(summary)} folds)")
    print("=" * 74 + "\n")


def plot_wf(oos, deff, mkt, ticker: str, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6))
    for r, label, color, ls in [
        (oos, "Walk-forward (adaptive)", "tab:blue", "-"),
        (deff, "Fixed default 20/10", "tab:orange", "-"),
        (mkt, "Buy & hold", "gray", "--"),
    ]:
        eq = (1.0 + r).cumprod()
        ax.plot(eq.index, eq, label=label, color=color, ls=ls, lw=1.4)
    ax.set_yscale("log")
    ax.set_title(f"{ticker} — walk-forward out-of-sample (log scale)")
    ax.set_ylabel("Growth of $1 (log)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Chart saved to {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Donchian walk-forward test.")
    p.add_argument("--ticker", default="TSLA")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--train", type=int, default=3, help="Train window (years).")
    p.add_argument("--test", type=int, default=1, help="Test window (years).")
    p.add_argument("--anchored", action="store_true",
                   help="Expanding train window instead of rolling.")
    p.add_argument("--select", default="sharpe",
                   choices=["sharpe", "cagr", "total_return", "max_drawdown"],
                   help="In-sample metric used to pick parameters.")
    p.add_argument("--min-trades", type=int, default=3,
                   help="Skip params with fewer in-sample position changes.")
    p.add_argument("--entries", type=_int_list, default=DEFAULT_ENTRIES)
    p.add_argument("--exits", type=_int_list, default=DEFAULT_EXITS)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-path", default="walkforward.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = dc.load_data(args.ticker, args.start, args.end)
    streams, market = precompute_returns(base, args.entries, args.exits, args.cost_bps)
    folds = make_folds(base.index, args.train, args.test, args.anchored)
    if not folds:
        raise SystemExit("Not enough history for even one fold — shorten --train.")

    print(f"\n{args.ticker}: {len(folds)} folds "
          f"({'anchored/expanding' if args.anchored else 'rolling'} "
          f"{args.train}y train / {args.test}y test), "
          f"{len(streams)} candidate parameter sets")

    summary, oos, deff, mkt = walk_forward(
        streams, market, folds, args.select, args.min_trades, (20, 10))
    print_report(summary, oos, deff, mkt, args.select)

    if args.plot:
        plot_wf(oos, deff, mkt, args.ticker, args.plot_path)


if __name__ == "__main__":
    main()
