"""
Parameter sweep for the Donchian breakout strategy.

Grids over entry/exit channel windows, backtests each combination on the same
price history, and reports the results as a ranked table plus heatmaps. Useful
for gauging how sensitive performance is to the parameters (i.e. how much of
the headline result is luck vs. a broad robust region).

Usage:
    python sweep.py --ticker TSLA --start 2015-01-01
    python sweep.py --ticker TSLA --metric cagr --plot
    python sweep.py --entries 10,20,30,40,55 --exits 5,10,15,20
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import donchian as dc


DEFAULT_ENTRIES = [10, 15, 20, 25, 30, 40, 55]
DEFAULT_EXITS = [5, 10, 15, 20, 25]

# Metric -> (nice label, whether higher is better)
METRIC_INFO = {
    "sharpe": ("Sharpe", True),
    "cagr": ("CAGR", True),
    "total_return": ("Total return", True),
    "max_drawdown": ("Max drawdown", True),   # less negative is better
    "ann_vol": ("Ann. vol", False),
    "n_trades": ("# trades", None),
}


def run_sweep(
    base: pd.DataFrame,
    entries: list[int],
    exits: list[int],
    cost_bps: float,
) -> pd.DataFrame:
    """Backtest every (entry, exit) pair; return one row of metrics per pair.

    ``base`` is the raw OHLC frame (channels are recomputed per combination).
    ``exit`` windows >= their ``entry`` window are skipped: an exit channel
    wider than the entry channel would trigger the exit before an entry can
    survive, which isn't a meaningful configuration.
    """
    rows = []
    for entry in entries:
        for exit_ in exits:
            if exit_ >= entry:
                continue
            df = dc.donchian_channels(base, entry, exit_)
            df["position"] = dc.generate_positions(df)
            m = dc.backtest(df, cost_bps=cost_bps).metrics
            rows.append(
                {
                    "entry": entry,
                    "exit": exit_,
                    "total_return": m["total_return"],
                    "cagr": m["cagr"],
                    "sharpe": m["sharpe"],
                    "ann_vol": m["ann_vol"],
                    "max_drawdown": m["max_drawdown"],
                    "n_trades": m["n_trades"],
                    "win_rate": m["win_rate"],
                    "time_in_market": m["time_in_market"],
                }
            )
    return pd.DataFrame(rows)


def print_table(results: pd.DataFrame, metric: str, top: int = 15) -> None:
    label, higher_better = METRIC_INFO[metric][0], METRIC_INFO[metric][1]
    ranked = results.sort_values(
        metric, ascending=not (higher_better or higher_better is None)
    )

    show = ranked.head(top).copy()
    show["total_return"] = (show["total_return"] * 100).round(0).astype(int).astype(str) + "%"
    show["cagr"] = (show["cagr"] * 100).round(1)
    show["sharpe"] = show["sharpe"].round(2)
    show["ann_vol"] = (show["ann_vol"] * 100).round(1)
    show["max_drawdown"] = (show["max_drawdown"] * 100).round(1)
    show["win_rate"] = (show["win_rate"] * 100).round(0)
    show["time_in_market"] = (show["time_in_market"] * 100).round(0)
    show = show.rename(
        columns={
            "cagr": "cagr_%",
            "ann_vol": "vol_%",
            "max_drawdown": "maxDD_%",
            "win_rate": "win_%",
            "time_in_market": "inMkt_%",
        }
    )

    pd.set_option("display.width", 160)
    print(f"\nTop {min(top, len(show))} combinations by {label}:")
    print(show.to_string(index=False))

    print(f"\nSweep summary ({len(results)} combinations):")
    for col in ("sharpe", "cagr", "max_drawdown"):
        vals = results[col]
        scale = 100 if col != "sharpe" else 1
        suffix = "" if col == "sharpe" else "%"
        print(
            f"  {METRIC_INFO[col][0]:<13}: "
            f"median {vals.median() * scale:6.2f}{suffix}  "
            f"best {vals.max() * scale:6.2f}{suffix}  "
            f"worst {vals.min() * scale:6.2f}{suffix}"
        )


def plot_heatmaps(
    results: pd.DataFrame, ticker: str, path: str, metrics: list[str]
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        pivot = results.pivot(index="entry", columns="exit", values=metric)
        data = pivot.to_numpy(dtype=float)

        label, higher_better = METRIC_INFO[metric][0], METRIC_INFO[metric][1]
        cmap = "RdYlGn" if higher_better is not False else "RdYlGn_r"
        im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Exit window")
        ax.set_ylabel("Entry window")
        ax.set_title(f"{label}")

        # Annotate each valid cell.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    continue
                txt = f"{v:.2f}" if metric in ("sharpe",) else f"{v * 100:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{ticker} — Donchian parameter sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nHeatmap saved to {path}")


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Donchian parameter sweep.")
    p.add_argument("--ticker", default="TSLA")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--entries", type=_int_list, default=DEFAULT_ENTRIES,
                   help="Comma-separated entry windows.")
    p.add_argument("--exits", type=_int_list, default=DEFAULT_EXITS,
                   help="Comma-separated exit windows.")
    p.add_argument("--metric", default="sharpe", choices=list(METRIC_INFO),
                   help="Metric used to rank the results table.")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--plot", action="store_true", help="Save heatmaps.")
    p.add_argument("--plot-path", default="sweep.png")
    p.add_argument("--csv", default=None, help="Optional path to dump full grid.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = dc.load_data(args.ticker, args.start, args.end)

    results = run_sweep(base, args.entries, args.exits, args.cost_bps)
    if results.empty:
        raise SystemExit("No valid (entry, exit) combinations — need exit < entry.")

    print(f"\n{args.ticker}: swept {len(results)} combinations "
          f"({results['entry'].nunique()} entry x {results['exit'].nunique()} exit windows)")
    print_table(results, args.metric, args.top)

    if args.csv:
        results.to_csv(args.csv, index=False)
        print(f"\nFull grid written to {args.csv}")

    if args.plot:
        plot_heatmaps(results, args.ticker, args.plot_path,
                      metrics=["sharpe", "cagr", "max_drawdown"])


if __name__ == "__main__":
    main()
