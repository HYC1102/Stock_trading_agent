"""
Single-stock Donchian channel breakout trading strategy.

Strategy (long-only, "Turtle"-style):
    - Entry channel : highest high / lowest low over `entry_window` days.
    - Exit channel  : highest high / lowest low over `exit_window` days.
    - Go long  when the close breaks ABOVE the prior entry-window high.
    - Go flat  when the close breaks BELOW the prior exit-window low.

The backtest is long-or-flat only (no shorting, no leverage). Signals are
generated on the close and executed on the NEXT bar's open to avoid
look-ahead bias.

Usage:
    python donchian.py --ticker TSLA --start 2015-01-01
    python donchian.py --ticker TSLA --entry 20 --exit 10 --plot
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download OHLCV data and return a clean single-index DataFrame."""
    df = yf.download(
        ticker, start=start, end=end, progress=False, auto_adjust=True
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker!r}. Check ticker/dates.")

    # yfinance returns MultiIndex columns for a single ticker; flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index.name = "Date"
    return df


# --------------------------------------------------------------------------- #
# Indicator
# --------------------------------------------------------------------------- #
def donchian_channels(
    df: pd.DataFrame, entry_window: int, exit_window: int
) -> pd.DataFrame:
    """Add Donchian channel columns.

    We use the channel computed over the bars *prior* to the current one
    (``.shift(1)``) so that a close making a new high is compared against the
    previous window's extreme rather than including itself.
    """
    out = df.copy()
    out["entry_upper"] = df["High"].rolling(entry_window).max().shift(1)
    out["exit_lower"] = df["Low"].rolling(exit_window).min().shift(1)
    # Kept for plotting / context.
    out["entry_lower"] = df["Low"].rolling(entry_window).min().shift(1)
    out["entry_mid"] = (out["entry_upper"] + out["entry_lower"]) / 2.0
    return out


def atr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Average True Range using Wilder's smoothing (the Turtle "N").

    True range is the greatest of: today's high-low, |high - prev close|, and
    |low - prev close|. Wilder's running average is an EWMA with alpha = 1/window.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


# --------------------------------------------------------------------------- #
# Signals & positions
# --------------------------------------------------------------------------- #
def generate_positions(df: pd.DataFrame) -> pd.Series:
    """Return a position series (1 = long, 0 = flat), executed next-open.

    A breakout above ``entry_upper`` turns the position on; a breakdown below
    ``exit_lower`` turns it off. The state is carried forward between signals.
    """
    close = df["Close"]
    long_signal = close > df["entry_upper"]
    exit_signal = close < df["exit_lower"]

    # Build the target state bar-by-bar (state machine).
    state = np.zeros(len(df), dtype=int)
    pos = 0
    long_arr = long_signal.to_numpy()
    exit_arr = exit_signal.to_numpy()
    for i in range(len(df)):
        if pos == 0 and long_arr[i]:
            pos = 1
        elif pos == 1 and exit_arr[i]:
            pos = 0
        state[i] = pos

    position = pd.Series(state, index=df.index, name="position")
    # Execute on the next bar's open -> shift the decided position forward one bar.
    return position.shift(1).fillna(0).astype(int)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    data: pd.DataFrame          # per-bar frame incl. equity curves
    trades: pd.DataFrame        # one row per round-trip trade
    metrics: dict               # summary statistics


def backtest(
    df: pd.DataFrame, cost_bps: float = 2.0, periods_per_year: int = 252
) -> BacktestResult:
    """Run the long/flat backtest.

    Returns are computed on open-to-open moves while in a position, since we
    enter/exit at the open. `cost_bps` is charged (in bps of notional) on each
    entry and each exit.
    """
    d = df.copy()
    position = d["position"]

    # Open-to-open return realised while holding the position set for that bar.
    open_ret = d["Open"].pct_change().fillna(0.0)
    d["market_ret"] = open_ret
    gross = position * open_ret

    # Transaction costs charged when the position changes.
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_bps / 10_000.0)
    d["strategy_ret"] = gross - cost

    d["equity"] = (1.0 + d["strategy_ret"]).cumprod()
    d["buy_hold"] = (1.0 + d["market_ret"]).cumprod()

    trades = _extract_trades(d)
    metrics = _compute_metrics(d, trades, periods_per_year)
    return BacktestResult(data=d, trades=trades, metrics=metrics)


def _extract_trades(d: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct round-trip trades from the position series.

    A trade is a contiguous run of long bars: it opens at the open price of
    the first long bar and closes at the open price of the bar where the
    position returns to flat. A run still long on the final bar is marked
    ``open`` and closed at the last available open price.
    """
    pos = d["position"].to_numpy()
    opens = d["Open"].to_numpy()
    dates = d.index
    n = len(pos)

    trades = []
    entry_i = None
    for i in range(n):
        if pos[i] == 1 and entry_i is None:          # run starts
            entry_i = i
        # Close the run when we go flat, or when the series ends while long.
        closing = entry_i is not None and (pos[i] == 0 or i == n - 1)
        if closing:
            still_open = pos[i] == 1  # only true when the series ends long
            exit_i = i
            trades.append(
                {
                    "entry_date": dates[entry_i],
                    "exit_date": dates[exit_i],
                    "entry_price": opens[entry_i],
                    "exit_price": opens[exit_i],
                    "return_pct": (opens[exit_i] / opens[entry_i] - 1.0) * 100.0,
                    "bars_held": exit_i - entry_i,
                    "open": still_open,
                }
            )
            entry_i = None

    return pd.DataFrame(trades)


def _compute_metrics(
    d: pd.DataFrame, trades: pd.DataFrame, periods_per_year: int
) -> dict:
    strat = d["strategy_ret"]
    eq = d["equity"]

    n_years = len(d) / periods_per_year
    total_return = eq.iloc[-1] - 1.0
    bh_total_return = d["buy_hold"].iloc[-1] - 1.0
    cagr = eq.iloc[-1] ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan

    vol = strat.std() * np.sqrt(periods_per_year)
    sharpe = (
        strat.mean() / strat.std() * np.sqrt(periods_per_year)
        if strat.std() > 0
        else np.nan
    )

    running_max = eq.cummax()
    drawdown = eq / running_max - 1.0
    max_dd = drawdown.min()

    if not trades.empty:
        wins = trades["return_pct"] > 0
        win_rate = wins.mean()
        # Store as fractions so the pct() formatter (which ×100s) is consistent.
        avg_win = trades.loc[wins, "return_pct"].mean() / 100.0
        avg_loss = trades.loc[~wins, "return_pct"].mean() / 100.0
        n_trades = len(trades)
    else:
        win_rate = avg_win = avg_loss = np.nan
        n_trades = 0

    time_in_market = (d["position"] != 0).mean()

    return {
        "start": d.index[0].date().isoformat(),
        "end": d.index[-1].date().isoformat(),
        "years": round(n_years, 2),
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "time_in_market": time_in_market,
        "buy_hold_return": bh_total_return,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(ticker: str, params: dict, m: dict) -> None:
    def pct(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:6.2f}%"

    print("\n" + "=" * 52)
    print(f"  Donchian breakout backtest — {ticker}")
    print("=" * 52)
    print(f"  Period            : {m['start']} -> {m['end']}  ({m['years']} yrs)")
    print(f"  Entry / Exit win  : {params['entry']} / {params['exit']} days")
    print(f"  Cost per side     : {params['cost_bps']:.1f} bps")
    print("-" * 52)
    print(f"  Total return      : {pct(m['total_return'])}")
    print(f"  CAGR              : {pct(m['cagr'])}")
    print(f"  Ann. volatility   : {pct(m['ann_vol'])}")
    print(f"  Sharpe ratio      : {m['sharpe']:6.2f}")
    print(f"  Max drawdown      : {pct(m['max_drawdown'])}")
    print(f"  Time in market    : {pct(m['time_in_market'])}")
    print("-" * 52)
    print(f"  Trades            : {m['n_trades']}")
    print(f"  Win rate          : {pct(m['win_rate'])}")
    print(f"  Avg win / loss    : {pct(m['avg_win_pct'])} / {pct(m['avg_loss_pct'])}")
    print("-" * 52)
    print(f"  Buy & hold return : {pct(m['buy_hold_return'])}")
    print("=" * 52 + "\n")


def plot_result(ticker: str, res: BacktestResult, path: str) -> None:
    import matplotlib.pyplot as plt

    d = res.data
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(d.index, d["Close"], color="black", lw=1.0, label="Close")
    ax1.plot(d.index, d["entry_upper"], color="tab:green", lw=0.9, ls="--",
             label="Entry upper")
    ax1.plot(d.index, d["exit_lower"], color="tab:red", lw=0.9, ls="--",
             label="Exit lower")

    for _, t in res.trades.iterrows():
        ax1.scatter(t["entry_date"], t["entry_price"], marker="^",
                    color="green", s=70, zorder=5)
        if not t["open"]:
            ax1.scatter(t["exit_date"], t["exit_price"], marker="v",
                        color="red", s=70, zorder=5)

    ax1.set_title(f"{ticker} — Donchian breakout")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(d.index, d["equity"], color="tab:blue", lw=1.3, label="Strategy")
    ax2.plot(d.index, d["buy_hold"], color="gray", lw=1.0, ls="--",
             label="Buy & hold")
    ax2.set_ylabel("Growth of $1")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Chart saved to {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Donchian channel breakout backtest.")
    p.add_argument("--ticker", default="TSLA")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--entry", type=int, default=20, help="Entry channel window.")
    p.add_argument("--exit", type=int, default=10, help="Exit channel window.")
    p.add_argument("--cost-bps", type=float, default=2.0,
                   help="Per-side transaction cost in basis points.")
    p.add_argument("--plot", action="store_true", help="Save a chart.")
    p.add_argument("--plot-path", default="backtest.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.ticker, args.start, args.end)
    df = donchian_channels(df, args.entry, args.exit)
    df["position"] = generate_positions(df)

    res = backtest(df, cost_bps=args.cost_bps)
    params = {"entry": args.entry, "exit": args.exit, "cost_bps": args.cost_bps}
    print_report(args.ticker, params, res.metrics)

    if not res.trades.empty:
        show = res.trades.copy()
        show["entry_date"] = show["entry_date"].dt.date
        show["exit_date"] = show["exit_date"].dt.date
        print("Last 5 trades:")
        print(show.tail(5).to_string(index=False))

    if args.plot:
        plot_result(args.ticker, res, args.plot_path)


if __name__ == "__main__":
    main()
