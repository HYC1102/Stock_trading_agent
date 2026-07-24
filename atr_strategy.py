"""
Donchian breakout with ATR-based risk control ("Turtle"-style).

Two additions on top of the plain Donchian breakout in donchian.py:

  1. Trailing ATR stop (chandelier exit)
     While long, track the highest close since entry. Exit if the close falls
     more than ``stop_mult`` * ATR below that high — a volatility-adaptive stop
     that tightens in calm markets and gives room in wild ones. The position
     still also exits on the usual Donchian exit-channel breakdown; whichever
     triggers first wins.

  2. Volatility-based position sizing (risk parity per trade)
     Size each position so that being stopped out costs a fixed fraction
     ``risk_frac`` of equity. The stop sits ``stop_mult`` * ATR away, i.e. a
     fractional move of stop_mult * ATR / price, so

         weight = risk_frac / (stop_mult * ATR / price)

     capped at ``max_weight`` (default 1.0 = no leverage). Volatile regimes get
     a smaller position, quiet regimes a larger one. Weight is fixed at entry.

As in the base engine, signals are decided on the close and executed on the
next bar's open (no look-ahead); costs are charged on weight turnover.

Usage:
    python atr_strategy.py --ticker TSLA --plot
    python atr_strategy.py --stop-mult 3 --risk-frac 0.02 --max-weight 1.5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

import donchian as dc


@dataclass
class AtrParams:
    entry: int = 20
    exit: int = 10
    atr_window: int = 20
    stop_mult: float = 2.0     # trailing stop distance, in ATRs (Turtle "2N")
    risk_frac: float = 0.01    # equity risked per trade if stopped out
    max_weight: float = 1.0    # position cap (1.0 = fully invested, no leverage)
    cost_bps: float = 2.0
    trend_ma: int = 0          # #4: only enter longs above this SMA (0 = off)
    max_units: int = 1         # #5: pyramid up to this many units (1 = no pyramiding)
    pyramid_atr: float = 0.5   # #5: add a unit every this-many ATRs of advance


# --------------------------------------------------------------------------- #
# State machine: target weights + trade log
# --------------------------------------------------------------------------- #
def build_weights(df: pd.DataFrame, p: AtrParams) -> tuple[pd.Series, list[dict]]:
    """Walk the bars, producing a target-weight series and a trade log.

    ``target_weight[i]`` is the weight decided at the close of bar ``i`` (held
    from the next bar's open). Options:

      * Trend filter (#4): if ``trend_ma`` > 0, only open a long when the close
        is above its ``trend_ma``-day SMA -- no counter-trend breakouts.
      * Pyramiding (#5): if ``max_units`` > 1, add a unit each time price
        advances another ``pyramid_atr`` * N (ATR fixed at initial entry, the
        Turtle "N"), up to ``max_units``. Each unit adds ``per_unit_w`` weight,
        capped at ``max_weight``. The whole stack exits together on the trailing
        stop or the Donchian exit.
    """
    close = df["Close"].to_numpy()
    upper = df["entry_upper"].to_numpy()
    lower = df["exit_lower"].to_numpy()
    atr = df["atr"].to_numpy()
    trend = df["trend_ma"].to_numpy()
    n = len(df)

    target = np.zeros(n)
    trades: list[dict] = []

    in_pos = False
    entry_i = 0
    entry_close = atr_entry = per_unit_w = next_add = highest = np.nan
    units = 0

    def ready(i: int) -> bool:
        return not (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(atr[i]))

    def trend_ok(i: int) -> bool:
        return p.trend_ma <= 0 or (not np.isnan(trend[i]) and close[i] > trend[i])

    for i in range(n):
        if not in_pos:
            if ready(i) and close[i] > upper[i] and trend_ok(i):
                atr_entry = atr[i]
                atr_frac = atr_entry / close[i]
                per_unit_w = (
                    min(p.max_weight, p.risk_frac / (p.stop_mult * atr_frac))
                    if atr_frac > 0 else p.max_weight
                )
                in_pos, entry_i, entry_close, highest = True, i, close[i], close[i]
                units = 1
                next_add = close[i] + p.pyramid_atr * atr_entry
                target[i] = min(p.max_weight, per_unit_w * units)
            # else stays flat (target already 0)
        else:
            highest = max(highest, close[i])
            # #5: add units as the trend extends (N fixed at entry).
            while p.max_units > 1 and units < p.max_units and close[i] >= next_add:
                units += 1
                next_add += p.pyramid_atr * atr_entry

            stop_level = highest - p.stop_mult * atr[i]
            hit_stop = close[i] < stop_level
            hit_don = (not np.isnan(lower[i])) and close[i] < lower[i]

            if hit_stop or hit_don:
                trades.append({
                    "entry_i": entry_i, "exit_i": i, "units": units,
                    "weight": min(p.max_weight, per_unit_w * units),
                    "reason": "stop" if hit_stop else "donchian",
                })
                in_pos, units = False, 0
                target[i] = 0.0
            else:
                target[i] = min(p.max_weight, per_unit_w * units)

    if in_pos:  # still open on the last bar
        trades.append({"entry_i": entry_i, "exit_i": n - 1, "units": units,
                       "weight": min(p.max_weight, per_unit_w * units),
                       "reason": "open"})

    return pd.Series(target, index=df.index, name="target_weight"), trades


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest_atr(df: pd.DataFrame, p: AtrParams) -> dc.BacktestResult:
    d = df.copy()
    target, raw_trades = build_weights(d, p)

    # Execute next open: the weight held on bar t was decided at t-1.
    held = target.shift(1).fillna(0.0)
    d["position"] = held

    open_ret = d["Open"].pct_change().fillna(0.0)
    d["market_ret"] = open_ret
    turnover = held.diff().abs().fillna(held.abs())
    d["strategy_ret"] = held * open_ret - turnover * (p.cost_bps / 10_000.0)
    d["equity"] = (1.0 + d["strategy_ret"]).cumprod()
    d["buy_hold"] = (1.0 + open_ret).cumprod()

    trades = _trade_frame(d, raw_trades)
    metrics = dc._compute_metrics(d, trades, periods_per_year=252)
    # Sizing-specific extras.
    active = d["position"][d["position"] > 0]
    metrics["avg_weight"] = active.mean() if not active.empty else np.nan
    if not trades.empty:
        metrics["pct_exit_stop"] = (trades["reason"] == "stop").mean()
        metrics["pct_exit_donchian"] = (trades["reason"] == "donchian").mean()
        metrics["avg_units"] = trades["units"].mean()
        metrics["max_units_hit"] = int(trades["units"].max())
    else:
        metrics["pct_exit_stop"] = metrics["pct_exit_donchian"] = np.nan
        metrics["avg_units"] = np.nan
        metrics["max_units_hit"] = 0
    return dc.BacktestResult(data=d, trades=trades, metrics=metrics)


def _trade_frame(d: pd.DataFrame, raw: list[dict]) -> pd.DataFrame:
    """Turn raw (entry_i, exit_i, ...) records into a dated trade log.

    Entry/exit fill at the *next* bar's open to match execution; a trade open on
    the final bar is marked to the last close.
    """
    opens = d["Open"].to_numpy()
    closes = d["Close"].to_numpy()
    dates = d.index
    n = len(d)
    rows = []
    for t in raw:
        ei, xi = t["entry_i"], t["exit_i"]
        entry_px = opens[ei + 1] if ei + 1 < n else closes[ei]
        still_open = t["reason"] == "open"
        exit_px = opens[xi + 1] if (not still_open and xi + 1 < n) else closes[xi]
        rows.append({
            "entry_date": dates[min(ei + 1, n - 1)],
            "exit_date": dates[xi],
            "entry_price": entry_px,
            "exit_price": exit_px,
            "weight": t["weight"],
            "units": t.get("units", 1),
            "return_pct": (exit_px / entry_px - 1.0) * 100.0,
            "reason": t["reason"],
            "open": still_open,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(ticker: str, p: AtrParams, m: dict, base_m: dict) -> None:
    def pct(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:7.2f}%"

    print("\n" + "=" * 60)
    print(f"  Donchian + ATR stop/sizing — {ticker}")
    print("=" * 60)
    print(f"  Period            : {m['start']} -> {m['end']}  ({m['years']} yrs)")
    print(f"  Entry/Exit/ATR    : {p.entry} / {p.exit} / {p.atr_window}")
    print(f"  Stop / risk / cap : {p.stop_mult:.1f}N  /  {p.risk_frac*100:.1f}%  /  {p.max_weight:.2f}x")
    trend_txt = f"{p.trend_ma}d SMA" if p.trend_ma > 0 else "off"
    pyr_txt = (f"up to {p.max_units} units @ {p.pyramid_atr:.2f}N"
               if p.max_units > 1 else "off")
    print(f"  Trend filter (#4) : {trend_txt}")
    print(f"  Pyramiding   (#5) : {pyr_txt}")
    print("-" * 60)
    print(f"  {'metric':<18}{'ATR sized':>14}{'base all-in':>16}")
    print("-" * 60)
    rows = [
        ("Total return", "total_return"),
        ("CAGR", "cagr"),
        ("Ann. volatility", "ann_vol"),
        ("Max drawdown", "max_drawdown"),
        ("Time in market", "time_in_market"),
        ("Win rate", "win_rate"),
    ]
    for label, key in rows:
        print(f"  {label:<18}{pct(m[key]):>14}{pct(base_m[key]):>16}")
    print(f"  {'Sharpe':<18}{m['sharpe']:>14.2f}{base_m['sharpe']:>16.2f}")
    print(f"  {'# trades':<18}{m['n_trades']:>14}{base_m['n_trades']:>16}")
    print("-" * 60)
    print(f"  Avg position weight : {pct(m['avg_weight'])}")
    if p.max_units > 1:
        print(f"  Avg / max units     : {m['avg_units']:.2f} / {m['max_units_hit']}")
    print(f"  Exits via ATR stop  : {pct(m['pct_exit_stop'])}")
    print(f"  Exits via Donchian  : {pct(m['pct_exit_donchian'])}")
    print(f"  Buy & hold return   : {pct(m['buy_hold_return'])}")
    print("=" * 60 + "\n")


def plot_result(ticker: str, res: dc.BacktestResult, base: dc.BacktestResult,
                path: str) -> None:
    import matplotlib.pyplot as plt

    d = res.data
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(d.index, d["equity"], color="tab:blue", lw=1.4, label="ATR sized")
    ax1.plot(base.data.index, base.data["equity"], color="tab:orange", lw=1.2,
             label="Base all-in")
    ax1.plot(d.index, d["buy_hold"], color="gray", lw=1.0, ls="--", label="Buy & hold")
    ax1.set_yscale("log")
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title(f"{ticker} — Donchian + ATR stop/sizing")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3, which="both")

    ax2.fill_between(d.index, 0, d["position"], color="tab:blue", alpha=0.4, step="pre")
    ax2.set_ylabel("Position weight")
    ax2.set_ylim(0, max(1.0, d["position"].max() * 1.1))
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Chart saved to {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Donchian + ATR stop/sizing backtest.")
    p.add_argument("--ticker", default="TSLA")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--entry", type=int, default=20)
    p.add_argument("--exit", type=int, default=10)
    p.add_argument("--atr-window", type=int, default=20)
    p.add_argument("--stop-mult", type=float, default=2.0,
                   help="Trailing stop distance in ATRs.")
    p.add_argument("--risk-frac", type=float, default=0.01,
                   help="Fraction of equity risked per trade.")
    p.add_argument("--max-weight", type=float, default=1.0,
                   help="Position cap (1.0 = no leverage).")
    p.add_argument("--trend-ma", type=int, default=0,
                   help="#4: only enter longs above this SMA (0 = off, e.g. 200).")
    p.add_argument("--max-units", type=int, default=1,
                   help="#5: pyramid up to this many units (1 = off, Turtle = 4).")
    p.add_argument("--pyramid-atr", type=float, default=0.5,
                   help="#5: add a unit every this-many ATRs of advance.")
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-path", default="atr_strategy.png")
    return p.parse_args()


def prepare(df: pd.DataFrame, p: AtrParams) -> pd.DataFrame:
    out = dc.donchian_channels(df, p.entry, p.exit)
    out["atr"] = dc.atr(df, p.atr_window)
    out["trend_ma"] = (
        df["Close"].rolling(p.trend_ma).mean() if p.trend_ma > 0
        else pd.Series(np.nan, index=df.index)
    )
    return out


def main() -> None:
    args = parse_args()
    p = AtrParams(
        entry=args.entry, exit=args.exit, atr_window=args.atr_window,
        stop_mult=args.stop_mult, risk_frac=args.risk_frac,
        max_weight=args.max_weight, cost_bps=args.cost_bps,
        trend_ma=args.trend_ma, max_units=args.max_units,
        pyramid_atr=args.pyramid_atr,
    )
    raw = dc.load_data(args.ticker, args.start, args.end)
    df = prepare(raw, p)

    res = backtest_atr(df, p)

    # Base all-in Donchian for comparison, on the same data.
    base_df = dc.donchian_channels(raw, p.entry, p.exit)
    base_df["position"] = dc.generate_positions(base_df)
    base = dc.backtest(base_df, cost_bps=p.cost_bps)

    print_report(args.ticker, p, res.metrics, base.metrics)

    if not res.trades.empty:
        show = res.trades.copy()
        show["entry_date"] = show["entry_date"].dt.date
        show["exit_date"] = show["exit_date"].dt.date
        show["weight"] = show["weight"].round(3)
        show["return_pct"] = show["return_pct"].round(2)
        print("Last 6 trades:")
        cols = ["entry_date", "exit_date", "weight", "units", "return_pct", "reason"]
        print(show[cols].tail(6).to_string(index=False))

    if args.plot:
        plot_result(args.ticker, res, base, args.plot_path)


if __name__ == "__main__":
    main()
