"""
Bollinger-band mean-reversion strategy for SPY (or any index/ETF).

The opposite of the trend strategy: instead of buying breakouts, it buys
oversold dips and sells the bounce back to the mean. Documented edge on equity
indices, which short-term mean-revert inside their long-term uptrend.

  * Entry  : close below the lower Bollinger band (MA(window) - n_std * std).
  * Exit   : close back above the middle band (the MA), OR a `max_hold` day
             time-stop so it doesn't marry a losing dip.
  * Regime : optionally only buy dips when close > 200-day MA (buy dips in
             uptrends, avoid catching falling knives in bear markets).
  * Timing : decided at the close, filled next open -- no look-ahead.

Long-or-flat, full position. Compares against buy & hold.

Usage:
    python mean_reversion.py --ticker SPY
    python mean_reversion.py --ticker SPY --regime          # 200-MA filter on
    python mean_reversion.py --ticker QQQ --window 20 --n-std 2 --max-hold 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

import donchian as dc

PPY = 252


@dataclass
class MRParams:
    window: int = 20       # Bollinger MA / std window
    n_std: float = 2.0     # band width in std devs
    max_hold: int = 10     # time-stop (bars)
    regime: bool = True    # only buy dips above the 200-day MA (default ON)
    regime_ma: int = 200
    stop_pct: float = 0.0  # optional hard stop-loss vs entry close (0 = off)
    cost_bps: float = 2.0


# --------------------------------------------------------------------------- #
# Indicator + signal
# --------------------------------------------------------------------------- #
def bollinger(df: pd.DataFrame, window: int, n_std: float) -> pd.DataFrame:
    out = df.copy()
    mid = df["Close"].rolling(window).mean()
    sd = df["Close"].rolling(window).std()
    out["bb_mid"] = mid
    out["bb_lower"] = mid - n_std * sd
    out["bb_upper"] = mid + n_std * sd
    return out


def generate_positions(df: pd.DataFrame, p: MRParams) -> pd.Series:
    """Long/flat state decided at each close (executed next open).

    Enter when the close is below the lower band (and, if regime is on, the
    close is above its 200-day MA). Exit when the close is back above the
    middle band, or the position has been held `max_hold` bars.
    """
    close = df["Close"].to_numpy()
    lower = df["bb_lower"].to_numpy()
    mid = df["bb_mid"].to_numpy()
    ma = df["Close"].rolling(p.regime_ma).mean().to_numpy() if p.regime else None
    n = len(df)

    state = np.zeros(n, dtype=int)
    in_pos = False
    held = 0
    entry_ref = np.nan
    for i in range(n):
        ready = not (np.isnan(lower[i]) or np.isnan(mid[i]))
        regime_ok = (not p.regime) or (ma is not None and not np.isnan(ma[i]) and close[i] > ma[i])
        if not in_pos:
            if ready and regime_ok and close[i] < lower[i]:
                in_pos, held, entry_ref = True, 0, close[i]
                state[i] = 1
        else:
            held += 1
            hit_stop = p.stop_pct > 0 and close[i] < entry_ref * (1 - p.stop_pct)
            if close[i] > mid[i] or held >= p.max_hold or hit_stop:
                in_pos = False
                state[i] = 0
            else:
                state[i] = 1
    return pd.Series(state, index=df.index, name="state")


# --------------------------------------------------------------------------- #
# Backtest (honest next-open fills)
# --------------------------------------------------------------------------- #
def backtest(df: pd.DataFrame, state: pd.Series, p: MRParams) -> dict:
    d = df.copy()
    held = state.shift(1).fillna(0)                     # position held on bar t (decided t-1)
    d["position"] = held
    open_ret = d["Open"].pct_change()
    exec_ret = open_ret.shift(-1).fillna(0.0)          # earn the move AFTER the next-open fill
    turnover = held.diff().abs().fillna(held.abs())
    d["strategy_ret"] = held * exec_ret - turnover * (p.cost_bps / 10_000.0)
    d["market_ret"] = exec_ret
    d["equity"] = (1 + d["strategy_ret"]).cumprod()
    trades = _trades(d, state)
    return dict(data=d, trades=trades, metrics=_metrics(d, trades))


def _trades(d: pd.DataFrame, state: pd.Series) -> pd.DataFrame:
    s = state.to_numpy(); opens = d["Open"].to_numpy(); closes = d["Close"].to_numpy()
    dates = d.index; n = len(s); rows = []
    i = 0
    while i < n:
        if s[i] == 1:
            j = i
            while j + 1 < n and s[j + 1] == 1:
                j += 1
            e = i + 1 if i + 1 < n else i                       # fill next open
            x = j + 1 if j + 1 < n else j
            epx, xpx = opens[e], (opens[x] if x != j else closes[j])
            rows.append(dict(entry_date=dates[e], exit_date=dates[x],
                             entry=epx, exit=xpx, ret=(xpx / epx - 1) * 100,
                             bars=j - i + 1, open=(j == n - 1)))
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows)


def _metrics(d: pd.DataFrame, trades: pd.DataFrame) -> dict:
    r = d["strategy_ret"]; eq = d["equity"]; ny = len(d) / PPY
    m = d["market_ret"]; beq = (1 + m).cumprod()
    wins = trades[trades["ret"] > 0]["ret"] if not trades.empty else pd.Series(dtype=float)
    los = trades[trades["ret"] <= 0]["ret"] if not trades.empty else pd.Series(dtype=float)
    pf = wins.sum() / abs(los.sum()) if len(los) and los.sum() != 0 else np.nan
    return dict(
        total=eq.iloc[-1] - 1, cagr=eq.iloc[-1] ** (1 / ny) - 1,
        vol=r.std() * np.sqrt(PPY), sharpe=r.mean() / r.std() * np.sqrt(PPY) if r.std() > 0 else np.nan,
        maxdd=(eq / eq.cummax() - 1).min(), time_in=(d["position"] > 0).mean(),
        n_trades=len(trades), win_rate=(trades["ret"] > 0).mean() if not trades.empty else np.nan,
        avg_win=wins.mean(), avg_loss=los.mean(), profit_factor=pf,
        avg_bars=trades["bars"].mean() if not trades.empty else np.nan,
        bh_total=beq.iloc[-1] - 1, bh_cagr=beq.iloc[-1] ** (1 / ny) - 1,
        bh_sharpe=m.mean() / m.std() * np.sqrt(PPY) if m.std() > 0 else np.nan,
        bh_maxdd=(beq / beq.cummax() - 1).min(),
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report(ticker: str, p: MRParams, m: dict) -> None:
    def pct(x): return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.1f}%"
    print("\n" + "=" * 58)
    print(f"  BOLLINGER MEAN-REVERSION — {ticker}"
          f"   (regime filter: {'ON' if p.regime else 'off'})")
    print("=" * 58)
    print(f"  Bands / hold      : {p.window}d, {p.n_std:.1f} std / max {p.max_hold}d hold")
    print("-" * 58)
    print(f"  {'':<18}{'strategy':>12}{'buy & hold':>14}")
    print(f"  {'Total return':<18}{pct(m['total']):>12}{pct(m['bh_total']):>14}")
    print(f"  {'CAGR':<18}{pct(m['cagr']):>12}{pct(m['bh_cagr']):>14}")
    print(f"  {'Max drawdown':<18}{pct(m['maxdd']):>12}{pct(m['bh_maxdd']):>14}")
    print(f"  {'Sharpe':<18}{m['sharpe']:>12.2f}{m['bh_sharpe']:>14.2f}")
    print("-" * 58)
    print(f"  Time in market    : {pct(m['time_in'])}")
    print(f"  Trades            : {m['n_trades']}")
    print(f"  Win rate          : {pct(m['win_rate'])}")
    print(f"  Avg win / loss    : {pct(m['avg_win']/100 if not np.isnan(m['avg_win']) else np.nan)}"
          f" / {pct(m['avg_loss']/100 if not np.isnan(m['avg_loss']) else np.nan)}")
    print(f"  Profit factor     : {m['profit_factor']:.2f}")
    print(f"  Avg hold          : {m['avg_bars']:.1f} bars")
    print("=" * 58 + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Bollinger mean-reversion backtest.")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--n-std", type=float, default=2.0)
    p.add_argument("--max-hold", type=int, default=10)
    p.add_argument("--no-regime", action="store_true",
                   help="Disable the 200-day MA regime filter (default ON).")
    p.add_argument("--stop-pct", type=float, default=0.0,
                   help="Hard stop-loss vs entry close, e.g. 0.05 = -5%% (0 = off).")
    p.add_argument("--cost-bps", type=float, default=2.0)
    return p.parse_args()


def run_config(raw, p):
    df = bollinger(raw, p.window, p.n_std)
    state = generate_positions(df, p)
    return backtest(df, state, p)


def main():
    args = parse_args()
    raw = dc.load_data(args.ticker, args.start, args.end)
    p = MRParams(window=args.window, n_std=args.n_std, max_hold=args.max_hold,
                 regime=not args.no_regime, stop_pct=args.stop_pct, cost_bps=args.cost_bps)
    res = run_config(raw, p)
    report(args.ticker, p, res["metrics"])


if __name__ == "__main__":
    main()
