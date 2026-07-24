"""
Diversified multi-asset-class trend portfolio ("managed futures"-style).

This is the one place in this project where the honestly-measured numbers held
up. Single-stock long/flat breakouts showed no edge once the look-ahead was
removed; the edge that survived came from DIVERSIFICATION across uncorrelated
asset-class trends. This module builds that properly.

  * Universe : a fixed set of liquid asset-class ETFs (equities, rates, credit,
    commodities, gold, FX, real estate). ETFs rarely delist to zero, so this
    benchmark carries far less survivorship bias than a single-stock universe.
  * Signal   : symmetric Donchian breakout per asset -- long above the N-day
    high, (optionally) short below the N-day low, carry otherwise.
  * Sizing   : inverse-volatility per asset (equal risk), then scale the whole
    book to a target portfolio volatility, capped by a gross-leverage limit.
  * Timing   : decided at the close, filled next open, earning the FOLLOWING
    open-to-open move (open_ret.shift(-1)) -- no look-ahead.

Usage:
    python trend_portfolio.py --plot
    python trend_portfolio.py --long-short --channel 100 --target-vol 0.12
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import donchian as dc

PPY = 252

# Fixed, diversified asset-class universe.
UNIVERSE = {
    "SPY": "US equity", "QQQ": "US equity", "IWM": "US equity",
    "EFA": "Intl equity", "EEM": "EM equity",
    "TLT": "Long bond", "IEF": "Mid bond",
    "LQD": "IG credit", "HYG": "HY credit",
    "DBC": "Commodities", "GLD": "Gold", "SLV": "Silver", "USO": "Oil",
    "UUP": "US dollar", "VNQ": "REIT",
}


# --------------------------------------------------------------------------- #
# Signal & per-asset panels
# --------------------------------------------------------------------------- #
def trend_signal(raw: pd.DataFrame, channel: int, long_short: bool) -> pd.Series:
    """Symmetric Donchian breakout: +1 above the N-day high, -1 below the N-day
    low, carried between. Long-only clips shorts to flat."""
    hi = raw["High"].rolling(channel).max().shift(1).to_numpy()
    lo = raw["Low"].rolling(channel).min().shift(1).to_numpy()
    c = raw["Close"].to_numpy()
    st = np.zeros(len(c))
    pos = 0
    for i in range(len(c)):
        if not np.isnan(hi[i]) and c[i] > hi[i]:
            pos = 1
        elif not np.isnan(lo[i]) and c[i] < lo[i]:
            pos = -1
        st[i] = pos
    s = pd.Series(st, index=raw.index)
    return s if long_short else s.clip(lower=0)


def build_panels(tickers, start, end, cfg):
    sig, cret, oret, avol = {}, {}, {}, {}
    for t in tickers:
        try:
            raw = dc.load_data(t, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {t}: skipped ({exc})")
            continue
        sig[t] = trend_signal(raw, cfg["channel"], cfg["long_short"])
        cret[t] = raw["Close"].pct_change()
        # honest forward return: decided at close, filled next open
        oret[t] = raw["Open"].pct_change().shift(-1)
        avol[t] = (cret[t].rolling(cfg["vol_window"]).std() * np.sqrt(PPY)).shift(1)

    idx = sorted(set().union(*[s.index for s in sig.values()]))
    SIG = pd.DataFrame(sig).reindex(idx)
    CR = pd.DataFrame(cret).reindex(idx)
    OR = pd.DataFrame(oret).reindex(idx)
    AV = pd.DataFrame(avol).reindex(idx)
    return SIG, CR, OR, AV, list(sig)


# --------------------------------------------------------------------------- #
# Allocation: inverse-vol per asset, then portfolio vol target
# --------------------------------------------------------------------------- #
def allocate(SIG, CR, AV, cfg):
    sig = SIG.to_numpy()
    cr = CR.to_numpy()
    av = AV.to_numpy()
    n, m = sig.shape

    # per-asset inverse-vol weight (equal risk), capped
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = sig * (cfg["asset_vol"] / av)
    raw = np.clip(np.nan_to_num(raw), -cfg["asset_cap"], cfg["asset_cap"])

    vw = cfg["vol_window"]
    tv = cfg["target_vol"]
    mg = cfg["max_gross"]
    W = np.zeros((n, m))
    for t in range(n):
        w = raw[t]
        g = np.abs(w).sum()
        if g <= 0 or t < vw:
            continue
        win = np.nan_to_num(cr[t - vw + 1:t + 1])          # trailing returns
        book_hist = win.dot(w)                             # book P&L at these weights
        sd = book_hist.std()
        if not np.isfinite(sd) or sd <= 0:
            continue
        est_vol = sd * np.sqrt(PPY)
        s = min(tv / est_vol, mg / g)                      # vol target, capped by gross
        W[t] = s * w
    return pd.DataFrame(W, index=SIG.index, columns=SIG.columns)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def curve_stats(ret):
    r = ret.dropna()
    if r.empty:
        return dict(cagr=np.nan, vol=np.nan, sharpe=np.nan, maxdd=np.nan, total=np.nan)
    eq = (1 + r).cumprod(); ny = len(r) / PPY
    return dict(cagr=eq.iloc[-1] ** (1 / ny) - 1, vol=r.std() * np.sqrt(PPY),
                sharpe=r.mean() / r.std() * np.sqrt(PPY) if r.std() > 0 else np.nan,
                maxdd=(eq / eq.cummax() - 1).min(), total=eq.iloc[-1] - 1)


def run(SIG, CR, OR, AV, cfg):
    W = allocate(SIG, CR, AV, cfg)
    held = W.shift(1).fillna(0.0)
    oret = OR.fillna(0.0)                                  # already open_ret.shift(-1)
    gross_ret = (held * oret).sum(axis=1)
    turnover = held.diff().abs().fillna(held.abs()).sum(axis=1)
    port_ret = gross_ret - turnover * (cfg["cost_bps"] / 10_000.0)
    return dict(weights=held, port_ret=port_ret,
                gross=held.abs().sum(axis=1), net=held.sum(axis=1),
                n_pos=(held.abs() > 1e-6).sum(axis=1),
                contrib=(held * oret).sum(axis=0), turnover=turnover)


# --------------------------------------------------------------------------- #
# Benchmarks
# --------------------------------------------------------------------------- #
def benchmark_6040(start, end, index):
    try:
        spy = dc.load_data("SPY", start, end)["Open"].pct_change().shift(-1)
        agg = dc.load_data("IEF", start, end)["Open"].pct_change().shift(-1)
    except Exception:  # noqa: BLE001
        return None, None
    spy = spy.reindex(index).fillna(0.0)
    agg = agg.reindex(index).fillna(0.0)
    return (0.6 * spy + 0.4 * agg), spy


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(res, cfg, bench6040, spy, classes):
    port = res["port_ret"]
    ps = curve_stats(port)
    active = res["gross"][res["gross"] > 0]

    print("\n" + "=" * 66)
    print("  DIVERSIFIED TREND PORTFOLIO  (managed-futures style)")
    print("=" * 66)
    print(f"  Signal   : {cfg['channel']}-day Donchian breakout, "
          f"{'LONG/SHORT' if cfg['long_short'] else 'long-only'}")
    print(f"  Sizing   : inverse-vol, target {cfg['target_vol']*100:.0f}% vol, "
          f"max {cfg['max_gross']:.1f}x gross")
    print("-" * 66)
    print(f"  {'':<16}{'Trend port':>12}{'60/40':>10}{'SPY':>10}")
    b1 = curve_stats(bench6040) if bench6040 is not None else None
    b2 = curve_stats(spy) if spy is not None else None
    def c(v, pct=True):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return f"{'n/a':>10}"
        return f"{v*100:>9.1f}%" if pct else f"{v:>10.2f}"
    for label, k in [("Total return", "total"), ("CAGR", "cagr"),
                     ("Volatility", "vol"), ("Max drawdown", "maxdd")]:
        print(f"  {label:<16}{c(ps[k])}"
              f"{c(b1[k]) if b1 else c(None)}{c(b2[k]) if b2 else c(None)}")
    print(f"  {'Sharpe':<16}{c(ps['sharpe'],0)}{c(b1['sharpe'],0) if b1 else c(None)}"
          f"{c(b2['sharpe'],0) if b2 else c(None)}")
    print("-" * 66)
    print(f"  Realised vol vs target : {ps['vol']*100:.1f}%  (target {cfg['target_vol']*100:.0f}%)")
    print(f"  Avg gross / net expo   : {active.mean():.2f}x / {res['net'].mean():.2f}x")
    print(f"  Avg # positions        : {res['n_pos'][res['n_pos']>0].mean():.1f}")
    print(f"  Turnover / year        : {res['turnover'].sum()/(len(port)/PPY):.1f}x")
    print("=" * 66)
    top = res["contrib"].sort_values(ascending=False)
    lab = lambda k: f"{k}({classes.get(k,'')})"
    print("  Top contributors  :", ", ".join(f"{lab(k)} {v*100:+.0f}%" for k, v in top.head(4).items()))
    print("  Worst contributors:", ", ".join(f"{lab(k)} {v*100:+.0f}%" for k, v in top.tail(3).items()))
    print("=" * 66 + "\n")


def plot(res, bench6040, spy, cfg, path):
    import matplotlib.pyplot as plt
    port = res["port_ret"]
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    a = ax[0, 0]
    a.plot((1 + port).cumprod(), color="tab:blue", lw=1.6, label="Trend portfolio")
    if bench6040 is not None:
        a.plot((1 + bench6040).cumprod(), color="tab:orange", lw=1.1, ls="--", label="60/40")
    if spy is not None:
        a.plot((1 + spy).cumprod(), color="gray", lw=1.0, ls=":", label="SPY")
    a.set_yscale("log"); a.set_ylabel("Growth of $1 (log)")
    a.set_title("Diversified trend vs benchmarks"); a.legend(loc="upper left"); a.grid(alpha=0.3, which="both")

    a = ax[0, 1]
    eq = (1 + port).cumprod(); dd = eq / eq.cummax() - 1
    a.fill_between(dd.index, dd * 100, 0, color="tab:blue", alpha=0.4)
    a.set_title("Drawdown"); a.set_ylabel("%"); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.fill_between(res["gross"].index, res["gross"], 0, color="tab:green", alpha=0.3, label="Gross")
    a.plot(res["net"].index, res["net"], color="tab:blue", lw=0.8, label="Net")
    a.axhline(0, color="k", lw=0.5); a.set_title("Leverage (gross & net)"); a.legend(loc="upper left"); a.grid(alpha=0.3)

    a = ax[1, 1]
    c = res["contrib"].sort_values() * 100
    a.barh(c.index, c.values, color=["tab:red" if x < 0 else "tab:blue" for x in c.values])
    a.set_title("Per-asset P&L contribution (%)"); a.grid(alpha=0.3, axis="x"); a.tick_params(labelsize=7)

    fig.suptitle(f"Diversified {cfg['channel']}d trend portfolio", fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"Chart saved to {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Diversified multi-asset trend portfolio.")
    p.add_argument("--tickers", default=None)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--channel", type=int, default=50, help="Donchian breakout window.")
    p.add_argument("--long-short", action="store_true", help="Enable the short side.")
    p.add_argument("--target-vol", type=float, default=0.10)
    p.add_argument("--max-gross", type=float, default=3.0)
    p.add_argument("--asset-vol", type=float, default=0.10, help="Per-asset vol scaling.")
    p.add_argument("--asset-cap", type=float, default=2.0, help="Per-asset leverage cap.")
    p.add_argument("--vol-window", type=int, default=60)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-path", default="trend_portfolio.png")
    return p.parse_args()


def main():
    args = parse_args()
    tickers = ([t.strip().upper() for t in args.tickers.split(",")]
               if args.tickers else list(UNIVERSE))
    cfg = dict(channel=args.channel, long_short=args.long_short,
               target_vol=args.target_vol, max_gross=args.max_gross,
               asset_vol=args.asset_vol, asset_cap=args.asset_cap,
               vol_window=args.vol_window, cost_bps=args.cost_bps)

    print(f"Building diversified trend portfolio over {len(tickers)} assets...")
    SIG, CR, OR, AV, names = build_panels(tickers, args.start, args.end, cfg)
    res = run(SIG, CR, OR, AV, cfg)
    bench6040, spy = benchmark_6040(args.start, args.end, SIG.index)

    print_report(res, cfg, bench6040, spy, UNIVERSE)
    if args.plot:
        plot(res, bench6040, spy, cfg, args.plot_path)


if __name__ == "__main__":
    main()
