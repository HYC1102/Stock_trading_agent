"""
40/40/20 diversified portfolio strategy.

  * 40%  QQQ                 — US growth engine (held plain, no vol overlay)
  * 40%  diversified trend   — managed-futures across 18 asset-class ETFs
  * 20%  bonds (IEF)         — ballast

Rebalanced with a 15% no-trade band: a position is only traded when its weight
has drifted more than 15% from target (plus trend entries/exits always execute).
Signals are decided on the close and filled at the next open (no look-ahead).

This module is self-contained. Run `python strategy.py` for a summary, or import
`current_state()` / `backtest()` (used by dashboard.py).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yfinance as yf

PPY = 252

# The trend sleeve's universe: diversified asset-class ETFs.
UNIVERSE = {
    "SPY": "US equity", "QQQ": "US equity", "IWM": "US equity",
    "EFA": "Intl equity", "EEM": "EM equity",
    "TLT": "Long bond", "IEF": "Mid bond",
    "LQD": "IG credit", "HYG": "HY credit",
    "DBC": "Commodities", "GLD": "Gold", "SLV": "Silver", "USO": "Oil",
    "UUP": "US dollar", "VNQ": "REIT",
    "SOXX": "Semis", "XLK": "Tech sector", "XLE": "Energy sector",
}

CONFIG = dict(
    qqq_w=0.0, trend_w=1.0, bond_w=0.0, bond_ticker="IEF",
    band=0.15, channel=50, vol_window=60, asset_budget=0.028,
    max_name=0.25, cost_bps=2.0, start="2010-01-01",
    # trend-sleeve entry signal: "donchian" (N-day channel) or "atr" (Keltner-style
    # ATR breakout, reconstruction of the RAAM paper's ATR Trend/Breakout System).
    signal="donchian", atr_period=42, atr_mult=2.0,
    # vol-targeted QQQ core: hold qqq_w x min(1, target/realized_vol); trimmed
    # part goes to cash. target_vol=None -> flat qqq_w (no scaling).
    # (qqq_w=0 -> no separate QQQ core; QQQ can still be held via the trend sleeve.)
    qqq_target_vol=None,
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_prices(ticker: str, start: str, end: str | None = None, retries: int = 3) -> pd.DataFrame:
    import time
    last = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df[["Open", "High", "Low", "Close"]].dropna()
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(1.5 * (attempt + 1))                    # back off on rate limits
    raise ValueError(f"No data for {ticker!r} after {retries} tries ({last})")


def build_panels(start: str, end: str | None = None):
    """Return aligned (signal, next-open-return, trailing-vol, close) frames."""
    sig, oret, vol, close = {}, {}, {}, {}
    for t in UNIVERSE:
        try:
            raw = load_prices(t, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {t}: skipped ({exc})")
            continue
        sig[t] = entry_signal(raw)
        oret[t] = raw["Open"].pct_change().shift(-1)           # honest next-open fill
        vol[t] = (raw["Close"].pct_change().rolling(CONFIG["vol_window"]).std()
                  * np.sqrt(PPY)).shift(1)
        close[t] = raw["Close"]
    idx = sorted(set().union(*[s.index for s in oret.values()]))
    return (pd.DataFrame(sig).reindex(idx), pd.DataFrame(oret).reindex(idx),
            pd.DataFrame(vol).reindex(idx), pd.DataFrame(close).reindex(idx))


# --------------------------------------------------------------------------- #
# Signal + sleeves
# --------------------------------------------------------------------------- #
def trend_signal(raw: pd.DataFrame, channel: int) -> pd.Series:
    """Long/flat 50-day Donchian breakout (long above the N-day high, out below
    the N-day low)."""
    hi = raw["High"].rolling(channel).max().shift(1).to_numpy()
    lo = raw["Low"].rolling(channel).min().shift(1).to_numpy()
    c = raw["Close"].to_numpy()
    st = np.zeros(len(c)); pos = 0
    for i in range(len(c)):
        if not np.isnan(hi[i]) and c[i] > hi[i]:
            pos = 1
        elif not np.isnan(lo[i]) and c[i] < lo[i]:
            pos = 0
        st[i] = pos
    return pd.Series(st, index=raw.index)


def atr_breakout_signal(raw: pd.DataFrame, period: int, mult: float) -> pd.Series:
    """Long/flat Keltner-style ATR breakout — reconstruction (A) of the RAAM paper's
    ATR Trend/Breakout System: go long when the day's HIGH pierces EMA + mult*ATR,
    go flat when the LOW pierces EMA - mult*ATR. Bands use prior-day values so there
    is no look-ahead (mirrors trend_signal's shifted-channel convention)."""
    h, l, c = raw["High"], raw["Low"], raw["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    basis = c.ewm(span=period, adjust=False).mean()
    upper = (basis + mult * atr).shift(1).to_numpy()
    lower = (basis - mult * atr).shift(1).to_numpy()
    H, L = h.to_numpy(), l.to_numpy()
    st = np.zeros(len(c)); pos = 0
    for i in range(len(c)):
        if not np.isnan(upper[i]) and H[i] > upper[i]:
            pos = 1
        elif not np.isnan(lower[i]) and L[i] < lower[i]:
            pos = 0
        st[i] = pos
    return pd.Series(st, index=raw.index)


def entry_signal(raw: pd.DataFrame) -> pd.Series:
    """Dispatch to the configured trend-sleeve entry signal."""
    if CONFIG.get("signal", "donchian") == "atr":
        return atr_breakout_signal(raw, CONFIG["atr_period"], CONFIG["atr_mult"])
    return trend_signal(raw, CONFIG["channel"])


def trend_sleeve(SIG, OR, AV) -> pd.DataFrame:
    """Signal-only trend sleeve: buy inverse-vol-sized at a breakout, hold
    untouched, sell at breakdown. Returns per-ETF weights (fraction of the
    sleeve), no leverage."""
    S = SIG.shift(1).fillna(0).to_numpy()
    R = OR.fillna(0.0).to_numpy()
    V = AV.to_numpy()
    n, m = S.shape
    cost = CONFIG["cost_bps"] / 1e4
    budget, capw = CONFIG["asset_budget"], CONFIG["max_name"]
    pos = np.zeros(m); cash = 1.0
    W = np.zeros((n, m))
    for t in range(n):
        eq_open = cash + pos.sum()
        for a in range(m):
            want, have = S[t, a] > 0.5, pos[a] > 1e-12
            if have and not want:
                cash += pos[a] * (1 - cost); pos[a] = 0.0
            elif want and not have:
                va = V[t, a]
                w = 0.0 if (np.isnan(va) or va <= 0) else min(capw, budget / va)
                spend = min(w * eq_open, cash)
                if spend > 1e-9:
                    pos[a] = spend * (1 - cost); cash -= spend
        pos = pos * (1 + np.nan_to_num(R[t]))
        e = cash + pos.sum()
        W[t] = pos / e if e > 0 else 0.0
    return pd.DataFrame(W, index=SIG.index, columns=SIG.columns)


def qqq_core_weight(AV) -> pd.Series:
    """Vol-targeted QQQ core: qqq_w capped, scaled by min(1, target/realized_vol).
    Calm markets -> full qqq_w; turbulent -> less (trimmed part held as cash).
    ``qqq_target_vol=None`` disables scaling (flat qqq_w)."""
    tv = CONFIG.get("qqq_target_vol")
    if not tv:
        return pd.Series(CONFIG["qqq_w"], index=AV.index)
    scale = np.minimum(1.0, tv / AV["QQQ"]).clip(upper=1.0).fillna(1.0)
    return CONFIG["qqq_w"] * scale


def net_targets(SIG, OR, AV) -> pd.DataFrame:
    """Combined per-ETF target weights: vol-targeted QQQ + 40% trend + 20% bonds
    (the QQQ portion trimmed by vol-targeting is left in cash)."""
    TW = trend_sleeve(SIG, OR, AV)
    NET = CONFIG["trend_w"] * TW
    NET["QQQ"] = NET["QQQ"] + qqq_core_weight(AV)
    NET[CONFIG["bond_ticker"]] = NET[CONFIG["bond_ticker"]] + CONFIG["bond_w"]
    return NET


# --------------------------------------------------------------------------- #
# Backtest with the no-trade band
# --------------------------------------------------------------------------- #
def backtest(NET, OR, capital=23_000.0, band=None):
    band = CONFIG["band"] if band is None else band
    idx = NET.index
    tickers = [t for t in NET.columns if NET[t].abs().max() > 1e-4]
    RET = OR.reindex(idx).fillna(0.0)
    pos = {t: NET[t].iloc[0] * capital for t in tickers}
    cash = capital - sum(pos.values())
    eqs = [capital]; trades = []; gross = [sum(abs(v) for v in pos.values())]
    for i in range(1, len(idx)):
        for t in tickers:
            pos[t] *= (1 + RET[t].iloc[i])
        equity = sum(pos.values()) + cash
        for t in tickers:
            tgt = NET[t].iloc[i] * equity; cur = pos[t]
            opening = tgt > 1e-6 * equity and cur <= 1e-6 * equity
            closing = tgt <= 1e-6 * equity and cur > 1e-6 * equity
            rel = abs(cur - tgt) / tgt if tgt > 1e-6 * equity else (9 if cur > 1e-6 * equity else 0)
            if opening or closing or rel > band:
                trades.append(dict(date=idx[i], ticker=t, delta=tgt - cur,
                                   target=tgt, equity=equity))
                cash += cur - tgt; pos[t] = tgt
        eqs.append(sum(pos.values()) + cash)
        gross.append(sum(v for v in pos.values() if v > 0) / (sum(pos.values()) + cash))
    eq = pd.Series(eqs, index=idx)
    return dict(equity=eq, trades=pd.DataFrame(trades), positions=pos, cash=cash,
                tickers=tickers, gross=pd.Series(gross, index=idx))


# --------------------------------------------------------------------------- #
# Current state (for the dashboard)
# --------------------------------------------------------------------------- #
def metrics(eq: pd.Series) -> dict:
    r = eq.pct_change().dropna(); ny = len(r) / PPY
    dd = eq / eq.cummax() - 1
    return dict(cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / ny) - 1,
                sharpe=r.mean() / r.std() * np.sqrt(PPY) if r.std() > 0 else np.nan,
                vol=r.std() * np.sqrt(PPY), maxdd=dd.min(), cur_dd=dd.iloc[-1],
                total=eq.iloc[-1] / eq.iloc[0] - 1)


def current_state(capital=23_000.0, start=None, end=None, track_start=None):
    """Everything the dashboard needs: target book, latest trades, exposure.

    ``track_start`` — if given, the chart is a live account tracker from that
    date (rebased so the account starts at ``capital``) instead of the full
    backtest history. Before any data exists past ``track_start`` it is a
    single anchor point at ``capital`` that fills in as you re-run it.
    """
    start = start or CONFIG["start"]
    SIG, OR, AV, CLOSE = build_panels(start, end)
    NET = net_targets(SIG, OR, AV)
    res = backtest(NET, OR, capital=capital)
    eq = res["equity"]; last = NET.index[-1]

    # target book (what to hold now) at the latest close
    tgt = (NET.loc[last] * capital)
    book = tgt[tgt > capital * 0.001].sort_values(ascending=False)
    prices = {t: float(CLOSE.loc[last, t]) for t in book.index}   # last close = buy reference

    # most recent rebalancing day, scaled to the user's capital
    tdf = res["trades"].copy()
    if not tdf.empty:
        tdf["pct"] = tdf["delta"] / tdf["equity"]              # trade as % of portfolio
        tdf["scaled"] = tdf["pct"] * capital                   # $ on the user's capital
        latest_trades = tdf[tdf["date"] == tdf["date"].max()]
    else:
        latest_trades = tdf

    # asset-class exposure from the target weights
    cls = {}
    for t, w in NET.loc[last].items():
        if abs(w) > 1e-4:
            cls[UNIVERSE.get(t, "?")] = cls.get(UNIVERSE.get(t, "?"), 0) + w
    cls = dict(sorted(cls.items(), key=lambda x: -x[1]))

    # SPY correlation + equity chart data (strategy vs SPY, rebased to capital)
    corr = np.nan; chart = None; cur_value = float(eq.iloc[-1])
    try:
        spy_r = load_prices("SPY", start, end)["Open"].pct_change().shift(-1).reindex(eq.index)
        corr = pd.DataFrame({"p": eq.pct_change(), "s": spy_r}).dropna().corr().iloc[0, 1]
        spy_eq = capital * (1 + spy_r.fillna(0)).cumprod()

        if track_start is not None:
            # live account tracker: rebase both to `capital` at track_start
            ts = pd.Timestamp(track_start)
            w = eq[eq.index >= ts]; sw = spy_eq[spy_eq.index >= ts]
            if len(w):
                strat_v = capital * w / w.iloc[0]
                spy_v = (capital * sw / sw.iloc[0]).reindex(strat_v.index).ffill()
                if len(strat_v) > 180:                        # thin out long histories
                    strat_v = strat_v.resample("W").last(); spy_v = spy_v.resample("W").last()
                idx = strat_v.index
                dates = [d.strftime("%d %b %y") for d in idx]
                strat = [round(float(v)) for v in strat_v.values]
                spyl = [round(float(v)) for v in spy_v.values]
            else:                                             # not started yet — single anchor
                dates = [ts.strftime("%d %b %y")]; strat = [round(capital)]; spyl = [round(capital)]
            cur_value = float(strat[-1])
            chart = dict(mode="track", start=ts.strftime("%d %b %Y"),
                         dates=dates, strat=strat, spy=spyl, current=strat[-1])
        else:
            em = eq.resample("ME").last(); sm = spy_eq.resample("ME").last()
            chart = dict(mode="history",
                         dates=[d.strftime("%b %Y") for d in em.index],
                         strat=[round(float(v)) for v in em.values],
                         spy=[round(float(v)) for v in sm.values],
                         current=round(float(em.values[-1])))
    except Exception:  # noqa: BLE001
        pass

    qcore = float(qqq_core_weight(AV).loc[last])              # vol-scaled QQQ core now
    sleeves = {}                                             # only show sleeves with weight
    if CONFIG["qqq_w"] > 1e-9:
        sleeves["QQQ (growth)"] = qcore
        if CONFIG["qqq_w"] - qcore > 1e-3:                    # vol-trimmed QQQ sits in cash
            sleeves["Cash (vol de-risk)"] = CONFIG["qqq_w"] - qcore
    sleeves["Trend (diversifier)"] = CONFIG["trend_w"]
    if CONFIG["bond_w"] > 1e-9:
        sleeves["Bonds (ballast)"] = CONFIG["bond_w"]

    return dict(
        asof=last, capital=capital, equity=eq, book=book, prices=prices, trades=latest_trades,
        cur_value=cur_value, qqq_core=qcore, sleeves=sleeves,
        asset_class=cls, gross=res["gross"].iloc[-1], corr_spy=corr, chart=chart,
        metrics=metrics(eq), universe=UNIVERSE, config=CONFIG,
    )


# --------------------------------------------------------------------------- #
# CLI summary
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="40/40/20 diversified strategy.")
    p.add_argument("--capital", type=float, default=23_000.0)
    p.add_argument("--start", default=CONFIG["start"])
    args = p.parse_args()

    s = current_state(capital=args.capital, start=args.start)
    m = s["metrics"]
    print(f"\n40/40/20 strategy — as of {s['asof'].date()}  (${args.capital:,.0f})")
    print("=" * 56)
    print(f"  CAGR {m['cagr']*100:.1f}%   Sharpe {m['sharpe']:.2f}   "
          f"MaxDD {m['maxdd']*100:.1f}%   corr(SPY) {s['corr_spy']:.2f}")
    print(f"  Value: ${s['equity'].iloc[-1]:,.0f}   current drawdown {m['cur_dd']*100:.1f}%")
    print("\n  TARGET BOOK (hold now):")
    for t, v in s["book"].items():
        print(f"    {t:<6} ${v:>8,.0f}  ({v/args.capital*100:4.1f}%)")
    print("\n  ASSET-CLASS MIX:")
    for c, w in s["asset_class"].items():
        print(f"    {c:<14} {w*100:4.1f}%")
    if not s["trades"].empty:
        print(f"\n  MOST RECENT REBALANCE ({s['trades']['date'].iloc[0].date()}) "
              f"— scaled to ${args.capital:,.0f}:")
        for _, r in s["trades"].iterrows():
            print(f"    {'BUY ' if r['delta']>0 else 'SELL'} {r['ticker']:<6} "
                  f"${abs(r['scaled']):,.0f}  ({r['pct']*100:+.1f}%)")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
