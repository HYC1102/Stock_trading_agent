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
import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

PPY = 252


def _expected_last_session() -> dt.date:
    """Most recent US trading session that should be fully available.

    UTC-based, accounts for weekends and the ~20:00 UTC close only (holidays are
    ignored -- at worst an exchange holiday triggers a couple of harmless extra
    retries).  Used to detect when yfinance hands back a stale response so we can
    re-fetch instead of silently publishing yesterday's data.
    """
    now = dt.datetime.utcnow()
    d = now.date()
    if now.hour < 21:                       # today's ~20:00 UTC close not settled yet
        d -= dt.timedelta(days=1)
    while d.weekday() >= 5:                  # step back over Sat/Sun
        d -= dt.timedelta(days=1)
    return d

# The trend sleeve's universe: diversified asset-class ETFs.
UNIVERSE = {
    "SPY": "US equity", "QQQ": "US equity", "IWM": "US equity",
    "EFA": "Intl equity", "EEM": "EM equity", "FXI": "China equity",
    "TLT": "Long bond", "IEF": "Mid bond",
    "LQD": "IG credit", "HYG": "HY credit",
    "DBC": "Commodities", "GLD": "Gold", "SLV": "Silver", "USO": "Oil",
    "UUP": "US dollar", "VNQ": "REIT",
    "SOXX": "Semis", "XLK": "Tech sector", "XLE": "Energy sector",
    "XBI": "Biotech", "BOTZ": "Robotics",
}

CONFIG = dict(
    qqq_w=0.0, trend_w=1.0, bond_w=0.0, bond_ticker="IEF",
    band=0.15, channel=50, vol_window=60, asset_budget=0.028,
    max_name=0.25, cost_bps=2.0, start="2010-01-01",
    # trend-sleeve entry signal: "donchian" (N-day channel) or "atr" (Keltner-style
    # ATR breakout, reconstruction of the RAAM paper's ATR Trend/Breakout System).
    signal="donchian", atr_period=42, atr_mult=2.0,
    # portfolio volatility targeting (ON): trim total exposure when the book's own
    # trailing vol runs above vol_target. cap=1.0 -> de-risk only, never levers.
    # Robustly lifts OOS Sharpe (~0.84->0.90) and halves drawdown. None -> off.
    vol_target=0.11, vol_target_window=60, vol_target_cap=1.0, vol_target_band=0.10,
    # vol-targeted QQQ core: hold qqq_w x min(1, target/realized_vol); trimmed
    # part goes to cash. target_vol=None -> flat qqq_w (no scaling).
    # (qqq_w=0 -> no separate QQQ core; QQQ can still be held via the trend sleeve.)
    qqq_target_vol=None,
)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _tiingo_token() -> str | None:
    """Tiingo API token from $TIINGO_API_KEY, or a gitignored local file."""
    tok = os.environ.get("TIINGO_API_KEY")
    if tok and tok.strip():
        return tok.strip()
    for p in (".tiingo_token", os.path.join("data", "tiingo_token.txt")):
        if os.path.exists(p):
            with open(p) as f:
                t = f.read().strip()
                if t:
                    return t
    return None


def _tiingo_prices(ticker: str, start: str, end: str | None = None) -> pd.DataFrame | None:
    """Split/dividend-adjusted daily OHLC from Tiingo, or None if unavailable
    (no token, bad response, or error) so the caller can fall back to yfinance."""
    tok = _tiingo_token()
    if not tok:
        return None
    import requests
    params = {"startDate": start, "token": tok, "format": "json"}
    if end:
        params["endDate"] = end
    try:
        r = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                         params=params, timeout=30)
        if r.status_code != 200:
            return None
        rows = r.json()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
        out = pd.DataFrame({"Open": df["adjOpen"], "High": df["adjHigh"],
                            "Low": df["adjLow"], "Close": df["adjClose"]}, index=df.index)
        return out.sort_index().dropna()
    except Exception:  # noqa: BLE001
        return None


PRICE_SOURCE: dict[str, str] = {}   # ticker -> "Tiingo" | "yfinance", from the most recent load


def load_prices(ticker: str, start: str, end: str | None = None, retries: int = 3) -> pd.DataFrame:
    # Primary: Tiingo (a real API contract -- reliable, no stale-response lottery).
    tg = _tiingo_prices(ticker, start, end)
    if tg is not None and not tg.empty:
        PRICE_SOURCE[ticker] = "Tiingo"
        return tg
    # Fallback: yfinance (free but flaky; keeps working with no Tiingo token).
    last = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                PRICE_SOURCE[ticker] = "yfinance"
                return df[["Open", "High", "Low", "Close"]].dropna()
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(1.5 * (attempt + 1))                    # back off on rate limits
    raise ValueError(f"No data for {ticker!r} after {retries} tries ({last})")


def build_panels(start: str, end: str | None = None):
    """Return aligned (signal, next-open-return, trailing-vol, close) frames.

    yfinance intermittently serves a stale response (missing the latest 1-2
    sessions) that flips back to fresh minutes later. When querying live
    (end is None) and the assembled panel is behind the most recent expected
    session, wait briefly and rebuild -- so the dashboard doesn't publish
    yesterday's data just because one fetch came back stale.
    """
    exp = _expected_last_session() if end is None else None
    for attempt in range(4):
        sig, oret, vol, close = {}, {}, {}, {}
        PRICE_SOURCE.clear()
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
        latest = idx[-1].date() if idx else None
        if exp is None or attempt == 3 or (latest and latest >= exp):
            return (pd.DataFrame(sig).reindex(idx), pd.DataFrame(oret).reindex(idx),
                    pd.DataFrame(vol).reindex(idx), pd.DataFrame(close).reindex(idx))
        print(f"  trend data stale (latest {latest} < expected {exp}); "
              f"refetching in 30s (attempt {attempt + 1}/4)...")
        time.sleep(30)


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
def vol_target_scale(returns: pd.Series) -> pd.Series:
    """De-risk multiplier in [0, cap]: vol_target / trailing realized vol, banded so
    it only re-scales on meaningful moves (keeps turnover tiny). cap<=1 -> never
    levers, only trims exposure when the book's own volatility runs hot."""
    vt = CONFIG["vol_target"]; w = CONFIG["vol_target_window"]
    cap = CONFIG["vol_target_cap"]; band = CONFIG["vol_target_band"]
    rv = returns.rolling(w).std() * np.sqrt(PPY)
    tgt = (vt / rv.shift(1)).clip(lower=0.0, upper=cap)   # use vol through prior day
    applied = []; cur = cap
    for v in tgt.to_numpy():
        if np.isfinite(v) and abs(v - cur) > band:
            cur = float(v)
        applied.append(cur)
    return pd.Series(applied, index=returns.index)


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
    gross_s = pd.Series(gross, index=idx)
    scale = pd.Series(1.0, index=idx)
    if CONFIG.get("vol_target"):                          # volatility-targeting overlay
        scale = vol_target_scale(eq.pct_change())
        r = scale * eq.pct_change() - scale.diff().abs().fillna(0.0) * (CONFIG["cost_bps"] / 1e4)
        eq = capital * (1 + r.fillna(0.0)).cumprod()
        gross_s = gross_s * scale
    return dict(equity=eq, trades=pd.DataFrame(trades), positions=pos, cash=cash,
                tickers=tickers, gross=gross_s, scale=scale, scale_now=float(scale.iloc[-1]))


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


def _entry_prices(SIG: pd.DataFrame, CLOSE: pd.DataFrame, tickers, asof) -> dict:
    """For each ticker, the date/price of its most recent breakout entry (last
    0->1 signal flip at or before `asof`) -- used to show gain/loss since entry
    on the target book. Approximate: the sleeve is continuously mark-to-market
    and vol-target-rescaled, not a fixed buy-and-hold lot, so this is "how has
    the price moved since this name was last (re)entered", not a precise
    realized-P&L reconstruction."""
    out = {}
    for t in tickers:
        if t not in SIG.columns:
            continue
        s = (SIG[t].reindex(CLOSE.index).fillna(0) > 0.5).astype(int).loc[:asof]
        flips = s.index[(s == 1) & (s.shift(1).fillna(0) == 0)]
        if len(flips):
            ed = flips[-1]
            ep = CLOSE.loc[ed, t] if ed in CLOSE.index else np.nan
            if np.isfinite(ep):
                out[t] = (ed, float(ep))
    return out


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
    scale_now = res.get("scale_now", 1.0)                 # vol-target de-risk multiplier

    # target book (what to hold now) at the latest close, trimmed by the risk scale
    tgt = (NET.loc[last] * capital * scale_now)
    book = tgt[tgt > capital * 0.001].sort_values(ascending=False)
    prices = {t: float(CLOSE.loc[last, t]) for t in book.index}   # last close = buy reference

    # gain/loss since each name's last breakout entry, + where/when its price came from
    entries = _entry_prices(SIG, CLOSE, book.index, last)
    sources = {t: PRICE_SOURCE.get(t, "yfinance") for t in book.index}
    asof_per_ticker = {t: (CLOSE[t].last_valid_index().date()
                           if CLOSE[t].last_valid_index() is not None else None)
                       for t in book.index}

    # Most recent rebalancing day, scaled to the user's capital and to the
    # volatility overlay in force on that day.  The backtest records trades in
    # the unscaled strategy book, so exposing delta/equity directly would
    # overstate live orders whenever the portfolio is de-risked.
    tdf = res["trades"].copy()
    if not tdf.empty:
        trade_dates = pd.to_datetime(tdf["date"])
        tdf["risk_scale"] = trade_dates.map(res["scale"]).fillna(scale_now)
        tdf["raw_pct"] = tdf["delta"] / tdf["equity"]          # before vol overlay
        tdf["raw_dollars"] = tdf["raw_pct"] * capital
        tdf["pct"] = tdf["raw_pct"] * tdf["risk_scale"]        # executable order weight
        tdf["scaled"] = tdf["pct"] * capital                   # executable order dollars
        tdf["target_pct"] = (tdf["target"] / tdf["equity"]
                             * tdf["risk_scale"])
        tdf["dollar_target"] = tdf["target_pct"] * capital     # holding after the order
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
            # Live account tracker: mark the held book to each CLOSE
            # (close-to-close, weights from the prior close) so the value
            # reflects the latest close. The backtest equity above books
            # returns open-to-open (honest next-open fills), so its final bar
            # stays flat until the next open exists -- not what a live account
            # holder wants to see mid-week.
            ts = pd.Timestamp(track_start)
            cret = CLOSE.pct_change()
            book_r = (NET.reindex(CLOSE.index).shift(1) * cret).sum(axis=1, min_count=1)
            beq = (1 + book_r.fillna(0.0)).cumprod()
            spy_eq = load_prices("SPY", start, end)["Close"].reindex(CLOSE.index).ffill()
            w = beq[beq.index >= ts]; sw = spy_eq[spy_eq.index >= ts]
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
        cur_value=cur_value, qqq_core=qcore, sleeves=sleeves, scale_now=scale_now,
        asset_class=cls, gross=res["gross"].iloc[-1], corr_spy=corr, chart=chart,
        metrics=metrics(eq), universe=UNIVERSE, config=CONFIG,
        entries=entries, sources=sources, asof_per_ticker=asof_per_ticker,
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
