"""
Forward paper-trading engine for the breakout strategy, with persistence.

State (positions, cash, every trade + its sentiment, daily equity) is stored in
data/paper_breakout.json and mirrored to CSVs. Run it daily (e.g. via the
dashboard): it executes the previous session's planned orders at the latest open,
marks the book, checks stops, and plans the next session's orders — ranking fresh
breakouts by the live momentum + news-sentiment score. Nothing is recomputed
retroactively, so the log is a genuine forward record.
"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd

import breakout_sentiment as bs

STATE_PATH = os.path.join("data", "paper_breakout.json")
TRADES_CSV = os.path.join("data", "paper_trades.csv")
EQUITY_CSV = os.path.join("data", "paper_equity.csv")


def load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def save_state(st):
    os.makedirs("data", exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=2, default=str)
    if st["trades"]:
        pd.DataFrame(st["trades"]).to_csv(TRADES_CSV, index=False)
    if st["equity"]:
        pd.DataFrame(st["equity"]).to_csv(EQUITY_CSV, index=False)


def init_state(start_date, capital):
    return dict(start_date=str(start_date), capital=float(capital), cash=float(capital),
                positions={}, trades=[], equity=[], pending=[], last_close=None)


def _cost():
    return bs.CONFIG["cost_bps"] / 1e4


def _equity(positions, cash, price_row):
    v = cash
    for tk, p in positions.items():
        px = price_row.get(tk, np.nan)
        if np.isfinite(px):
            v += p["shares"] * px
    return v


def stop_level(p, ticker, prices):
    """Live exit trigger for a held position: higher of the 1.5-ATR trail and the
    N-day low. Returns (level, binding-rule)."""
    df = prices[ticker]
    atr = float(bs.atr(df, bs.CONFIG["atr_window"]).iloc[-1])
    atr_stop = p["hi"] - bs.CONFIG["atr_stop"] * atr
    low = float(df["Low"].rolling(bs.CONFIG["exit_low"]).min().iloc[-1])
    return (atr_stop, "1.5-ATR trail") if atr_stop >= low else (low, f"{bs.CONFIG['exit_low']}-day low")


def _plan(st, asof, prices, P, regime):
    """Orders to place at the NEXT open, decided on the `asof` close."""
    cl = P["close"].loc[asof]
    pending, sells = [], set()
    for tk, p in st["positions"].items():
        c = cl.get(tk, np.nan)
        if not np.isfinite(c):
            continue                       # transient data gap -> HOLD, never force-sell
        try:
            lvl, rule = stop_level(p, tk, prices)
        except Exception:  # noqa: BLE001
            continue                       # can't compute a stop (missing history) -> hold
        if np.isfinite(lvl) and c < lvl:   # only a genuine, finite stop breach exits
            pending.append(dict(side="SELL", ticker=tk, reason=rule))
            sells.add(tk)
    free = bs.CONFIG["slots"] - (len(st["positions"]) - len(sells))
    reg = bool(regime.get(asof, False)) if regime is not None else True
    if reg and free > 0:
        cand = bs.rank_breakouts(prices, bs.build_universe(prices))
        held = set(st["positions"])
        for _, r in cand.iterrows():
            if r.ticker in held or r.ticker in sells:
                continue
            pending.append(dict(side="BUY", ticker=r.ticker, price_hint=round(float(r.close), 2),
                                momentum=round(float(r.momentum), 1),
                                sentiment=round(float(r.sentiment), 1),
                                combined=round(float(r.combined), 1)))
            if len([o for o in pending if o["side"] == "BUY"]) >= free:
                break
    return pending


def _execute(st, d, prices, P):
    """Fill the pending orders at day d's open."""
    op, cl, c = P["open"].loc[d], P["close"].loc[d], _cost()
    for o in [x for x in st["pending"] if x["side"] == "SELL"]:
        tk = o["ticker"]; p = st["positions"].get(tk); px = op.get(tk, np.nan)
        if p and np.isfinite(px):
            val = p["shares"] * px
            st["cash"] += val * (1 - c)
            st["trades"].append(dict(date=str(d.date()), side="SELL", ticker=tk,
                                     shares=round(p["shares"], 3), price=round(float(px), 2),
                                     value=round(val, 2), reason=o.get("reason", ""),
                                     momentum="", sentiment="", combined=""))
            del st["positions"][tk]
    eq = _equity(st["positions"], st["cash"], op)                # equity after sells, for sizing
    for o in [x for x in st["pending"] if x["side"] == "BUY"]:
        tk = o["ticker"]
        if tk in st["positions"] or len(st["positions"]) >= bs.CONFIG["slots"]:
            continue
        px = op.get(tk, np.nan)
        if not np.isfinite(px) or px <= 0:
            continue
        budget = min(eq / bs.CONFIG["slots"], st["cash"] / (1 + c))
        shares = budget / px
        if shares <= 0:
            continue
        st["cash"] -= shares * px * (1 + c)
        st["positions"][tk] = dict(shares=shares, entry=round(float(px), 2),
                                   entry_date=str(d.date()), hi=float(cl.get(tk, px)))
        st["trades"].append(dict(date=str(d.date()), side="BUY", ticker=tk,
                                 shares=round(shares, 3), price=round(float(px), 2),
                                 value=round(shares * px, 2), reason="breakout",
                                 momentum=o.get("momentum"), sentiment=o.get("sentiment"),
                                 combined=o.get("combined")))


def advance(st, prices, P, regime):
    """Process any new trading days since last run, then plan the next session."""
    close = P["close"]; asof = close.index[-1]
    start = pd.Timestamp(st["start_date"]); processed = False
    if asof >= start:
        last = pd.Timestamp(st["last_close"]) if st.get("last_close") else None
        for d in close.loc[start:asof].index:
            if last is not None and d <= last:
                continue
            _execute(st, d, prices, P)
            st["equity"].append(dict(date=str(d.date()),
                                     value=round(_equity(st["positions"], st["cash"], close.loc[d]), 2)))
            for tk, p in st["positions"].items():
                cc = close.loc[d].get(tk, np.nan)
                if np.isfinite(cc):
                    p["hi"] = max(p["hi"], float(cc))
            st["last_close"] = str(d.date())
            st["pending"] = _plan(st, d, prices, P, regime)
            processed = True
    if not processed:
        st["pending"] = _plan(st, asof, prices, P, regime)
    return st, asof
