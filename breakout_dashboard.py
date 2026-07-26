"""
Live dashboard for the breakout / 3-slot swing strategy (separate from the
40/40/20 portfolio dashboard). Generates a self-contained breakout_dashboard.html:

  1. Universe        top-223 US stocks by dollar volume, rebuilt WEEKLY
  2. Breakouts today ranked by a MOMENTUM index + SENTIMENT score (combined)
  3. Trades & positions the strategy would have made
  4. Strategy measures

Config lives in breakout_sentiment.CONFIG (locked: 1.5-ATR trail + 10-day-low,
fixed 1/3 sizing, 200-day SPY regime filter, 3 slots).
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

import breakout_sentiment as bs

CSS = """
:root{--bg:#fbfbfa;--card:#fff;--ink:#1a1a19;--mut:#6b6a66;--line:#e6e4dd;
--blue:#2a78d6;--green:#1baf7a;--amber:#c98500;--red:#c0392b}
*{box-sizing:border-box;margin:0}
.dash{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);
color:var(--ink);padding:28px;line-height:1.5;max-width:960px;margin:0 auto}
h1{font-size:22px;font-weight:600}h2{font-size:16px;font-weight:600;margin:26px 0 12px}
.sub{color:var(--mut);font-size:13px;margin-top:2px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .l{font-size:12px;color:var(--mut)}.card .v{font-size:21px;font-weight:600;margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line)}
th{font-size:12px;color:var(--mut);font-weight:500;background:#f6f5f1}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.buy{color:var(--green);font-weight:600}.sell{color:var(--red);font-weight:600}
.pos{color:var(--green)}.neg{color:var(--red)}
.hold{background:#eef7f2}
.bar{height:16px;border-radius:4px;background:var(--blue);display:inline-block;vertical-align:middle}
.pill{display:inline-block;background:#eef4fb;color:var(--blue);font-size:12px;padding:2px 9px;border-radius:20px}
.foot{color:var(--mut);font-size:12px;margin-top:24px;border-top:1px solid var(--line);padding-top:12px}
.tag{font-size:11px;color:var(--amber);background:#fdf6e9;padding:1px 7px;border-radius:10px}
"""


def episodes(trades: pd.DataFrame):
    """Walk the trade log into closed round-trips + still-open positions."""
    ep, closed = {}, []
    for _, t in trades.sort_values("date").iterrows():
        e = ep.get(t.ticker)
        if e is None:
            e = dict(cost=0.0, proceeds=0.0, shares=0.0, entry=t.date, exit=t.date)
            ep[t.ticker] = e
        if t.side == "BUY":
            e["cost"] += t.value; e["shares"] += t.shares
        else:
            e["proceeds"] += t.value; e["shares"] -= t.shares; e["exit"] = t.date
        if e["shares"] <= 1e-6 and e["cost"] > 0:
            closed.append(dict(ticker=t.ticker, cost=e["cost"], proceeds=e["proceeds"],
                               pnl=e["proceeds"] - e["cost"], entry=e["entry"], exit=e["exit"]))
            del ep[t.ticker]
    return closed, ep


def build(capital: float, start: str):
    bs.CONFIG.update(sizing="full", weight_mode="fixed", slots=3, regime=True,
                     rank_mode="proxy", atr_stop=1.5, exit_low=10, use_exit_low=True,
                     pct_stop=None, take_profit=None, time_stop=None,
                     universe_size=223, rebuild="W", pool="broad")

    tickers = bs.broad_universe()
    prices = bs.download_prices(tickers, period="3y")
    P = bs.build_panels(prices)
    regime = bs.spy_regime(P["close"].index)
    r = bs.backtest(capital=capital, P=P, regime_full=regime, start=start)
    eq, trades, holds, m = r["equity"], r["trades"], r["holds"], r["metrics"]
    asof = eq.index[-1]

    # universe + today's ranked breakouts
    universe = bs.build_universe(prices)
    cand = bs.rank_breakouts(prices, universe)

    # current positions (open episodes) with unrealised P&L
    closed, open_pos = episodes(trades)
    last_book = holds[-1][2]; cash = holds[-1][1]
    positions = []
    for tk, e in open_pos.items():
        val = last_book.get(tk, np.nan)
        if not (np.isfinite(val) and val > 1):
            continue
        avg = e["cost"] / e["shares"] if e["shares"] else np.nan
        px = float(P["close"].loc[asof, tk])
        positions.append(dict(ticker=tk, shares=e["shares"], avg=avg, price=px, value=val,
                              ret=px / avg - 1 if avg else 0, entry=e["entry"]))
    positions.sort(key=lambda x: -x["value"])

    # measures
    ret_tot = eq.iloc[-1] / eq.iloc[0] - 1
    wins = [c for c in closed if c["pnl"] > 0]
    win_rate = len(wins) / len(closed) if closed else float("nan")
    avg_hold = np.mean([(c["exit"] - c["entry"]).days for c in closed]) if closed else float("nan")
    n_buys = int((trades["side"] == "BUY").sum())

    # SPY benchmark, rebased
    spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy = spy.reindex(eq.index).ffill()
    spy_eq = capital * spy / spy.iloc[0]

    return dict(capital=capital, start=start, asof=asof, eq=eq, spy_eq=spy_eq, m=m,
                ret_tot=ret_tot, win_rate=win_rate, avg_hold=avg_hold, n_buys=n_buys,
                n_closed=len(closed), cand=cand, positions=positions, cash=cash,
                trades=trades, universe=universe)


def html(s) -> str:
    cap, m = s["capital"], s["m"]
    val = s["eq"].iloc[-1]
    sret = s["spy_eq"].iloc[-1] / s["spy_eq"].iloc[0] - 1
    rc = lambda x: "pos" if x >= 0 else "neg"

    cards = [("Account value", f"${val:,.0f}"),
             ("Return", f"{s['ret_tot']*100:+.1f}%"),
             ("vs SPY", f"{(s['ret_tot']-sret)*100:+.1f} pts"),
             ("Sharpe", f"{m['sharpe']:.2f}"),
             ("Max drawdown", f"{m['maxdd']*100:.0f}%"),
             ("Win rate", f"{s['win_rate']*100:.0f}%" if np.isfinite(s['win_rate']) else "—")]
    cards_html = "".join(f'<div class="card"><div class="l">{l}</div><div class="v">{v}</div></div>'
                         for l, v in cards)

    # positions
    if s["positions"]:
        prows = "".join(
            f'<tr><td><b>{p["ticker"]}</b></td><td class="n">{p["shares"]:.1f}</td>'
            f'<td class="n">${p["avg"]:,.2f}</td><td class="n">${p["price"]:,.2f}</td>'
            f'<td class="n">${p["value"]:,.0f}</td>'
            f'<td class="n {rc(p["ret"])}">{p["ret"]*100:+.1f}%</td>'
            f'<td class="n">{p["entry"]:%d %b}</td></tr>' for p in s["positions"])
        pos_html = (f'<table><tr><th>Ticker</th><th class="n">Shares</th><th class="n">Avg cost</th>'
                    f'<th class="n">Price</th><th class="n">Value</th><th class="n">Unreal. P&L</th>'
                    f'<th class="n">Since</th></tr>{prows}'
                    f'<tr><td style="color:var(--mut)">Cash</td><td></td><td></td><td></td>'
                    f'<td class="n">${s["cash"]:,.0f}</td><td></td><td></td></tr></table>')
    else:
        pos_html = '<p class="sub">Flat — no open positions.</p>'

    # today's breakouts (ranked by combined momentum+sentiment)
    c = s["cand"]
    if len(c):
        held = {p["ticker"] for p in s["positions"]}
        crows = ""
        for i, r in c.head(15).iterrows():
            flag = ' <span class="tag">held</span>' if r.ticker in held else (
                ' <span class="tag" style="color:var(--green);background:#eef7f2">top pick</span>' if i < 3 else "")
            cls = ' class="hold"' if r.ticker in held else ""
            crows += (f'<tr{cls}><td>{i+1}</td><td><b>{r.ticker}</b>{flag}</td>'
                      f'<td class="n">${r.close:,.2f}</td>'
                      f'<td class="n">{r.mom_ret*100:+.0f}%</td>'
                      f'<td class="n">{r.momentum:.0f}</td>'
                      f'<td class="n">{r.sentiment:.0f}</td>'
                      f'<td class="n"><b>{r.combined:.0f}</b></td>'
                      f'<td class="n">${r.stop:,.2f}</td></tr>')
        cand_html = (f'<table><tr><th>#</th><th>Ticker</th><th class="n">Price</th>'
                     f'<th class="n">3-mo mom</th><th class="n">Momentum idx</th>'
                     f'<th class="n">Sentiment</th><th class="n">Combined</th>'
                     f'<th class="n">ATR stop</th></tr>{crows}</table>')
    else:
        cand_html = '<p class="sub">No fresh breakouts in the universe today.</p>'

    # recent trades
    tr = s["trades"].sort_values("date").tail(16)
    trows = "".join(
        f'<tr><td>{t.date:%d %b}</td>'
        f'<td class="{"buy" if t.side=="BUY" else "sell"}">{t.side}</td>'
        f'<td><b>{t.ticker}</b></td><td class="n">{t.shares:.1f}</td>'
        f'<td class="n">${t.px:,.2f}</td><td class="n">${t.value:,.0f}</td></tr>'
        for _, t in tr.iterrows())
    trades_html = (f'<table><tr><th>Date</th><th>Action</th><th>Ticker</th>'
                   f'<th class="n">Shares</th><th class="n">Price</th><th class="n">$ value</th>'
                   f'</tr>{trows}</table>')

    # equity chart data (weekly)
    ew = s["eq"].resample("W").last(); sw = s["spy_eq"].resample("W").last()
    import json
    labels = json.dumps([d.strftime("%d %b") for d in ew.index])
    strat = json.dumps([round(float(v)) for v in ew.values])
    spyv = json.dumps([round(float(v)) for v in sw.values])

    return f"""<!doctype html><meta charset="utf-8"><title>Breakout strategy</title>
<style>{CSS}</style><div class="dash">
<h1>Breakout Momentum &mdash; 3-slot swing</h1>
<p class="sub">Top-223 US stocks by volume (weekly) &nbsp;·&nbsp; 20-day breakout &nbsp;·&nbsp;
1.5-ATR trail + 10-day-low exit &nbsp;·&nbsp; 200-day regime filter &nbsp;|&nbsp;
as of {s['asof']:%d %b %Y} &nbsp;·&nbsp; <span class="pill">${cap:,.0f} paper account</span></p>

<div class="cards">{cards_html}</div>
<div style="position:relative;height:230px"><canvas id="eq"></canvas></div>

<h2>Today&rsquo;s breakouts &mdash; ranked by momentum + sentiment</h2>
<p class="sub">Fresh 20-day-high breakouts in the top-223 universe. Combined score =
{(1-bs.CONFIG['sent_weight'])*100:.0f}% momentum index + {bs.CONFIG['sent_weight']*100:.0f}% sentiment.
The top 3 fill the slots. <span class="tag">Sentiment = VADER on recent news headlines</span>
(0 = very bearish, 50 = neutral/no news, 100 = very bullish).</p>
{cand_html}

<h2>Current positions</h2>
{pos_html}

<h2>Recent trades</h2>
{trades_html}

<h2>Strategy measures</h2>
<p class="sub">Since {dt.date.fromisoformat(s['start']):%d %b %Y}, ${cap:,.0f} start.</p>
<div class="cards">
<div class="card"><div class="l">Total return</div><div class="v {rc(s['ret_tot'])}">{s['ret_tot']*100:+.1f}%</div></div>
<div class="card"><div class="l">SPY return</div><div class="v">{sret*100:+.1f}%</div></div>
<div class="card"><div class="l">Sharpe</div><div class="v">{m['sharpe']:.2f}</div></div>
<div class="card"><div class="l">Volatility</div><div class="v">{m['vol']*100:.0f}%</div></div>
<div class="card"><div class="l">Max drawdown</div><div class="v">{m['maxdd']*100:.0f}%</div></div>
<div class="card"><div class="l">Round-trips</div><div class="v">{s['n_closed']}</div></div>
<div class="card"><div class="l">Win rate</div><div class="v">{s['win_rate']*100:.0f}%</div></div>
<div class="card"><div class="l">Avg hold</div><div class="v">{s['avg_hold']:.0f}d</div></div>
</div>

<p class="foot">Generated {dt.datetime.now():%Y-%m-%d %H:%M} by breakout_dashboard.py.
Universe = S&amp;P 500 + Nasdaq-100 + liquid extras, top {bs.CONFIG['universe_size']} by
{bs.CONFIG['adv_window']}-day dollar volume, rebuilt weekly (a free proxy for the true
US-by-volume universe). Survivorship-biased; backtest optimistic. Not investment advice.</p>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
new Chart(document.getElementById('eq'),{{type:'line',
 data:{{labels:{labels},datasets:[
  {{label:'Strategy',data:{strat},borderColor:'#2a78d6',borderWidth:2,pointRadius:0,tension:.1}},
  {{label:'SPY',data:{spyv},borderColor:'#898781',borderDash:[4,3],borderWidth:1.5,pointRadius:0,tension:.1}}]}},
 options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{boxWidth:12,font:{{size:11}}}}}},
  tooltip:{{callbacks:{{label:c=>c.dataset.label+': $'+c.parsed.y.toLocaleString()}}}}}},
  scales:{{y:{{ticks:{{color:'#898781',callback:v=>'$'+(v/1000).toFixed(1)+'k'}},grid:{{color:'#eee'}}}},
   x:{{ticks:{{color:'#898781',maxTicksLimit:8}},grid:{{display:false}}}}}}}}}});
</script>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capital", type=float, default=8800.0)
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--out", default="breakout_dashboard.html")
    a = p.parse_args()
    print("Building breakout dashboard (universe + data + backtest)...")
    s = build(a.capital, a.start)
    with open(a.out, "w") as f:
        f.write(html(s))
    print(f"Wrote {a.out}  (account ${s['eq'].iloc[-1]:,.0f}, {len(s['cand'])} breakouts today)")


if __name__ == "__main__":
    main()
