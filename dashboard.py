"""
Generate a self-contained HTML dashboard for the 40/40/20 strategy.

Two panels (no external dependencies — opens offline in any browser):
  1. Today's actions   — target holdings and the most recent rebalance trades.
  2. Risk & exposure    — sleeve mix, asset-class breakdown, current risk stats.

    python dashboard.py                 # -> dashboard.html
    python dashboard.py --capital 50000
"""

from __future__ import annotations

import argparse
import datetime as dt

import strategy as st

CSS = """
:root{--bg:#fbfbfa;--card:#fff;--ink:#1a1a19;--mut:#6b6a66;--line:#e6e4dd;
--blue:#2a78d6;--green:#1baf7a;--amber:#c98500;--red:#c0392b}
*{box-sizing:border-box;margin:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);
color:var(--ink);padding:28px;line-height:1.5;max-width:900px;margin:0 auto}
h1{font-size:22px;font-weight:600}h2{font-size:16px;font-weight:600;margin:26px 0 12px}
.sub{color:var(--mut);font-size:13px;margin-top:2px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .l{font-size:12px;color:var(--mut)}.card .v{font-size:22px;font-weight:600;margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:14px;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line)}
th{font-size:12px;color:var(--mut);font-weight:500;background:#f6f5f1}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.bar{height:22px;border-radius:5px;display:flex;align-items:center;padding-left:8px;
color:#fff;font-size:12px;font-weight:500;min-width:30px}
.brow{display:grid;grid-template-columns:120px 1fr 56px;align-items:center;gap:10px;margin:6px 0;font-size:13px}
.brow .lab{color:var(--mut)}.brow .val{text-align:right;font-variant-numeric:tabular-nums}
.buy{color:var(--green);font-weight:600}.sell{color:var(--red);font-weight:600}
.foot{color:var(--mut);font-size:12px;margin-top:24px;border-top:1px solid var(--line);padding-top:12px}
.pill{display:inline-block;background:#eef4fb;color:var(--blue);font-size:12px;padding:2px 9px;border-radius:20px}
"""


def bar_row(label, pct, color, maxpct):
    w = max(4, pct / maxpct * 100)
    return (f'<div class="brow"><span class="lab">{label}</span>'
            f'<div class="bar" style="width:{w:.0f}%;background:{color}">{pct*100:.0f}%</div>'
            f'<span class="val">{pct*100:.1f}%</span></div>')


def build_html(s):
    m = s["metrics"]; cap = s["capital"]
    colors = ["#2a78d6", "#1baf7a", "#c98500", "#8a63d2", "#d95f2b", "#4a9",
              "#e07a9a", "#5f9ea0", "#b07d3a"]

    cards = "".join(f'<div class="card"><div class="l">{l}</div><div class="v">{v}</div></div>'
                    for l, v in [
                        ("Portfolio value", f"${s['equity'].iloc[-1]:,.0f}"),
                        ("CAGR", f"{m['cagr']*100:.1f}%"),
                        ("Sharpe", f"{m['sharpe']:.2f}"),
                        ("Max drawdown", f"{m['maxdd']*100:.0f}%"),
                        ("Current drawdown", f"{m['cur_dd']*100:.1f}%"),
                    ])

    book_rows = "".join(
        f'<tr><td>{t}</td><td>{s["universe"].get(t,"")}</td>'
        f'<td class="n">{w/cap*100:.1f}%</td><td class="n">${w:,.0f}</td></tr>'
        for t, w in s["book"].items())

    if not s["trades"].empty:
        tr = "".join(
            f'<tr><td class="{"buy" if r["delta"]>0 else "sell"}">'
            f'{"BUY" if r["delta"]>0 else "SELL"}</td><td>{r["ticker"]}</td>'
            f'<td class="n">${abs(r["scaled"]):,.0f}</td>'
            f'<td class="n">{r["pct"]*100:+.1f}%</td></tr>'
            for _, r in s["trades"].iterrows())
        trades_html = (f'<p class="sub">Most recent rebalance — {s["trades"]["date"].iloc[0].date()} '
                       f'(only positions that broke the 15% band)</p>'
                       f'<table><tr><th>Action</th><th>Ticker</th><th class="n">Amount</th>'
                       f'<th class="n">% of book</th></tr>{tr}</table>')
    else:
        trades_html = '<p class="sub">No rebalancing needed at the latest close.</p>'

    sleeve_max = max(s["sleeves"].values())
    sleeves = "".join(bar_row(l, w, colors[i % len(colors)], sleeve_max)
                      for i, (l, w) in enumerate(s["sleeves"].items()))
    ac_max = max(s["asset_class"].values())
    ac = "".join(bar_row(c, w, colors[i % len(colors)], ac_max)
                 for i, (c, w) in enumerate(s["asset_class"].items()))

    risk_cards = "".join(f'<div class="card"><div class="l">{l}</div><div class="v">{v}</div></div>'
                         for l, v in [
                             ("Gross exposure", f"{s['gross']*100:.0f}%"),
                             ("Correlation to SPY", f"{s['corr_spy']:.2f}"),
                             ("Annualised vol", f"{m['vol']*100:.0f}%"),
                             ("# positions", f"{len(s['book'])}"),
                         ])

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>40/40/20 Dashboard</title><style>{CSS}</style></head><body>
<h1>40 / 40 / 20 diversified portfolio</h1>
<p class="sub">40% QQQ &nbsp;·&nbsp; 40% diversified trend &nbsp;·&nbsp; 20% bonds &nbsp;·&nbsp;
15% no-trade band &nbsp;|&nbsp; as of {s['asof'].date()} &nbsp;·&nbsp;
<span class="pill">${cap:,.0f} account</span></p>
<div class="cards">{cards}</div>

<h2>Today's actions</h2>
<p class="sub">Target book — hold these weights (buy the full list if starting fresh)</p>
<table><tr><th>Ticker</th><th>Asset class</th><th class="n">Weight</th><th class="n">Amount</th></tr>
{book_rows}</table>
{trades_html}

<h2>Risk &amp; exposure</h2>
<p class="sub">Sleeve breakdown</p>{sleeves}
<p class="sub" style="margin-top:16px">Asset-class mix (where the risk actually sits)</p>{ac}
<div class="cards">{risk_cards}</div>

<p class="foot">Generated {dt.datetime.now():%Y-%m-%d %H:%M} by dashboard.py.
Backtest since {s['config']['start']}; total-return data (dividends &amp; coupons included).
Not investment advice — a research tool.</p>
</body></html>"""


def main():
    p = argparse.ArgumentParser(description="Generate the strategy dashboard.")
    p.add_argument("--capital", type=float, default=23_000.0)
    p.add_argument("--start", default=st.CONFIG["start"])
    p.add_argument("--out", default="dashboard.html")
    args = p.parse_args()

    print("Building dashboard (fetching data + running strategy)...")
    s = st.current_state(capital=args.capital, start=args.start)
    with open(args.out, "w") as f:
        f.write(build_html(s))
    print(f"Dashboard written to {args.out}  (open it in a browser)")


if __name__ == "__main__":
    main()
