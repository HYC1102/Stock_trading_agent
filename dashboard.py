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
import json

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
.tg{background:#fff;border:1px solid var(--line);border-radius:8px;padding:5px 13px;font-size:13px;
cursor:pointer;margin-right:6px;color:var(--mut)}
.tg.on{background:var(--blue);color:#fff;border-color:var(--blue)}
.leg{display:inline-flex;gap:14px;font-size:12px;color:var(--mut);margin-left:6px}
.leg span{display:inline-flex;align-items:center;gap:5px}
.sw{width:13px;height:3px;display:inline-block}
"""


def perf_section(s):
    c = s.get("chart")
    if not c:
        return ""
    track = c.get("mode") == "track"
    n = len(c["dates"])
    if track:
        heading = ("Your account "
                   f'<span style="font-weight:400;font-size:13px;color:var(--mut)">'
                   f'&mdash; tracking since {c["start"]} (rebased to ${s["capital"]:,.0f})</span>')
        note = ('<p class="sub" style="margin:-4px 0 8px">Live tracker: it starts at your '
                'current value and fills in each time you re-run the dashboard.</p>'
                if n <= 1 else "")
        vfmt = "v=>'$'+Math.round(v).toLocaleString()"           # full dollars, small range
        rfmt = "v=>v.toFixed(1)+'%'"
        pr = 4 if n <= 1 else 2                                   # show the anchor dot
    else:
        heading = ('Performance '
                   f'<span style="font-weight:400;font-size:13px;color:var(--mut)">'
                   f'&mdash; strategy track record (backtest, ${s["capital"]:,.0f} '
                   f'since {s["config"]["start"][:4]})</span>')
        note = ""
        vfmt = "v=>'$'+(v/1000).toFixed(0)+'k'"
        rfmt = "v=>v.toFixed(0)+'%'"
        pr = 0
    return f"""
<h2>{heading}</h2>{note}
<div style="margin:6px 0 10px">
  <button id="mV" class="tg on">Account value ($)</button><button id="mR" class="tg">Return (%)</button>
  <span class="leg"><span><span class="sw" style="background:#2a78d6"></span>Strategy</span>
  <span><span class="sw" style="background:#898781"></span>SPY</span></span>
</div>
<div style="position:relative;height:280px"><canvas id="perf"></canvas></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const D={json.dumps(c['dates'])},S={json.dumps(c['strat'])},P={json.dumps(c['spy'])};
const vfmt={vfmt},rfmt={rfmt};
const rb=a=>a.map(v=>(v/a[0]-1)*100);
let mode='value';
const ch=new Chart(document.getElementById('perf'),{{type:'line',
 data:{{labels:D,datasets:[
  {{label:'Strategy',data:S,borderColor:'#2a78d6',borderWidth:2,pointRadius:{pr},tension:.1}},
  {{label:'SPY',data:P,borderColor:'#898781',borderDash:[4,3],borderWidth:1.5,pointRadius:{pr},tension:.1}}]}},
 options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},
  tooltip:{{callbacks:{{label:c=>c.dataset.label+': '+(mode==='value'?'$'+Math.round(c.parsed.y).toLocaleString():c.parsed.y.toFixed(1)+'%')}}}}}},
  scales:{{y:{{ticks:{{color:'#898781',callback:vfmt}},grid:{{color:'#e6e4dd'}}}},
   x:{{ticks:{{color:'#898781',maxTicksLimit:8}},grid:{{display:false}}}}}}}}}});
function setMode(m){{mode=m;
 ch.data.datasets[0].data=m==='return'?rb(S):S;
 ch.data.datasets[1].data=m==='return'?rb(P):P;
 ch.options.scales.y.ticks.callback=m==='return'?rfmt:vfmt;
 ch.update();
 document.getElementById('mV').classList.toggle('on',m==='value');
 document.getElementById('mR').classList.toggle('on',m==='return');}}
document.getElementById('mV').onclick=()=>setMode('value');
document.getElementById('mR').onclick=()=>setMode('return');
</script>"""


def bar_row(label, pct, color, maxpct):
    w = max(4, pct / maxpct * 100)
    return (f'<div class="brow"><span class="lab">{label}</span>'
            f'<div class="bar" style="width:{w:.0f}%;background:{color}">{pct*100:.0f}%</div>'
            f'<span class="val">{pct*100:.1f}%</span></div>')


def next_monday():
    today = dt.date.today()
    ahead = (0 - today.weekday()) % 7 or 7
    return today + dt.timedelta(days=ahead)


def build_html(s, fresh=False, start_date=None, track_date=None):
    m = s["metrics"]; cap = s["capital"]
    since = start_date or track_date          # inception for the account cards / start pill
    colors = ["#2a78d6", "#1baf7a", "#c98500", "#8a63d2", "#d95f2b", "#4a9",
              "#e07a9a", "#5f9ea0", "#b07d3a"]

    if fresh or track_date:
        curv = s.get("cur_value", cap)
        pnl = curv / cap - 1
        card_data = [("Starting capital", f"${cap:,.0f}"),
                     ("Current value", f"${curv:,.0f}"),
                     (f"Return since {since:%d %b}" if since else "Return since start",
                      f"{pnl*100:+.1f}%"),
                     ("Backtest CAGR", f"{m['cagr']*100:.1f}%"),
                     ("Backtest Sharpe", f"{m['sharpe']:.2f}")]
    else:
        card_data = [("Portfolio value", f"${s['equity'].iloc[-1]:,.0f}"),
                     ("CAGR", f"{m['cagr']*100:.1f}%"),
                     ("Sharpe", f"{m['sharpe']:.2f}"),
                     ("Max drawdown", f"{m['maxdd']*100:.0f}%"),
                     ("Current drawdown", f"{m['cur_dd']*100:.1f}%")]
    cards = "".join(f'<div class="card"><div class="l">{l}</div><div class="v">{v}</div></div>'
                    for l, v in card_data)

    # fresh-start buy list: every target position is a BUY
    if fresh:
        px = s.get("prices", {})
        buy_rows = "".join(
            f'<tr><td class="buy">BUY</td><td>{t}</td><td>{s["universe"].get(t,"")}</td>'
            f'<td class="n">{w/cap*100:.1f}%</td><td class="n">${w:,.0f}</td>'
            f'<td class="n">${px.get(t,0):,.2f}</td><td class="n">{w/px[t]:.1f}</td></tr>'
            for t, w in s["book"].items() if px.get(t))
        total = s["book"].sum()
        actions_html = (
            f'<h2>Opening buy list — {start_date:%A %d %b %Y}</h2>'
            f'<p class="sub">Place these buys at the next open ({start_date:%d %b}). '
            f'{len(s["book"])} orders, ${total:,.0f} invested (${cap-total:,.0f} cash), then check back weekly. '
            f'Price = last close on {s["asof"]:%d %b}; the open will differ slightly.</p>'
            f'<table><tr><th>Action</th><th>Ticker</th><th>Asset class</th>'
            f'<th class="n">Weight</th><th class="n">Amount</th>'
            f'<th class="n">Price</th><th class="n">Shares</th></tr>{buy_rows}'
            f'<tr><td></td><td></td><td style="color:var(--mut)">Total invested</td>'
            f'<td class="n">{total/cap*100:.0f}%</td><td class="n">${total:,.0f}</td>'
            f'<td class="n"></td><td class="n"></td></tr></table>')
        book_rows = ""; trades_html = ""
    else:
        book_rows = "".join(
            f'<tr><td>{t}</td><td>{s["universe"].get(t,"")}</td>'
            f'<td class="n">{w/cap*100:.1f}%</td><td class="n">${w:,.0f}</td></tr>'
            for t, w in s["book"].items())
        actions_html = (
            '<h2>Today\'s actions</h2>'
            '<p class="sub">Target book — hold these weights (buy the full list if starting fresh)</p>'
            f'<table><tr><th>Ticker</th><th>Asset class</th><th class="n">Weight</th>'
            f'<th class="n">Amount</th></tr>{book_rows}</table>')

    if not fresh and not s["trades"].empty:
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
    elif not fresh:
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

    startpill = (f' &nbsp;·&nbsp; <span class="pill">start {since:%d %b %Y}</span>'
                 if since else "")
    C = s["config"]
    parts = ([f'{C["qqq_w"]:.0%} QQQ'] if C["qqq_w"] > 1e-9 else []) + \
            [f'{C["trend_w"]:.0%} diversified trend'] + \
            ([f'{C["bond_w"]:.0%} bonds ({C["bond_ticker"]})'] if C["bond_w"] > 1e-9 else [])
    alloc = " &nbsp;·&nbsp; ".join(parts)
    if C["qqq_w"] > 1e-9:
        title = f'{C["qqq_w"]:.0%} / {C["trend_w"]:.0%} / {C["bond_w"]:.0%} diversified portfolio'
    elif C["bond_w"] > 1e-9:
        title = f'{C["trend_w"]:.0%} Trend / {C["bond_w"]:.0%} Bonds portfolio'
    else:
        title = "Diversified Trend sleeve"
    vt_txt = f' &nbsp;·&nbsp; vol-target {C["vol_target"]:.0%}' if C.get("vol_target") else ""
    sc = s.get("scale_now", 1.0)
    scale_line = (f'<p class="sub" style="margin-top:6px">Risk scale: <b>{sc*100:.0f}%</b> invested '
                  + ('&mdash; <span style="color:var(--amber)">de-risked</span> (book vol above target, '
                     'rest held in cash)' if sc < 0.99 else '&mdash; full exposure (book vol at/below target)')
                  + '</p>') if C.get("vol_target") else ""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head><body>
<h1>{title}</h1>
<p class="sub">{alloc} &nbsp;·&nbsp;
{C["band"]:.0%} no-trade band{vt_txt}
&nbsp;|&nbsp; signals as of {s['asof'].date()} &nbsp;·&nbsp;
<span class="pill">${cap:,.0f} account</span>{startpill}</p>
{scale_line}
<div class="cards">{cards}</div>
{perf_section(s)}

{actions_html}
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
    p.add_argument("--fresh", action="store_true",
                   help="Frame as a fresh-start buy list (defaults to next Monday).")
    p.add_argument("--date", default=None, help="Fresh-start date, YYYY-MM-DD.")
    p.add_argument("--track", default=None,
                   help="Ongoing-account inception YYYY-MM-DD: live tracker view (current "
                        "holdings + return since inception), not a fresh buy list.")
    args = p.parse_args()

    print("Building dashboard (fetching data + running strategy)...")
    sd = (dt.date.fromisoformat(args.date) if args.date else next_monday()) if args.fresh else None
    td = dt.date.fromisoformat(args.track) if args.track else None
    ts = sd or td                                     # inception used to rebase the tracker
    s = st.current_state(capital=args.capital, start=args.start,
                         track_start=ts.isoformat() if ts else None)
    with open(args.out, "w") as f:
        f.write(build_html(s, fresh=args.fresh, start_date=sd, track_date=td))
    print(f"Dashboard written to {args.out}  (open it in a browser)")


if __name__ == "__main__":
    main()
