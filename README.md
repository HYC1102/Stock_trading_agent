# Trend & Breakout Strategies

Two hand-runnable, honestly-validated strategies with self-contained dashboards
that can auto-publish to GitHub Pages. Everything uses honest next-open fills
(signals decided on the close, filled at the next open — no look-ahead) and
total-return data (dividends and coupons included).

| # | Strategy | File | Dashboard |
|---|---|---|---|
| 1 | **Diversified Trend sleeve** | `strategy.py` | `dashboard.py` → `dashboard.html` |
| 2 | **Breakout momentum (3-slot swing)** | `breakout_sentiment.py` + `paper_trade.py` | `breakout_dashboard.py` → `breakout_dashboard.html` |

`index.html` tabs between the two. A daily GitHub Actions workflow regenerates and
publishes both to Pages (see [DEPLOY.md](DEPLOY.md)).

---

## 1 · Diversified Trend sleeve  *(the core strategy)*

Managed-futures-style trend-following across **18 asset-class ETFs**, inverse-vol
sized, holding only the assets currently in an uptrend:

- **Signal** — 50-day Donchian breakout (long above the N-day high, flat below the
  N-day low), per asset. *(An ATR-breakout variant is available via
  `CONFIG["signal"]="atr"`; testing showed it does not beat Donchian, so Donchian
  is the default.)*
- **Sizing** — inverse-volatility (risk-parity-style), unlevered, capped per name.
- **Rebalance** — a **15% no-trade band** keeps trading to ~35 trades/year.

**Honest performance (2010–2026):** ~**8.9% CAGR**, Sharpe ~**0.83–0.97**, max
drawdown ~**−21%**, and — notably — only **~0.50 correlation to SPY**. It was the
one configuration in this whole project that **beat SPY on risk-adjusted terms
out-of-sample** (2018–2026 Sharpe 0.83 vs SPY 0.87, at far lower correlation), and
its edge held up in walk-forward. It gives up bull-market upside for a smoother,
diversifying, crisis-resilient return stream — a genuine diversifier, not a
market-beater.

The strategy is configurable (`CONFIG` in `strategy.py`): the shipped default is
**100% trend** (`trend_w=1.0`), but it also supports a fixed bond sleeve or a
vol-targeted QQQ core. Adding a fixed 1/3 bonds was tested and *did not* improve
risk-adjusted returns out-of-sample (and hurt in 2022) — the trend sleeve already
holds bonds/credit dynamically when they trend, so the default is bond-free.

**Universe (18 ETFs):** equities (SPY, QQQ, IWM, EFA, EEM) · bonds (TLT, IEF) ·
credit (LQD, HYG) · commodities/metals (DBC, GLD, SLV, USO) · dollar (UUP) ·
REITs (VNQ) · sectors (SOXX, XLK, XLE).

---

## 2 · Breakout momentum  *(high-octane satellite / paper-trade)*

A 3-slot swing trader on the most liquid US stocks: each day, scan the **top-223
by dollar volume** (rebuilt weekly) for 20-day breakouts, **rank by momentum +
real news sentiment** (yfinance headlines → VADER), hold the top 3 with a
**1.5-ATR trailing stop + 10-day-low exit** and a **200-day SPY regime filter**.

`paper_trade.py` runs it as a **persistent forward paper-trade** (state in
`data/paper_breakout.json`, every trade + its sentiment logged to CSV), starting
from a fixed date and advancing once per run.

⚠️ **Honest caveats:** the backtest is **survivorship-biased** (today's index
members only) and **regime-dependent** — impressive in momentum regimes, weaker
otherwise, with deep (−40%+) drawdowns. Treat it as a research/paper-trade
experiment, not a validated edge.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python strategy.py --capital 13500                     # trend book, trades, exposure
python dashboard.py --fresh --capital 13500            # -> dashboard.html (trend sleeve)
python breakout_dashboard.py --capital 8800            # -> breakout_dashboard.html
```

Then open `index.html` (or serve the folder: `python -m http.server 8000`).

## Deploy to GitHub Pages

`.github/workflows/paper-trade.yml` runs daily (and on push to `main`), regenerates
both dashboards, commits the forward paper-trade log back to the repo, and publishes
to Pages. One-time setup (repo → Pages source = *GitHub Actions*, Actions write
permission) is in **[DEPLOY.md](DEPLOY.md)**.

## Caveats

Backtests assume the future rhymes with the past — it won't exactly, and live
results run below backtest. **These are research tools, not investment advice.**
