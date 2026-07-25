# 40 / 40 / 20 Diversified Portfolio

A simple, robust, hand-runnable portfolio strategy:

| Sleeve | Weight | What it is |
|---|--:|---|
| **QQQ** | 40% | US growth engine (held plain — no vol overlay) |
| **Diversified trend** | 40% | Managed-futures-style trend across 18 asset-class ETFs |
| **Bonds (IEF)** | 20% | Ballast |

Rebalanced with a **15% no-trade band** — a position is only traded when its
weight drifts more than 15% from target (trend entries/exits always execute).
That keeps trading to **~35–40 trades/year** — a monthly chore, not a daily one.
Signals are decided on the close and filled at the next open (no look-ahead), and
all returns are total returns (dividends and bond coupons included).

## Why this design

Built and validated across a long research process (see git history), the honest
findings were:
- **You can't beat a broad index by timing it** — every single-asset timing edge
  died out of sample.
- **The only durable edge is diversification** — combining uncorrelated return
  streams (equity + trend + bonds) beats any of them alone on a risk-adjusted basis.
- **Trend-following is a genuine diversifier** (low correlation to stocks, crisis
  alpha), not a standalone winner.
- **A no-trade band removes ~97% of the trading noise with no performance cost.**

## What to expect (honest)

Over a full cycle (2006–2026) this delivered roughly **SPY-like returns with about
half the drawdown** — Sharpe ~1.0 vs SPY's ~0.7, max drawdown ~−22% vs −55%. But:
- It **lags straight stocks in bull markets** (it made ~half of SPY over 2020–2026).
- Its edge shows up in **crashes** (2008: −7% vs SPY −37%).
- It's a *smoother ride to a similar destination* — not a market-beater. Backtests
  overstate; expect ~6–9% real annual returns and be ready for a −20-something% drawdown.

## Universe (the trend sleeve, 18 ETFs)

Equities (SPY, QQQ, IWM, EFA, EEM) · bonds (TLT, IEF) · credit (LQD, HYG) ·
commodities/metals (DBC, GLD, SLV, USO) · dollar (UUP) · REITs (VNQ) ·
sectors (SOXX, XLK, XLE).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python strategy.py --capital 23000        # print current book, trades, exposure
python dashboard.py --capital 23000        # -> dashboard.html (open in a browser)
```

- `strategy.py` — the self-contained strategy (signal, sizing, band rebalancing,
  backtest, current state). Import `current_state()` / `backtest()` to build on it.
- `dashboard.py` — generates a self-contained `dashboard.html` with two panels:
  **Today's actions** (target holdings + the trades to place) and
  **Risk & exposure** (sleeve mix, asset-class breakdown, current risk stats).

## Running it for real

1. **Weekly (or monthly):** run `python dashboard.py` (or `strategy.py`).
2. **Buy the target book** if starting fresh; otherwise place only the trades it
   lists (the positions that broke the 15% band).
3. That's it — no daily attention needed.

Config (weights, band, universe) lives in `CONFIG` / `UNIVERSE` at the top of
`strategy.py`.

## Caveats

Backtests assume the future rhymes with the past — it won't exactly. Live results
run below backtest. Bonds-and-stocks-fall-together shocks (like 2022) are this
portfolio's weak spot. **This is a research tool, not investment advice.**
