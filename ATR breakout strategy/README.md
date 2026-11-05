# ATR Breakout Strategy

A self-contained, slot-based swing trader on liquid US large caps. Split out of
the main repo so it runs independently of the Diversified Trend Sleeve
(`../strategy.py`, `../dashboard.py`) — nothing here imports that code.

All commands below assume you are **inside this folder** (paths are relative to
the working directory).

## Files

| File | Role |
|---|---|
| `breakout_sentiment.py` | The strategy: universe, ATR-band entry, ranking, event-driven backtest |
| `paper_trade.py` | Persistent forward paper-trade (state in `data/paper_breakout.json`) |
| `prices.py` | Tiingo-first price adapter with a yfinance fallback |
| `breakout_dashboard.py` | Advances the paper account and renders `breakout_dashboard.html` |
| `data/` | Price caches, weekly universe snapshots, sentiment cache, paper-trade state |

## How it works

**Universe** — S&P 500 + Nasdaq-100 + liquid extras, narrowed to the top
**223 names by 60-day average dollar volume**, rebuilt weekly.

**Entry** (`entry="atr"`) — today's *High* pierces yesterday's Keltner-style
upper band, `EMA(20) + 3.0 × ATR(20)`. A `donchian` mode (close above the prior
20-day high) is available. New positions only open when **SPY > its 200-day MA**;
exits fire in any regime.

**Ranking** — more breakouts fire than there are slots, so candidates are scored
0–100 and the top ones win:
`combined = 0.6 × momentum + 0.4 × sentiment`, where momentum is the percentile
rank of 63-day trailing return and sentiment is VADER over recent headlines
(50 = neutral / no news). **The backtest ranks on momentum only** — sentiment is
live/dashboard-side.

**Sizing** — **5 concurrent slots**, each entry takes 1/5 of equity and is then
left alone (no rebalancing). Empty slots sit in cash.

**Exit** — whichever hits first: a trailing ATR stop at
`highest close since entry − 1.5 × ATR(14)` (modelled as a resting intraday
order), or a **10-day-low** exit. Take-profit, time-stop and hard % stop exist
but are off.

Signals fire on the close and fill at the **next open** — no look-ahead. Costs
are 2 bps per side.

## Running

```bash
pip install -r requirements.txt

python breakout_sentiment.py --backtest        # historical simulation
python breakout_dashboard.py --capital 8800    # advance the paper account -> breakout_dashboard.html
```

`prices.py` reads a Tiingo token from `$TIINGO_API_KEY`, or from a gitignored
`.tiingo_token` / `data/tiingo_token.txt`. Without one it falls back to yfinance.

## Honest-data caveats

* The universe uses **today's** index membership → survivorship bias, so
  backtested returns are optimistic.
* The backtest's momentum proxy is not sentiment; it validates the
  breakout/slot/stop mechanics, not the sentiment overlay.
