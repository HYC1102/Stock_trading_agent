# Donchian Channel Trading Strategy

A simple, single-stock **Donchian channel breakout** backtester (long-or-flat,
"Turtle"-style). Tested on TSLA.

## Strategy

- **Donchian channel** = the highest high / lowest low over a lookback window.
- **Entry:** go long when the close breaks **above** the prior `entry`-day high
  (default 20 days).
- **Exit:** go flat when the close breaks **below** the prior `exit`-day low
  (default 10 days).
- Long or flat only — no shorting, no leverage.

Signals are computed on the close and **executed on the next bar's open** to
avoid look-ahead bias. A per-side transaction cost (default 2 bps) is charged
on each entry and exit.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Default: TSLA, 20/10 windows, since 2015, plus a saved chart
python donchian.py --ticker TSLA --start 2015-01-01 --plot

# Tune the channel windows / costs
python donchian.py --ticker TSLA --entry 55 --exit 20 --cost-bps 5
```

### Options

| Flag         | Default        | Meaning                                |
|--------------|----------------|----------------------------------------|
| `--ticker`   | `TSLA`         | Any Yahoo Finance symbol               |
| `--start`    | `2015-01-01`   | History start date                     |
| `--end`      | today          | History end date                       |
| `--entry`    | `20`           | Entry channel lookback (days)          |
| `--exit`     | `10`           | Exit channel lookback (days)           |
| `--cost-bps` | `2.0`          | Per-side transaction cost (bps)        |
| `--plot`     | off            | Save a price + equity-curve chart      |
| `--plot-path`| `backtest.png` | Chart output path                      |

## Output

Prints a performance report (total return, CAGR, Sharpe, max drawdown, win
rate, average win/loss, time in market, and buy-&-hold comparison), the last
few trades, and optionally saves a chart.

## ATR stops & volatility sizing (`atr_strategy.py`)

Adds the two risk-management pieces of the full Turtle system:

- **Trailing ATR stop (chandelier exit):** while long, exit if the close falls
  more than `stop_mult` × ATR below the highest close since entry — on top of
  the normal Donchian exit.
- **Volatility-based sizing:** size each position so a stop-out costs a fixed
  `risk_frac` of equity: `weight = risk_frac / (stop_mult × ATR / price)`,
  capped at `max_weight`. Volatile regimes → smaller position.

```bash
python atr_strategy.py --ticker TSLA --plot
python atr_strategy.py --risk-frac 0.05 --stop-mult 2 --max-weight 2 --plot
```

Key knobs: `--stop-mult` (stop distance in ATRs, default 2), `--risk-frac`
(equity risked per trade, default 0.01), `--max-weight` (position cap, default
1.0 = no leverage), `--atr-window` (default 20). The report compares the sized
strategy against the base all-in Donchian on the same data.

Because vol-targeting scales the whole position linearly, `risk_frac` moves
you *along* a fixed risk/return line (Sharpe stays constant until the leverage
cap binds) — it sets how much risk you take, not the quality of the signal.

## Cross-sectional / basket test (`basket.py`)

Guards against the biggest overfitting risk: that the strategy only "works" on
TSLA (a 100x trending outlier). Runs the **un-tuned** default 20/10 across a
diverse 16-name universe — trending winners, indices, choppy cyclicals, and
outright decliners — and compares each against buy & hold, plus an equal-weight
basket curve. No parameter is tuned per name, so there is no data-snooping.

```bash
python basket.py --plot                    # base engine
python basket.py --engine atr --plot       # ATR-sized version
python basket.py --tickers SPY,KO,XOM,BA   # custom universe
```

On the default universe (2015–2026) the edge generalises: the strategy improved
risk-adjusted return (Sharpe) on **16/16 names** and cut max drawdown on
**16/16**, with the largest edges on decliners/choppy names — exactly where a
trend filter should help by sidestepping large drawdowns. See caveats below re:
the basket Sharpe (diversification is optimistic; correlations spike in crashes).

## Notes / caveats

- Uses **split- and dividend-adjusted** prices (`auto_adjust=True`).
- This is a research backtest, **not** trading advice or a live trading system.
  It ignores slippage beyond the flat bps cost, position sizing, and taxes.
- A trend strategy on a single high-beta name (TSLA) will show low win rates
  with a few large winners — results are highly sensitive to the window
  parameters and the sample period.
