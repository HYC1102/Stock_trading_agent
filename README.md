# Donchian Channel Trading Strategy

A simple, single-stock **Donchian channel breakout** backtester (long-or-flat,
"Turtle"-style). Tested on TSLA.

> **Preferred single-stock strategy.** Basic Donchian (20/10) + a 2-ATR trailing
> stop + 1% volatility-based sizing — i.e. the **default config of
> `atr_strategy.py`**. Run it with:
> ```bash
> python atr_strategy.py --ticker TSLA          # or any symbol
> ```
> On TSLA (2015–2026, honest fills): **Sharpe 0.88, max drawdown −5.9%** vs
> buy & hold's 0.74 / −75%. It is chosen as a **defensive drawdown-dampener**,
> not an alpha source (see honest note below). At 1% risk it holds ~13% average
> weight (CAGR 3.7%); scale `--risk-frac` to deploy more capital — the Sharpe is
> unchanged by leverage, only the risk level moves.

> **Honest results note.** After fixing a one-day look-ahead bug in the return
> calc (see `git log`), the single-stock breakout showed **no risk-adjusted edge**
> — on 30 names it beat buy & hold's Sharpe on only 4/30, and the days it holds
> long earn *less* than an average day. Its only real value is drawdown reduction.
> The one approach whose honest numbers held up is the **diversified multi-asset
> trend portfolio** (`trend_portfolio.py`) — not as a standalone winner, but as a
> **diversifier**: adding 30% of it to a 60/40 lifted the blend's Sharpe from
> 1.02 to 1.09 and cut max drawdown from -22% to -18%. See that section below.

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

### Optional: trend filter (#4) and pyramiding (#5)

Two opt-in extensions, both **off by default** so the results above are
unchanged:

- `--trend-ma 200` — only open longs when the close is above its 200-day SMA.
- `--max-units 4 --pyramid-atr 0.5` — add a unit every 0.5 × ATR of advance
  (Turtle-style), up to 4 units.

```bash
python atr_strategy.py --max-units 4 --pyramid-atr 0.5   # pyramiding on
python atr_strategy.py --trend-ma 200                    # trend filter on
```

**What the data said** (basket of 16 names, per-trade risk held constant):

| config | median CAGR | median Sharpe | median max DD |
|--------|-------------|---------------|---------------|
| baseline | 10.8% | 1.52 | −5.6% |
| + pyramiding (#5) | **18.3%** | 1.37 | −13.0% |
| + trend filter (#4) | 8.7% | 1.29 | −5.4% |

- **Pyramiding (#5)** is a *profit lever*: it roughly doubled CAGR (≈10× total
  return on TSLA) but proportionally increased volatility and drawdown and
  slightly lowered Sharpe. Enable it if you want upside and can take the
  drawdown; it does not improve risk-adjusted return.
- **Trend filter (#4)** *hurt* on 15 of 16 names here. The ATR trailing stop
  already handles downside, so a 200-day entry gate is redundant — it mainly
  blocks good entries. Left in as an option, but **not recommended in this
  configuration** (it may help in a variant without a tight ATR stop).

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

## Diversified trend portfolio (`trend_portfolio.py`) — the real-edge module

The single-stock work established that the breakout signal has no standalone
edge. Trend-following's documented value comes from **diversification across
uncorrelated asset classes** (managed-futures style), so this module applies the
same trend signal to a fixed universe of asset-class ETFs (equities, rates,
credit, commodities, gold, FX, REITs), sized inverse-vol and scaled to a target
portfolio volatility. ETFs rarely delist to zero, so the benchmark is far less
survivorship-biased than a single-stock universe.

```bash
python trend_portfolio.py --plot                 # long-only, 50-day, 10% vol
python trend_portfolio.py --long-short --channel 100
```

**Honest results (2010–2026, next-open fills):**

| | Trend port | 60/40 | SPY |
|---|--:|--:|--:|
| CAGR | 9.1% | 9.8% | 14.1% |
| Volatility | 10.6% | 9.6% | 16.5% |
| Sharpe | 0.87 | 1.02 | 0.88 |
| Max drawdown | −20% | −22% | −32% |

Standalone it does **not** beat 60/40 — trend-following had a poor 2010s. Its
value is as a **diversifier** (correlation ~0.5 to 60/40; it lost only −4.7% in
2022 while 60/40 fell −15.7%). Adding it to a 60/40 improves the blend:

| Blend | Sharpe | Max DD |
|---|--:|--:|
| 100% 60/40 | 1.02 | −22.3% |
| 70% 60/40 + 30% trend | **1.09** | **−18.4%** |

That ~0.07 Sharpe gain and 4-point drawdown reduction is modest but **real and
defensible** — the only such result in this repo. Caveats: one regime (2010–26),
a 15-ETF proxy for real managed futures, and the long/short variant did worse
(Sharpe 0.53) as the short side bled in the bull market.

## Notes / caveats

- Uses **split- and dividend-adjusted** prices (`auto_adjust=True`).
- This is a research backtest, **not** trading advice or a live trading system.
  It ignores slippage beyond the flat bps cost, position sizing, and taxes.
- A trend strategy on a single high-beta name (TSLA) will show low win rates
  with a few large winners — results are highly sensitive to the window
  parameters and the sample period.
