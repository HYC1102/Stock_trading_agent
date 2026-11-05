# Diversified Trend Sleeve — Greenfield Project Specification

## 1. Purpose

Build a reliable, auditable implementation of a diversified, long-or-cash trend-following strategy. The same event-driven portfolio engine must power historical backtests, paper/live portfolio tracking, order generation, and dashboard reporting.

The project must never reconstruct executed history from current signals or newly downloaded prices. Once an order is recorded as filled, its execution price, quantity, fees, and timestamp are immutable.

This is a research and portfolio-tracking tool, not an automated broker or investment-advice product.

## 2. Strategy identity

- **Name:** Diversified Trend Sleeve
- **Version:** 1.0
- **Style:** diversified time-series trend following
- **Direction:** long or cash only; no short positions
- **Leverage:** none
- **Portfolio allocation:** 100% trend strategy
- **Permanent core holdings:** none
- **Unused allocation:** cash
- **Portfolio-level volatility overlay:** none in v1.0
- **Decision frequency:** daily after the US market close
- **Execution assumption:** next eligible market open

Any change to the universe, signal, sizing, rebalance policy, or execution timing creates a new strategy version. Historical live records remain attached to the version that generated them.

## 3. Investable universe

The v1.0 universe contains 21 ETFs:

| Group | Tickers |
|---|---|
| US equity | SPY, QQQ, IWM |
| International and emerging equity | EFA, EEM, FXI |
| Government bonds | TLT, IEF |
| Credit | LQD, HYG |
| Commodities and metals | DBC, GLD, SLV, USO |
| Currency | UUP |
| Real estate | VNQ |
| Sectors and themes | SOXX, XLK, XLE, XBI, BOTZ |

The universe is intentionally fixed for v1.0. Backtests must account for each ETF's actual inception date. An unavailable ETF remains ineligible rather than being backfilled or replaced.

## 4. Market data contract

### Live decisions and dashboard

- Tiingo is the only permitted market-data provider.
- Use split- and dividend-adjusted daily Open, High, Low, and Close values.
- Cache one validated response per completed market session.
- Every required ticker must be present and current through the same expected session.
- If Tiingo is unavailable, stale, incomplete, or rate-limited, fail closed: retain the previously published dashboard and issue no new orders.
- Never silently substitute yfinance or mix providers in one live run.

### Research and backtests

- Tiingo is the canonical provider.
- yfinance may be used only as an independent research replication.
- Each run uses one provider and one frozen snapshot for every variant.
- Store provider, retrieval timestamp, requested inclusive end date, per-ticker coverage, and a SHA-256 content fingerprint.
- Recompute and verify the fingerprint whenever a cached snapshot is loaded.
- Provider-specific date semantics must be normalized. The project-level end date is always inclusive.

## 5. Signal definition

For each ETF independently, calculate a 50-session Donchian channel using only prior sessions:

- `prior_high[D] = max(High[D-50 ... D-1])`
- `prior_low[D] = min(Low[D-50 ... D-1])`
- Enter or remain long when Close on decision date D closes above `prior_high[D]`.
- Exit or remain in cash when Close on D closes below `prior_low[D]`.
- Between the two boundaries, retain the previous long/cash state.

The signal is decided only after date D has closed. It may create an order for the open of D+1, but it must never receive a D close fill.

## 6. Position sizing

Calculate annualized realized volatility from the most recent 60 close-to-close returns available through decision date D:

`annualized_vol = stdev(daily_returns, 60) × sqrt(252)`

The raw weight for an active ETF is:

`raw_weight = min(25%, 2.8% / annualized_vol)`

Rules:

- An inactive ETF has target weight zero.
- An ETF without 60 valid returns is ineligible.
- If aggregate raw weights exceed 100%, scale all active weights proportionally to total 100%.
- Allocation must be independent of ticker or dictionary order.
- Fractional shares are allowed in research and modeled live tracking. A later broker adapter may implement whole-share rounding explicitly.

### v1.0 execution sizing: entry-and-hold

On a new breakout, size the position once using portfolio equity and volatility known at the decision close. Hold that share quantity unchanged until the exit signal. Portfolio growth does not itself create additional purchases.

This is the v1.0 default because identical Tiingo and yfinance snapshots showed materially lower turnover and better net historical return and Sharpe than daily dynamic re-targeting.

### Required research challenger: dynamic 15% band

The backtest must also support a controlled comparison in which active positions are re-targeted from current equity and volatility. Trade only when:

- the position is opening or closing; or
- `abs(current_value - target_value) / target_value > 15%`.

This challenger is not the live default. Its output must be labeled separately and must use the identical data snapshot and assumptions as v1.0.

## 7. Event timing and portfolio accounting

Use one event-driven engine with the following daily sequence:

1. Mark the existing portfolio at date D close.
2. Calculate date D signals and sizing inputs using data available through that close.
3. Create pending orders tagged with decision date D and strategy version.
4. At the next eligible session open D+1, execute sells first and then buys.
5. Apply slippage and fees to each execution.
6. Update cash and share quantities from the executions.
7. Mark the resulting portfolio at D+1 close.

The engine owns cash and shares directly. Do not simulate a theoretical sleeve and then feed its weights into a second portfolio simulation.

Default research costs:

- Commission/fees: 2 basis points per traded notional
- Slippage: 2 basis points per side relative to the adjusted opening price
- Buy fill: `Open × (1 + slippage)`
- Sell fill: `Open × (1 - slippage)`

Cash must never become negative. When simultaneous buys are not fully affordable, scale all buys proportionally after sells, estimated fees, and slippage; do not allocate cash according to ticker iteration order.

## 8. Live account and execution ledger

The dashboard must follow the executed portfolio, not theoretical signals.

Maintain append-only records for:

### Orders

- unique order ID
- strategy version
- decision timestamp and market session
- ticker and side
- requested quantity or notional
- order reason: entry, exit, or rebalance
- status: pending, filled, partially filled, cancelled, or rejected

### Executions

- unique execution ID and order ID
- actual or explicitly modeled fill timestamp
- fill price and quantity
- fees
- data source or broker source
- recorded-at timestamp

### Account state

- starting cash and inception date
- cash movements unrelated to trading, including deposits and withdrawals
- current shares derived from executions
- average cost derived from executions
- realized and unrealized P&L

Recorded executions are immutable. Market-data refreshes may change future decisions but must never rewrite old fills or cost basis.

For the existing modeled account, the configured inception is **28 July 2026** and the initial capital is **$13,500**. These are account settings, not permanent strategy parameters.

## 9. Dashboard meaning

Every number must have one unambiguous definition:

- **Account value:** current cash plus current positions marked at the latest validated Tiingo close.
- **Position amount:** current market value of shares actually held, not original purchase amount and not a theoretical target based on starting capital.
- **Portfolio weight:** current position amount divided by current account value.
- **Average cost:** total remaining execution cost divided by remaining shares.
- **Gain $:** current position value minus remaining cost basis.
- **Gain %:** gain dollars divided by remaining cost basis.
- **Today's actions:** only orders created by the latest decision and still pending for the next open.
- **Target holding:** estimated post-order holding based on the latest close; label it as an estimate because the next opening fill is unknown.

Deposits and withdrawals change cash and account value but are not investment gains. Performance calculations must use time-weighted returns when external cash flows exist.

The dashboard must show the data source, market-data as-of session, latest ledger update, account inception, strategy version, and whether orders are pending or filled.

## 10. Backtest outputs

Each run must export:

- daily equity, cash, and gross exposure
- complete order and execution ledger
- daily signal and target-weight panels
- annual return table
- full-period and declared evaluation-period metrics
- snapshot metadata and fingerprint
- complete configuration used by the run

Required metrics:

- CAGR and total return
- annualized volatility
- Sharpe ratio, with the risk-free assumption stated
- maximum drawdown
- average gross exposure
- trade count
- annual two-sided turnover
- fees, slippage cost, and total modeled trading cost
- correlation to SPY

The benchmark is adjusted buy-and-hold SPY over the identical dates. Report both absolute return and risk; do not describe the sleeve as a market-beating strategy merely because its Sharpe ratio is higher.

## 11. Reliability and bias controls

The implementation must include automated tests for:

- close-D decisions filling only at open D+1
- no dependence on future prices
- Donchian channel using prior sessions only
- 60-session volatility warm-up
- entry, exit, and hysteresis behavior
- both sizing modes
- the 15% band boundary
- fees and slippage reducing returns correctly
- sells funding buys and cash never becoming negative
- order independence across ticker column order
- immutable execution history
- correct average cost after additions and partial sales
- deposits not being counted as gains
- inclusive snapshot end dates
- stale or incomplete live data failing closed
- Tiingo failure never triggering a live yfinance fallback
- deterministic repeat runs from the same snapshot hash

Backtest limitations must be stated prominently:

- the fixed present-day ETF universe introduces selection bias;
- adjusted OHLC data is a modeling convention and cannot reproduce every tradable historical open exactly;
- fractional shares, taxes, market impact, and broker-specific constraints are omitted unless explicitly modeled;
- comparing variants on known history is not a genuine untouched out-of-sample test;
- parameters must not be optimized on the evaluation period and then described as out-of-sample.

## 12. Required validation suite

Run both sizing variants on:

1. one frozen Tiingo snapshot;
2. the same date range from a separate yfinance snapshot as a vendor replication;
3. zero-cost, base-cost, and higher-cost scenarios;
4. the full common history and a declared recent evaluation period beginning 1 January 2022.

Provider results should be close enough that ordinary adjustment differences do not reverse the conclusion. Any material difference must be investigated at the ticker and date level.

The initial reference comparison used 3 January 2017 through 18 August 2026. It favored entry-and-hold sizing in both provider datasets and under all three cost assumptions. These results are a regression reference, not a promised return and not an untouched out-of-sample result.

## 13. Suggested project structure

```text
diversified-trend-sleeve/
├── project.md
├── pyproject.toml
├── src/
│   └── diversified_trend/
│       ├── config.py
│       ├── data.py
│       ├── signals.py
│       ├── sizing.py
│       ├── engine.py
│       ├── ledger.py
│       ├── backtest.py
│       ├── live.py
│       └── dashboard.py
├── tests/
│   ├── fixtures/
│   ├── test_data.py
│   ├── test_signals.py
│   ├── test_sizing.py
│   ├── test_engine.py
│   ├── test_ledger.py
│   └── test_backtest_regression.py
├── data/
│   ├── snapshots/        # ignored; immutable by hash
│   └── ledger/           # backed up and never regenerated from prices
└── outputs/              # ignored reports and dashboard artifacts
```

Dependencies must flow in one direction: data and immutable configuration feed signals and sizing; those feed the portfolio engine; the backtest and live runner call that same engine; the dashboard reads engine and ledger outputs without recalculating portfolio history.

## 14. Acceptance criteria

The greenfield project is ready for paper/live use only when:

- all reliability tests pass;
- a repeat run from the same snapshot produces identical trades and equity;
- Tiingo and yfinance research replications have been compared and explained;
- live generation fails closed on stale or missing Tiingo data;
- a deposit test proves account growth is separated from strategy return;
- dashboard amounts tie exactly to cash plus ledger-derived positions;
- every displayed gain ties to immutable fills;
- pending orders are clearly separated from completed executions;
- the strategy version and full configuration are stored with every run.

No component may silently guess, backfill, change providers, or rewrite an execution in order to produce a dashboard.
