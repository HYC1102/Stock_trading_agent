"""Deterministic event-driven engine for the Diversified Trend Sleeve.

Decisions use information available at session D's close. Orders fill at the
next session's open, after which the portfolio is marked at that session's
close. The engine owns cash and shares directly; it does not simulate a sleeve
and then run a second portfolio against the sleeve's weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PPY = 252
SizingMode = Literal["dynamic_band", "entry_hold"]


@dataclass(frozen=True)
class EngineConfig:
    channel: int = 50
    vol_window: int = 60
    asset_budget: float = 0.028
    max_name: float = 0.25
    band: float = 0.15
    fee_bps: float = 2.0
    slippage_bps: float = 2.0
    sizing_mode: SizingMode = "dynamic_band"

    def validate(self) -> None:
        if self.sizing_mode not in ("dynamic_band", "entry_hold"):
            raise ValueError(f"Unknown sizing mode: {self.sizing_mode}")
        if self.channel < 2 or self.vol_window < 2:
            raise ValueError("Signal and volatility windows must be at least 2")
        if not 0 < self.asset_budget <= 1:
            raise ValueError("asset_budget must be in (0, 1]")
        if not 0 < self.max_name <= 1:
            raise ValueError("max_name must be in (0, 1]")
        if not 0 <= self.band <= 1:
            raise ValueError("band must be in [0, 1]")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Trading costs cannot be negative")


@dataclass
class BacktestResult:
    equity: pd.Series
    cash: pd.Series
    exposure: pd.Series
    target_weights: pd.DataFrame
    trades: pd.DataFrame
    final_shares: pd.Series
    config: EngineConfig


def donchian_signal(raw: pd.DataFrame, channel: int) -> pd.Series:
    """Long/flat state using prior-channel highs and lows."""
    high = raw["High"].rolling(channel).max().shift(1)
    low = raw["Low"].rolling(channel).min().shift(1)
    close = raw["Close"]
    state = []
    held = False
    for date in raw.index:
        if pd.notna(high.loc[date]) and close.loc[date] > high.loc[date]:
            held = True
        elif pd.notna(low.loc[date]) and close.loc[date] < low.loc[date]:
            held = False
        state.append(float(held))
    return pd.Series(state, index=raw.index, name="signal")


def build_panels(prices: dict[str, pd.DataFrame], config: EngineConfig):
    """Build aligned open, close, signal and ex-ante volatility panels."""
    config.validate()
    if not prices:
        raise ValueError("No price histories supplied")

    opens, closes, signals, vols = {}, {}, {}, {}
    for ticker, frame in prices.items():
        missing = {"Open", "High", "Low", "Close"} - set(frame.columns)
        if missing:
            raise ValueError(f"{ticker} is missing columns: {sorted(missing)}")
        raw = frame[["Open", "High", "Low", "Close"]].copy()
        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        raw = raw[~raw.index.duplicated(keep="last")].sort_index().dropna()
        if raw.empty or (raw <= 0).any().any():
            raise ValueError(f"{ticker} has empty or non-positive OHLC history")
        opens[ticker] = raw["Open"]
        closes[ticker] = raw["Close"]
        signals[ticker] = donchian_signal(raw, config.channel)
        # The close on decision day D is known before the D+1 open fill, so the
        # volatility estimate legitimately includes D's close.
        vols[ticker] = raw["Close"].pct_change().rolling(config.vol_window).std() * np.sqrt(PPY)

    index = pd.DatetimeIndex(sorted(set().union(*(s.index for s in closes.values()))))
    return (
        pd.DataFrame(opens).reindex(index),
        pd.DataFrame(closes).reindex(index),
        pd.DataFrame(signals).reindex(index),
        pd.DataFrame(vols).reindex(index),
    )


def desired_weights(signal: pd.Series, vol: pd.Series, config: EngineConfig) -> pd.Series:
    """Order-independent inverse-volatility targets with a 100% gross cap."""
    active = signal.fillna(0).gt(0.5) & vol.notna() & vol.gt(0)
    weights = pd.Series(0.0, index=signal.index, dtype=float)
    weights.loc[active] = (config.asset_budget / vol.loc[active]).clip(upper=config.max_name)
    total = float(weights.sum())
    if total > 1.0:
        weights /= total
    return weights


def _mark(shares: pd.Series, prices: pd.Series, cash: float, date, label: str) -> tuple[float, float]:
    held = shares[shares.abs() > 1e-12]
    if not held.empty:
        marks = prices.reindex(held.index)
        missing = marks[marks.isna()].index.tolist()
        if missing:
            raise ValueError(f"Missing {label} prices on {pd.Timestamp(date).date()}: {missing}")
        invested = float((held * marks).sum())
    else:
        invested = 0.0
    return cash + invested, invested


def run_backtest(open_prices: pd.DataFrame, close_prices: pd.DataFrame,
                 signals: pd.DataFrame, volatility: pd.DataFrame,
                 capital: float, config: EngineConfig) -> BacktestResult:
    """Run a close-decision, next-open-fill backtest with explicit accounting."""
    config.validate()
    if capital <= 0:
        raise ValueError("capital must be positive")
    tickers = list(close_prices.columns)
    if not tickers or set(open_prices.columns) != set(tickers):
        raise ValueError("Open and close panels must have the same tickers")
    for panel in (signals, volatility):
        if set(panel.columns) != set(tickers):
            raise ValueError("All panels must have the same tickers")

    index = close_prices.index.intersection(open_prices.index).sort_values()
    if len(index) < 2:
        raise ValueError("At least two sessions are required")
    open_prices = open_prices.reindex(index=index, columns=tickers)
    close_prices = close_prices.reindex(index=index, columns=tickers)
    signals = signals.reindex(index=index, columns=tickers)
    volatility = volatility.reindex(index=index, columns=tickers)

    shares = pd.Series(0.0, index=tickers)
    cash = float(capital)
    fee_rate = config.fee_bps / 10_000
    slip_rate = config.slippage_bps / 10_000
    equity_log = {index[0]: float(capital)}
    cash_log = {index[0]: cash}
    exposure_log = {index[0]: 0.0}
    target_log = {}
    trade_log: list[dict] = []

    for i in range(len(index) - 1):
        decision_date, fill_date = index[i], index[i + 1]
        target_w = desired_weights(signals.loc[decision_date],
                                   volatility.loc[decision_date], config)
        target_log[decision_date] = target_w
        open_row = open_prices.loc[fill_date]
        close_row = close_prices.loc[fill_date]
        equity_open, _ = _mark(shares, open_row, cash, fill_date, "open")

        if config.sizing_mode == "dynamic_band":
            target_value = target_w * equity_open
            current_value = shares * open_row.reindex(tickers)
            desired_shares = shares.copy()
            for ticker in tickers:
                target = float(target_value[ticker])
                current = float(current_value[ticker]) if pd.notna(current_value[ticker]) else np.nan
                if target <= 1e-10:
                    if shares[ticker] > 1e-12:
                        desired_shares[ticker] = 0.0
                elif shares[ticker] <= 1e-12:
                    if pd.isna(open_row[ticker]):
                        raise ValueError(f"Missing fill price for opening {ticker} on {fill_date.date()}")
                    desired_shares[ticker] = target / float(open_row[ticker])
                else:
                    if not np.isfinite(current):
                        raise ValueError(f"Missing fill price for held {ticker} on {fill_date.date()}")
                    if abs(current - target) / target > config.band:
                        desired_shares[ticker] = target / float(open_row[ticker])
        else:  # entry_hold
            desired_shares = shares.copy()
            for ticker in tickers:
                if target_w[ticker] <= 1e-10 and shares[ticker] > 1e-12:
                    desired_shares[ticker] = 0.0
                elif target_w[ticker] > 1e-10 and shares[ticker] <= 1e-12:
                    if pd.isna(open_row[ticker]):
                        raise ValueError(f"Missing fill price for opening {ticker} on {fill_date.date()}")
                    desired_shares[ticker] = target_w[ticker] * equity_open / float(open_row[ticker])

        delta = desired_shares - shares

        # Sells execute first, releasing cash for entries/rebalances.
        for ticker in tickers:
            qty = -float(delta[ticker]) if delta[ticker] < -1e-12 else 0.0
            if qty <= 0:
                continue
            reference = float(open_row[ticker])
            fill = reference * (1 - slip_rate)
            notional = qty * fill
            fee = notional * fee_rate
            cash += notional - fee
            shares[ticker] -= qty
            reason = "exit" if desired_shares[ticker] <= 1e-12 else "rebalance"
            trade_log.append(dict(decision_date=decision_date, fill_date=fill_date,
                                  ticker=ticker, side="SELL", reason=reason,
                                  shares=qty, reference_open=reference, fill_price=fill,
                                  notional=notional, fee=fee,
                                  slippage_cost=qty * reference * slip_rate,
                                  target_weight=float(target_w[ticker]),
                                  equity_at_open=equity_open))

        # Scale all buys together if costs or cash constraints make the complete
        # target unaffordable. This removes ticker/dictionary order dependence.
        buy_qty = delta.clip(lower=0.0)
        required = 0.0
        for ticker, qty in buy_qty.items():
            if qty > 1e-12:
                reference = float(open_row[ticker])
                fill = reference * (1 + slip_rate)
                required += float(qty) * fill * (1 + fee_rate)
        buy_scale = min(1.0, cash / required) if required > 0 else 0.0

        for ticker in tickers:
            qty = float(buy_qty[ticker]) * buy_scale
            if qty <= 1e-12:
                continue
            reference = float(open_row[ticker])
            fill = reference * (1 + slip_rate)
            notional = qty * fill
            fee = notional * fee_rate
            cash -= notional + fee
            shares[ticker] += qty
            reason = "entry" if shares[ticker] - qty <= 1e-12 else "rebalance"
            trade_log.append(dict(decision_date=decision_date, fill_date=fill_date,
                                  ticker=ticker, side="BUY", reason=reason,
                                  shares=qty, reference_open=reference, fill_price=fill,
                                  notional=notional, fee=fee,
                                  slippage_cost=qty * reference * slip_rate,
                                  target_weight=float(target_w[ticker]),
                                  equity_at_open=equity_open))

        if cash < -1e-7:
            raise AssertionError(f"Cash became negative on {fill_date.date()}: {cash}")
        cash = max(0.0, cash)
        equity_close, invested = _mark(shares, close_row, cash, fill_date, "close")
        equity_log[fill_date] = equity_close
        cash_log[fill_date] = cash
        exposure_log[fill_date] = invested / equity_close if equity_close > 0 else 0.0

    trade_columns = ["decision_date", "fill_date", "ticker", "side", "reason",
                     "shares", "reference_open", "fill_price", "notional", "fee",
                     "slippage_cost", "target_weight", "equity_at_open"]
    trades = pd.DataFrame(trade_log, columns=trade_columns)
    targets = pd.DataFrame.from_dict(target_log, orient="index").reindex(columns=tickers)
    return BacktestResult(
        equity=pd.Series(equity_log).sort_index().rename("equity"),
        cash=pd.Series(cash_log).sort_index().rename("cash"),
        exposure=pd.Series(exposure_log).sort_index().rename("gross_exposure"),
        target_weights=targets,
        trades=trades,
        final_shares=shares.copy(),
        config=config,
    )


def performance_metrics(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict:
    equity = equity.dropna()
    returns = equity.pct_change().dropna()
    if len(equity) < 2 or returns.empty:
        return dict(cagr=np.nan, sharpe=np.nan, vol=np.nan, max_drawdown=np.nan,
                    total_return=np.nan, trades=0, fees=0.0, slippage=0.0,
                    trading_costs=0.0, annual_turnover=np.nan)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / PPY)
    drawdown = equity / equity.cummax() - 1
    std = returns.std()
    fees = 0.0 if trades is None or trades.empty else float(trades["fee"].sum())
    slippage = (0.0 if trades is None or trades.empty
                else float(trades["slippage_cost"].sum()))
    turnover = np.nan
    if trades is not None and not trades.empty:
        turnover = float((trades["notional"] / trades["equity_at_open"]).sum() / years)
    return dict(
        cagr=(equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1,
        sharpe=returns.mean() / std * np.sqrt(PPY) if std > 0 else np.nan,
        vol=std * np.sqrt(PPY),
        max_drawdown=float(drawdown.min()),
        total_return=equity.iloc[-1] / equity.iloc[0] - 1,
        trades=0 if trades is None else int(len(trades)),
        fees=fees,
        slippage=slippage,
        trading_costs=fees + slippage,
        annual_turnover=turnover,
    )
