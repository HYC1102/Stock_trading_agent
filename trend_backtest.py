"""Reproducible comparison runner for Diversified Trend Sleeve sizing modes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

import strategy
from trend_engine import EngineConfig, build_panels, performance_metrics, run_backtest


def _snapshot_hash(prices: dict[str, pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for ticker in sorted(prices):
        digest.update(ticker.encode())
        frame = prices[ticker][["Open", "High", "Low", "Close"]]
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _validate_snapshot(prices: dict[str, pd.DataFrame], through: str) -> dict:
    """Fail closed on incomplete, malformed, or mixed-universe data."""
    missing = sorted(set(strategy.UNIVERSE) - set(prices))
    extra = sorted(set(prices) - set(strategy.UNIVERSE))
    if missing or extra:
        raise ValueError(f"Snapshot universe mismatch; missing={missing}, extra={extra}")
    coverage = {}
    required = pd.Timestamp(through)
    for ticker, frame in prices.items():
        ohlc = frame[["Open", "High", "Low", "Close"]]
        if frame.empty:
            raise ValueError(f"Snapshot contains no rows for {ticker}")
        if frame.index.has_duplicates:
            raise ValueError(f"Snapshot contains duplicate dates for {ticker}")
        if ohlc.isna().any().any():
            raise ValueError(f"Snapshot contains missing OHLC values for {ticker}")
        if (ohlc <= 0).any().any():
            raise ValueError(f"Snapshot contains non-positive OHLC values for {ticker}")
        latest = pd.Timestamp(frame.index.max()).tz_localize(None).normalize()
        if latest < required:
            raise ValueError(
                f"Snapshot is incomplete: {ticker} ends {latest.date()}, "
                f"before requested {required.date()}")
        coverage[ticker] = dict(first=str(pd.Timestamp(frame.index.min()).date()),
                                last=str(latest.date()), rows=int(len(frame)))
    return coverage


def load_snapshot(provider: str, fetch_start: str, through: str,
                  refresh: bool = False) -> tuple[dict[str, pd.DataFrame], str, Path, dict]:
    """Load or create one immutable-in-run data snapshot for every variant."""
    safe_end = through.replace("/", "-")
    path = Path("data") / f"backtest_v2_{provider}_{fetch_start}_{safe_end}.pkl"
    if path.exists() and not refresh:
        payload = pd.read_pickle(path)
        prices = payload["prices"]
        fingerprint = _snapshot_hash(prices)
        if fingerprint != payload.get("sha256"):
            raise ValueError(f"Cached snapshot hash mismatch: {path}")
        coverage = _validate_snapshot(prices, through)
        return prices, fingerprint, path, coverage

    prices = {}
    # yfinance's end is exclusive; Tiingo's endDate is inclusive.
    provider_end = through
    if provider == "yfinance":
        provider_end = (pd.Timestamp(through) + pd.Timedelta(days=1)).date().isoformat()
    for ticker in strategy.UNIVERSE:
        print(f"  fetching {ticker} from {provider}...")
        prices[ticker] = strategy.load_prices(
            ticker, fetch_start, provider_end, provider=provider)
    coverage = _validate_snapshot(prices, through)
    fingerprint = _snapshot_hash(prices)
    payload = dict(schema_version=2, provider=provider, fetch_start=fetch_start,
                   through=through, provider_end=provider_end,
                   created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                   sha256=fingerprint, coverage=coverage, prices=prices)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, path)
    return prices, fingerprint, path, coverage


def _period_metrics(result, start: str | None = None) -> dict:
    equity = result.equity if start is None else result.equity.loc[pd.Timestamp(start):]
    trades = result.trades
    if start is not None and not trades.empty:
        trades = trades[pd.to_datetime(trades["fill_date"]) >= pd.Timestamp(start)]
    return performance_metrics(equity, trades)


def _benchmark(close: pd.Series, capital: float, index: pd.Index) -> pd.Series:
    values = close.reindex(index).ffill().dropna()
    if values.empty:
        raise ValueError("SPY benchmark has no prices over the backtest period")
    return capital * values / values.iloc[0]


def compare(prices: dict[str, pd.DataFrame], start: str, capital: float,
            fee_bps: float, slippage_bps: float, out_dir: str | None = None) -> dict:
    base = EngineConfig(fee_bps=fee_bps, slippage_bps=slippage_bps)
    open_, close, signal, vol = build_panels(prices, base)
    start_ts = pd.Timestamp(start)
    panels = [p.loc[start_ts:] for p in (open_, close, signal, vol)]
    if len(panels[0]) < 2:
        raise ValueError("Backtest period has fewer than two sessions")

    results, report = {}, {}
    for mode in ("dynamic_band", "entry_hold"):
        cfg = EngineConfig(fee_bps=fee_bps, slippage_bps=slippage_bps,
                           sizing_mode=mode)
        result = run_backtest(*panels, capital=capital, config=cfg)
        results[mode] = result
        report[mode] = {
            "full": _period_metrics(result),
            "2022_plus": _period_metrics(result, "2022-01-01"),
            "ending_equity": float(result.equity.iloc[-1]),
            "average_exposure": float(result.exposure.mean()),
        }

    spy = _benchmark(close["SPY"].loc[start_ts:], capital,
                     results["dynamic_band"].equity.index)
    report["SPY"] = {
        "full": performance_metrics(spy),
        "2022_plus": performance_metrics(spy.loc["2022-01-01":]),
        "ending_equity": float(spy.iloc[-1]),
        "average_exposure": 1.0,
    }

    if out_dir:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for mode, result in results.items():
            result.equity.to_csv(destination / f"{mode}_equity.csv")
            result.trades.to_csv(destination / f"{mode}_trades.csv", index=False)
            result.target_weights.to_csv(destination / f"{mode}_targets.csv")
        spy.to_csv(destination / "spy_equity.csv")
        with open(destination / "summary.json", "w") as handle:
            json.dump(report, handle, indent=2, default=float)
    return report


def _fmt(value, percent=False):
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%" if percent else f"{value:.2f}"


def print_report(report: dict) -> None:
    print("\nDiversified Trend Sleeve — sizing comparison")
    print("=" * 86)
    print(f"{'Variant':<16} {'CAGR':>8} {'Sharpe':>8} {'Vol':>8} {'MaxDD':>9} "
          f"{'Trades':>8} {'Turn/yr':>8} {'End value':>12} {'Avg exp':>9}")
    for name in ("dynamic_band", "entry_hold", "SPY"):
        full = report[name]["full"]
        print(f"{name:<16} {_fmt(full['cagr'], True):>8} {_fmt(full['sharpe']):>8} "
              f"{_fmt(full['vol'], True):>8} {_fmt(full['max_drawdown'], True):>9} "
              f"{full['trades']:>8} {_fmt(full['annual_turnover'], True):>8} "
              f"${report[name]['ending_equity']:>11,.0f} "
              f"{_fmt(report[name]['average_exposure'], True):>9}")
    print("=" * 86)
    print("Recent period (2022+)")
    for name in ("dynamic_band", "entry_hold", "SPY"):
        recent = report[name]["2022_plus"]
        print(f"  {name:<14} CAGR {_fmt(recent['cagr'], True):>7}  "
              f"Sharpe {_fmt(recent['sharpe']):>5}  MaxDD "
              f"{_fmt(recent['max_drawdown'], True):>7}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("tiingo", "yfinance"), default="tiingo")
    parser.add_argument("--start", default="2017-01-03")
    parser.add_argument("--end", default=None,
                        help="Inclusive final session (YYYY-MM-DD); defaults to latest session")
    parser.add_argument("--capital", type=float, default=13_500.0)
    parser.add_argument("--fee-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    through = args.end or strategy._expected_last_session().isoformat()
    fetch_start = (pd.Timestamp(args.start) - pd.Timedelta(days=400)).date().isoformat()
    print(f"Loading frozen {args.provider} snapshot (warm-up from {fetch_start})...")
    prices, fingerprint, path, coverage = load_snapshot(
        args.provider, fetch_start, through, refresh=args.refresh)
    print(f"Snapshot: {path}  sha256={fingerprint}")
    report = compare(prices, args.start, args.capital,
                     args.fee_bps, args.slippage_bps, args.out_dir)
    report["metadata"] = dict(provider=args.provider, snapshot=str(path),
                              sha256=fingerprint, start=args.start, through=through,
                              coverage=coverage,
                              capital=args.capital, fee_bps=args.fee_bps,
                              slippage_bps=args.slippage_bps)
    if args.out_dir:
        with open(Path(args.out_dir) / "summary.json", "w") as handle:
            json.dump(report, handle, indent=2, default=float)
    print_report(report)


if __name__ == "__main__":
    main()
