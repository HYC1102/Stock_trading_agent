import unittest
from unittest import mock

import pandas as pd

import trend_backtest
from trend_engine import EngineConfig, desired_weights, run_backtest


def panels(days, opens, closes=None, signals=None, vols=None):
    index = pd.to_datetime(days)
    open_frame = pd.DataFrame(opens, index=index, dtype=float)
    close_frame = pd.DataFrame(closes or opens, index=index, dtype=float)
    signal_frame = pd.DataFrame(signals or {k: [1] * len(index) for k in opens},
                                index=index, dtype=float)
    vol_frame = pd.DataFrame(vols or {k: [0.20] * len(index) for k in opens},
                             index=index, dtype=float)
    return open_frame, close_frame, signal_frame, vol_frame


class EventTimingTests(unittest.TestCase):
    def test_close_decision_fills_at_next_open(self):
        data = panels(
            ["2026-01-05", "2026-01-06", "2026-01-07"],
            {"A": [10, 11, 12]}, signals={"A": [1, 1, 1]},
        )
        cfg = EngineConfig(asset_budget=0.02, fee_bps=0, slippage_bps=0)
        result = run_backtest(*data, capital=1_000, config=cfg)

        first = result.trades.iloc[0]
        self.assertEqual(first.decision_date, pd.Timestamp("2026-01-05"))
        self.assertEqual(first.fill_date, pd.Timestamp("2026-01-06"))
        self.assertEqual(first.fill_price, 11)

    def test_costs_reduce_equity_and_cash_never_goes_negative(self):
        data = panels(
            ["2026-01-05", "2026-01-06", "2026-01-07"],
            {"A": [10, 10, 10]}, signals={"A": [1, 0, 0]},
        )
        free = run_backtest(*data, 1_000,
                            EngineConfig(asset_budget=1, max_name=1,
                                         fee_bps=0, slippage_bps=0))
        costly = run_backtest(*data, 1_000,
                              EngineConfig(asset_budget=1, max_name=1,
                                           fee_bps=10, slippage_bps=10))

        self.assertLess(costly.equity.iloc[-1], free.equity.iloc[-1])
        self.assertGreaterEqual(costly.cash.min(), 0)
        self.assertGreater(costly.trades["slippage_cost"].sum(), 0)


class SizingModeTests(unittest.TestCase):
    def test_dynamic_band_rebalances_but_entry_hold_does_not(self):
        data = panels(
            ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"],
            {"A": [10, 10, 10, 10]},
            signals={"A": [1, 1, 1, 1]},
            vols={"A": [0.20, 0.10, 0.10, 0.10]},
        )
        common = dict(asset_budget=0.02, max_name=1, band=0.15,
                      fee_bps=0, slippage_bps=0)
        dynamic = run_backtest(*data, 1_000,
                               EngineConfig(sizing_mode="dynamic_band", **common))
        held = run_backtest(*data, 1_000,
                            EngineConfig(sizing_mode="entry_hold", **common))

        self.assertEqual(len(dynamic.trades), 2)
        self.assertEqual(len(held.trades), 1)
        self.assertAlmostEqual(dynamic.final_shares["A"], 20)
        self.assertAlmostEqual(held.final_shares["A"], 10)

    def test_dynamic_band_ignores_small_target_change(self):
        data = panels(
            ["2026-01-05", "2026-01-06", "2026-01-07"],
            {"A": [10, 10, 10]}, signals={"A": [1, 1, 1]},
            vols={"A": [0.20, 0.19, 0.19]},
        )
        result = run_backtest(*data, 1_000,
                              EngineConfig(asset_budget=0.02, max_name=1,
                                           band=0.15, fee_bps=0, slippage_bps=0))
        self.assertEqual(len(result.trades), 1)

    def test_targets_do_not_depend_on_ticker_order(self):
        signal = pd.Series({"A": 1.0, "B": 1.0})
        vol = pd.Series({"A": 0.04, "B": 0.04})
        cfg = EngineConfig(asset_budget=0.60, max_name=1)

        ab = desired_weights(signal, vol, cfg)
        ba = desired_weights(signal.reindex(["B", "A"]), vol.reindex(["B", "A"]), cfg)

        self.assertAlmostEqual(ab["A"], ba["A"])
        self.assertAlmostEqual(ab["B"], ba["B"])
        self.assertAlmostEqual(ab.sum(), 1.0)


class SnapshotValidationTests(unittest.TestCase):
    def test_rejects_series_that_ends_before_inclusive_through_date(self):
        frame = pd.DataFrame(
            {name: [10.0] for name in ("Open", "High", "Low", "Close")},
            index=pd.to_datetime(["2026-08-18"]),
        )
        with mock.patch.object(trend_backtest.strategy, "UNIVERSE", {"A": "test"}):
            with self.assertRaisesRegex(ValueError, "A ends 2026-08-18"):
                trend_backtest._validate_snapshot({"A": frame}, "2026-08-19")

    def test_snapshot_hash_changes_when_a_price_changes(self):
        frame = pd.DataFrame(
            {name: [10.0] for name in ("Open", "High", "Low", "Close")},
            index=pd.to_datetime(["2026-08-18"]),
        )
        changed = frame.copy()
        changed.loc[:, "Close"] = 11.0
        self.assertNotEqual(trend_backtest._snapshot_hash({"A": frame}),
                            trend_backtest._snapshot_hash({"A": changed}))


if __name__ == "__main__":
    unittest.main()
