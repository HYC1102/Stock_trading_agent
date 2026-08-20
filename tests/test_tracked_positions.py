import unittest

import pandas as pd

import strategy


class ExecutedPositionPnlTests(unittest.TestCase):
    def test_uses_account_start_fill_for_preexisting_signal(self):
        days = pd.to_datetime(["2026-07-27", "2026-07-28", "2026-08-19"])
        net = pd.DataFrame({"XBI": [0.10, 0.10, 0.10]}, index=days)
        opens = pd.DataFrame({"XBI": [99.0, 100.0, 112.0]}, index=days)
        closes = pd.DataFrame({"XBI": [99.0, 101.0, 113.0]}, index=days)
        scale = pd.Series(1.0, index=days)

        pnl = strategy._executed_position_pnl(
            net, opens, closes, pd.DataFrame(), scale, 10_000,
            days[-1], track_start="2026-07-28")

        self.assertEqual(pnl["XBI"]["entry_date"], pd.Timestamp("2026-07-28"))
        self.assertAlmostEqual(pnl["XBI"]["avg_cost"], 100.0)
        self.assertAlmostEqual(pnl["XBI"]["gain_pct"], 0.13)

    def test_waits_for_portfolio_trade_then_fills_next_open(self):
        days = pd.to_datetime([
            "2026-07-27", "2026-07-28", "2026-08-11",
            "2026-08-18", "2026-08-19",
        ])
        # A signal may exist on 11 Aug, but NET remains zero until capital is
        # allocated on 18 Aug. Only the portfolio trade should establish cost.
        net = pd.DataFrame({"XLE": [0.0, 0.0, 0.0, 0.04, 0.04]}, index=days)
        opens = pd.DataFrame({"XLE": [60.0, 60.0, 59.99, 63.41, 63.85]}, index=days)
        closes = pd.DataFrame({"XLE": [60.0, 60.0, 60.93, 63.68, 63.58]}, index=days)
        scale = pd.Series(1.0, index=days)
        trades = pd.DataFrame([{
            "date": pd.Timestamp("2026-08-18"), "ticker": "XLE",
            "target": 400.0, "equity": 10_000.0,
        }])

        pnl = strategy._executed_position_pnl(
            net, opens, closes, trades, scale, 10_000,
            days[-1], track_start="2026-07-28")

        self.assertEqual(pnl["XLE"]["entry_date"], pd.Timestamp("2026-08-19"))
        self.assertAlmostEqual(pnl["XLE"]["avg_cost"], 63.85)
        self.assertAlmostEqual(pnl["XLE"]["gain_pct"], 63.58 / 63.85 - 1)

    def test_additional_buy_updates_average_cost(self):
        days = pd.to_datetime([
            "2026-07-27", "2026-07-28", "2026-08-01",
            "2026-08-02", "2026-08-03",
        ])
        net = pd.DataFrame({"ETF": [0.10, 0.10, 0.10, 0.20, 0.20]}, index=days)
        opens = pd.DataFrame({"ETF": [10.0, 10.0, 11.0, 11.0, 12.0]}, index=days)
        closes = pd.DataFrame({"ETF": [10.0, 10.0, 11.0, 12.0, 13.0]}, index=days)
        scale = pd.Series(1.0, index=days)
        trades = pd.DataFrame([{
            "date": pd.Timestamp("2026-08-02"), "ticker": "ETF",
            "target": 2_000.0, "equity": 10_000.0,
        }])

        pnl = strategy._executed_position_pnl(
            net, opens, closes, trades, scale, 10_000,
            days[-1], track_start="2026-07-28")

        # Initial 100 shares at $10, then target 166.6667 shares at $12.
        self.assertAlmostEqual(pnl["ETF"]["shares"], 2_000 / 12)
        self.assertAlmostEqual(pnl["ETF"]["avg_cost"], 10.8)
        self.assertAlmostEqual(pnl["ETF"]["gain_dollar"], (2_000 / 12) * (13 - 10.8))


if __name__ == "__main__":
    unittest.main()
