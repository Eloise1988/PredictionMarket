from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import date, datetime, timezone

from prediction_agent.backtest.engine import BacktestEngine


@dataclass
class FakeIdea:
    created_at: datetime
    ticker: str
    direction: str
    score: float
    confidence: float
    event_theme: str


class BacktestEngineTests(unittest.TestCase):
    def test_uses_next_day_entry_to_avoid_lookahead(self) -> None:
        idea = FakeIdea(
            created_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
            ticker="ABC",
            direction="long",
            score=0.8,
            confidence=0.7,
            event_theme="theme",
        )

        series = {
            date(2025, 1, 1): 100.0,
            date(2025, 1, 2): 110.0,
            date(2025, 1, 3): 121.0,
            date(2025, 1, 6): 130.0,
        }

        engine = BacktestEngine(transaction_cost_bps=0.0, horizon_days=1)
        trades = engine.simulate([idea], {"ABC": series}, min_score=0.5)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].entry_date, date(2025, 1, 2))
        self.assertEqual(trades[0].exit_date, date(2025, 1, 3))
        self.assertAlmostEqual(trades[0].raw_return, 0.10, places=6)

    def test_short_direction_sign(self) -> None:
        idea = FakeIdea(
            created_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
            ticker="XYZ",
            direction="short",
            score=0.8,
            confidence=0.7,
            event_theme="theme",
        )

        series = {
            date(2025, 1, 2): 100.0,
            date(2025, 1, 3): 90.0,
            date(2025, 1, 6): 85.0,
        }

        engine = BacktestEngine(transaction_cost_bps=0.0, horizon_days=1)
        trades = engine.simulate([idea], {"XYZ": series}, min_score=0.5)

        self.assertEqual(len(trades), 1)
        self.assertGreater(trades[0].raw_return, 0.0)


if __name__ == "__main__":
    unittest.main()
