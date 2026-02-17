from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from prediction_agent.backtest.calibration import walk_forward_calibrate
from prediction_agent.backtest.engine import BacktestEngine


@dataclass
class FakeIdea:
    created_at: datetime
    ticker: str
    direction: str
    score: float
    confidence: float
    event_theme: str


class CalibrationTests(unittest.TestCase):
    def test_walk_forward_returns_threshold(self) -> None:
        ideas = []
        series = {}
        base_created = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        for i in range(8):
            ticker = f"T{i}"
            score = 0.8 if i % 2 == 0 else 0.6
            created = base_created + timedelta(days=4 * i)
            trade_day = (created + timedelta(days=1)).date()

            ideas.append(
                FakeIdea(
                    created_at=created,
                    ticker=ticker,
                    direction="long",
                    score=score,
                    confidence=0.7,
                    event_theme="theme",
                )
            )

            if score >= 0.75:
                # Positive outcome for high-score ideas.
                series[ticker] = {
                    trade_day: 100.0,
                    trade_day + timedelta(days=1): 110.0,
                    trade_day + timedelta(days=2): 112.0,
                }
            else:
                # Negative outcome for low-score ideas.
                series[ticker] = {
                    trade_day: 100.0,
                    trade_day + timedelta(days=1): 95.0,
                    trade_day + timedelta(days=2): 94.0,
                }

        engine = BacktestEngine(transaction_cost_bps=0.0, horizon_days=1)
        report = walk_forward_calibrate(
            ideas=ideas,
            series_by_ticker=series,
            engine=engine,
            thresholds=[0.55, 0.75],
            train_days=14,
            val_days=7,
            min_trades=1,
        )

        self.assertTrue(report.folds)
        self.assertIsNotNone(report.recommended_threshold)
        self.assertGreaterEqual(report.recommended_threshold, 0.55)


if __name__ == "__main__":
    unittest.main()
