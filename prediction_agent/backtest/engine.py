from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Protocol

from prediction_agent.backtest.models import BacktestMetrics, BacktestTrade
from prediction_agent.backtest.stats import compute_metrics

logger = logging.getLogger(__name__)


class IdeaLike(Protocol):
    created_at: datetime
    ticker: str
    direction: str
    score: float
    confidence: float
    event_theme: str


class BacktestEngine:
    def __init__(self, transaction_cost_bps: float, horizon_days: int):
        self.transaction_cost_bps = transaction_cost_bps
        self.horizon_days = horizon_days

    def simulate(
        self,
        ideas: Iterable[IdeaLike],
        price_series_by_ticker: Dict[str, Dict[date, float]],
        min_score: float,
    ) -> List[BacktestTrade]:
        trades: List[BacktestTrade] = []
        roundtrip_cost = (self.transaction_cost_bps / 10000.0) * 2

        for idea in sorted(ideas, key=lambda x: x.created_at):
            if idea.score < min_score:
                continue

            series = price_series_by_ticker.get(idea.ticker.upper())
            if not series:
                continue

            entry_date = _next_trading_day(series, idea.created_at.date() + timedelta(days=1))
            if entry_date is None:
                continue

            exit_date = _shift_trading_days(series, entry_date, self.horizon_days)
            if exit_date is None:
                continue

            entry_close = series.get(entry_date)
            exit_close = series.get(exit_date)
            if not entry_close or not exit_close:
                continue

            direction_sign = 1.0 if idea.direction.lower() == "long" else -1.0
            raw_return = direction_sign * ((exit_close / entry_close) - 1.0)
            net_return = raw_return - roundtrip_cost

            trades.append(
                BacktestTrade(
                    created_at=idea.created_at,
                    ticker=idea.ticker,
                    direction=idea.direction,
                    score=idea.score,
                    confidence=idea.confidence,
                    event_theme=idea.event_theme,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry_close=entry_close,
                    exit_close=exit_close,
                    raw_return=raw_return,
                    net_return=net_return,
                )
            )

        return trades

    def metrics(self, trades: Iterable[BacktestTrade]) -> BacktestMetrics:
        return compute_metrics([t.net_return for t in trades], self.horizon_days)


def _next_trading_day(series: Dict[date, float], day: date) -> Optional[date]:
    keys = sorted(series)
    for k in keys:
        if k >= day:
            return k
    return None


def _shift_trading_days(series: Dict[date, float], start_day: date, shift: int) -> Optional[date]:
    keys = sorted(series)
    try:
        idx = keys.index(start_day)
    except ValueError:
        return None

    target_idx = idx + shift
    if target_idx >= len(keys):
        return None
    return keys[target_idx]
