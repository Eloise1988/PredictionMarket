from __future__ import annotations

import logging
from datetime import date
from typing import Dict, Iterable

from prediction_agent.clients.alpha_vantage import AlphaVantageClient
from prediction_agent.storage.mongo import MongoStore

logger = logging.getLogger(__name__)


class PriceHistoryLoader:
    def __init__(self, store: MongoStore, alpha_vantage: AlphaVantageClient):
        self.store = store
        self.alpha_vantage = alpha_vantage

    def load_series(self, ticker: str, start_date: date, end_date: date) -> Dict[date, float]:
        cached = self.store.get_daily_bars(ticker, start_date=start_date, end_date=end_date)
        if _coverage_ratio(cached, start_date, end_date) >= 0.75:
            return cached

        if not self.alpha_vantage.enabled():
            return cached

        fetched = self.alpha_vantage.fetch_daily_adjusted_series(ticker, outputsize="full")
        if fetched:
            self.store.upsert_daily_bars(ticker, fetched, source="alpha_vantage")

        merged = self.store.get_daily_bars(ticker, start_date=start_date, end_date=end_date)
        if merged:
            return merged
        return fetched


def _coverage_ratio(series: Dict[date, float], start_date: date, end_date: date) -> float:
    if start_date > end_date:
        return 0.0
    days = (end_date - start_date).days + 1
    if days <= 0:
        return 0.0

    observed = sum(1 for d in series if start_date <= d <= end_date)
    return observed / days
