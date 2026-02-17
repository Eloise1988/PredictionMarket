from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

from prediction_agent.models import AlertPayload, CandidateIdea, EquitySnapshot, PredictionSignal

logger = logging.getLogger(__name__)


class MongoStore:
    def __init__(self, uri: str, db_name: str):
        # Ensure datetimes read from Mongo are timezone-aware (UTC).
        self.client = MongoClient(uri, tz_aware=True)
        self.db = self.client[db_name]
        self.signals_col: Collection = self.db["prediction_signals"]
        self.ideas_col: Collection = self.db["candidate_ideas"]
        self.alerts_col: Collection = self.db["alerts"]
        self.equity_cache_col: Collection = self.db["equity_cache"]
        self.equity_bars_col: Collection = self.db["equity_daily_bars"]
        self.backtest_reports_col: Collection = self.db["backtest_reports"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.signals_col.create_index([("source", ASCENDING), ("market_id", ASCENDING), ("updated_at", ASCENDING)])
        self.signals_col.create_index("created_at", expireAfterSeconds=14 * 24 * 3600)

        self.ideas_col.create_index([("ticker", ASCENDING), ("created_at", ASCENDING)])
        self.ideas_col.create_index("created_at", expireAfterSeconds=30 * 24 * 3600)

        self.alerts_col.create_index([("digest", ASCENDING), ("created_at", ASCENDING)])
        self.alerts_col.create_index("created_at", expireAfterSeconds=30 * 24 * 3600)

        self.equity_cache_col.create_index([("ticker", ASCENDING)], unique=True)
        self.equity_cache_col.create_index("fetched_at", expireAfterSeconds=7 * 24 * 3600)

        # Single canonical index for equity bars. Do not create the same key twice
        # with different options, or MongoDB raises IndexOptionsConflict.
        self.equity_bars_col.create_index(
            [("ticker", ASCENDING), ("date", ASCENDING)],
            unique=True,
            name="equity_bars_ticker_date_unique",
        )
        self.backtest_reports_col.create_index("created_at")

    def save_signals(self, signals: Iterable[PredictionSignal]) -> None:
        now = datetime.now(timezone.utc)
        docs = []
        for s in signals:
            docs.append({**s.model_dump(), "created_at": now})
        if docs:
            self.signals_col.insert_many(docs)

    def save_ideas(self, ideas: Iterable[CandidateIdea]) -> None:
        docs = [idea.model_dump() for idea in ideas]
        if docs:
            self.ideas_col.insert_many(docs)

    def get_candidate_ideas(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[CandidateIdea]:
        query: Dict = {}
        if start or end:
            query["created_at"] = {}
            if start:
                query["created_at"]["$gte"] = start
            if end:
                query["created_at"]["$lte"] = end

        cursor = self.ideas_col.find(query).sort("created_at", ASCENDING)
        ideas: List[CandidateIdea] = []
        for doc in cursor:
            doc.pop("_id", None)
            try:
                ideas.append(CandidateIdea(**doc))
            except Exception:
                continue
        return ideas

    def get_cached_equity(self, ticker: str, max_age_seconds: int) -> Optional[EquitySnapshot]:
        doc = self.equity_cache_col.find_one({"ticker": ticker.upper()})
        if not doc:
            return None
        fetched_at = _as_utc(doc.get("fetched_at"))
        if not isinstance(fetched_at, datetime):
            return None

        if datetime.now(timezone.utc) - fetched_at > timedelta(seconds=max_age_seconds):
            return None

        doc["fetched_at"] = fetched_at
        doc.pop("_id", None)
        return EquitySnapshot(**doc)

    def upsert_equity(self, snapshot: EquitySnapshot) -> None:
        self.equity_cache_col.update_one(
            {"ticker": snapshot.ticker.upper()},
            {"$set": snapshot.model_dump()},
            upsert=True,
        )

    def has_recent_alert(self, digest: str, cooldown_minutes: int) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)
        existing = self.alerts_col.find_one({"digest": digest, "created_at": {"$gte": cutoff}})
        return existing is not None

    def upsert_daily_bars(self, ticker: str, bars: Dict[date, float], source: str) -> None:
        if not bars:
            return

        operations = []
        for day, close in bars.items():
            dt = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            operations.append(
                {
                    "filter": {"ticker": ticker.upper(), "date": dt},
                    "update": {
                        "$set": {
                            "ticker": ticker.upper(),
                            "date": dt,
                            "close": float(close),
                            "source": source,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                    "upsert": True,
                }
            )

        if not operations:
            return

        from pymongo import UpdateOne

        self.equity_bars_col.bulk_write(
            [UpdateOne(op["filter"], op["update"], upsert=op["upsert"]) for op in operations],
            ordered=False,
        )

    def get_daily_bars(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[date, float]:
        query: Dict = {"ticker": ticker.upper()}
        if start_date or end_date:
            query["date"] = {}
            if start_date:
                query["date"]["$gte"] = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
            if end_date:
                query["date"]["$lte"] = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)

        cursor = self.equity_bars_col.find(query, {"_id": 0, "date": 1, "close": 1}).sort("date", ASCENDING)
        out: Dict[date, float] = {}
        for row in cursor:
            d = _as_utc(row.get("date"))
            c = row.get("close")
            if isinstance(d, datetime) and isinstance(c, (int, float)):
                out[d.date()] = float(c)
        return out

    def save_alert(self, payload: AlertPayload) -> None:
        self.alerts_col.insert_one(
            {
                "digest": payload.digest,
                "created_at": payload.created_at,
                "idea_count": len(payload.ideas),
                "tickers": [idea.ticker for idea in payload.ideas],
                "scores": [idea.score for idea in payload.ideas],
                "summary": payload.llm_summary,
            }
        )

    def save_backtest_report(self, report: Dict) -> None:
        self.backtest_reports_col.insert_one(report)

    @staticmethod
    def ideas_digest(ideas: List[CandidateIdea]) -> str:
        # Stable digest to suppress duplicate alerts across loop cycles.
        key = "|".join(
            f"{i.market_source}:{i.market_id}:{i.ticker}:{i.direction}:{round(i.score, 3)}"
            for i in sorted(ideas, key=lambda x: (x.market_source, x.market_id, x.ticker, x.direction))
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _as_utc(value: object) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
