from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from prediction_agent.clients.http_client import HttpClient
from prediction_agent.connectors.base import PredictionConnector
from prediction_agent.models import PredictionSignal

logger = logging.getLogger(__name__)


class KalshiConnector(PredictionConnector):
    source_name = "kalshi"

    def __init__(self, base_url: str, limit: int = 200, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.limit = limit
        self.http = HttpClient(timeout=timeout)

    def fetch_signals(self) -> List[PredictionSignal]:
        signals: List[PredictionSignal] = []
        cursor: Optional[str] = None

        while len(signals) < self.limit:
            remaining = self.limit - len(signals)
            params: Dict[str, Any] = {"status": "open", "limit": min(100, remaining)}
            if cursor:
                params["cursor"] = cursor

            payload = self.http.get_json(f"{self.base_url}/markets", params=params)
            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            if not markets:
                break

            for market in markets:
                question = (market.get("title") or market.get("name") or "").strip()
                if not question:
                    continue

                prob_yes = _extract_yes_probability(market)
                if prob_yes is None:
                    continue

                ticker = str(market.get("ticker") or market.get("id") or "")
                if not ticker:
                    continue

                volume_24h = _to_float(
                    market.get("volume_24h")
                    or market.get("volume")
                    or market.get("yes_volume")
                    or market.get("open_interest")
                    or 0
                )
                liquidity = _to_float(market.get("open_interest") or market.get("liquidity") or volume_24h)
                updated_at = _parse_dt(market.get("updated_time") or market.get("close_time"))

                event_ticker = market.get("event_ticker") or ""
                series_ticker = market.get("series_ticker") or ""
                slug = market.get("slug") or ""
                if slug:
                    url = f"https://kalshi.com/markets/{slug}"
                elif event_ticker:
                    url = f"https://kalshi.com/markets/{event_ticker.lower()}"
                else:
                    url = "https://kalshi.com/markets"

                signals.append(
                    PredictionSignal(
                        source=self.source_name,
                        market_id=ticker,
                        question=question,
                        url=url,
                        prob_yes=prob_yes,
                        volume_24h=volume_24h,
                        liquidity=liquidity,
                        updated_at=updated_at,
                        raw={
                            "ticker": ticker,
                            "event_ticker": event_ticker,
                            "series_ticker": series_ticker,
                            "title": market.get("title"),
                            "subtitle": market.get("subtitle"),
                            "yes_sub_title": market.get("yes_sub_title"),
                            "no_sub_title": market.get("no_sub_title"),
                            "status": market.get("status"),
                            "close_time": market.get("close_time"),
                        },
                    )
                )

            cursor = payload.get("cursor") if isinstance(payload, dict) else None
            if not cursor:
                break

        return signals


def _extract_yes_probability(market: Dict[str, Any]) -> Optional[float]:
    for key in ("yes_price", "yes_last_price", "yes_bid", "yes_ask", "last_price"):
        if key in market and market[key] is not None:
            value = _to_float(market[key])
            if value == 0:
                continue
            if value <= 1:
                return value
            if value <= 100:
                return value / 100.0

    yes_bid = _to_float(market.get("yes_bid"))
    yes_ask = _to_float(market.get("yes_ask"))
    if yes_bid > 0 and yes_ask > 0:
        midpoint = (yes_bid + yes_ask) / 2
        return midpoint if midpoint <= 1 else midpoint / 100.0

    no_bid = _to_float(market.get("no_bid"))
    if no_bid > 0:
        no_prob = no_bid if no_bid <= 1 else no_bid / 100.0
        return max(0.0, min(1.0, 1.0 - no_prob))

    return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_dt(raw: Any) -> datetime:
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)
