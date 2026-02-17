from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from prediction_agent.clients.http_client import HttpClient
from prediction_agent.connectors.base import PredictionConnector
from prediction_agent.models import PredictionSignal

logger = logging.getLogger(__name__)


class PolymarketConnector(PredictionConnector):
    source_name = "polymarket"

    def __init__(self, gamma_base_url: str, clob_base_url: str, limit: int = 200, timeout: int = 15):
        self.gamma_base_url = gamma_base_url.rstrip("/")
        self.clob_base_url = clob_base_url.rstrip("/")
        self.limit = limit
        self.http = HttpClient(timeout=timeout)

    def fetch_signals(self) -> List[PredictionSignal]:
        params = {"closed": False, "active": True, "limit": self.limit}
        markets = self.http.get_json(f"{self.gamma_base_url}/markets", params=params)
        if not isinstance(markets, list):
            logger.warning("Unexpected Polymarket response type", extra={"type": type(markets).__name__})
            return []

        signals: List[PredictionSignal] = []
        for market in markets:
            try:
                question = (market.get("question") or "").strip()
                if not question:
                    continue

                prob_yes, prob_source = self._extract_probability_with_source(market)
                if prob_yes is None:
                    prob_yes = self._fallback_clob_price(market)
                    prob_source = "clob.price"
                if prob_yes is None:
                    continue

                liquidity = _to_float(market.get("liquidityNum") or market.get("liquidity") or 0)
                volume_24h = _to_float(market.get("volume24hr") or market.get("volume24hrClob") or 0)

                slug = market.get("slug") or ""
                event_slug = market.get("eventSlug") or slug
                url = f"https://polymarket.com/event/{event_slug}" if event_slug else "https://polymarket.com"

                updated_at = _parse_dt(market.get("updatedAt"))
                market_id = str(market.get("id") or market.get("conditionId") or slug)
                if not market_id:
                    continue

                signals.append(
                    PredictionSignal(
                        source=self.source_name,
                        market_id=market_id,
                        question=question,
                        url=url,
                        prob_yes=prob_yes,
                        volume_24h=volume_24h,
                        liquidity=liquidity,
                        updated_at=updated_at,
                        raw={
                            "slug": slug,
                            "eventSlug": event_slug,
                            "endDate": market.get("endDate"),
                            "probability_source": prob_source,
                            "outcomes": market.get("outcomes"),
                            "outcomePrices": market.get("outcomePrices"),
                        },
                    )
                )
            except Exception as exc:
                logger.debug("Failed to parse Polymarket market", extra={"error": str(exc)})
                continue

        return signals

    def _extract_probability_with_source(self, market: Dict[str, Any]) -> tuple[Optional[float], str]:
        outcomes = _parse_json_list(market.get("outcomes"))
        prices = _parse_json_list(market.get("outcomePrices"))

        if outcomes and prices and len(outcomes) == len(prices):
            for i, outcome in enumerate(outcomes):
                if str(outcome).strip().lower() in {"yes", "true", "will happen", "happen"}:
                    prob = _to_float(prices[i])
                    if prob <= 1:
                        return prob, "gamma.outcomePrices.yes"
                    return prob / 100.0, "gamma.outcomePrices.yes"

            # Binary markets often store YES as first outcome.
            prob = _to_float(prices[0])
            if prob <= 1:
                return prob, "gamma.outcomePrices.first_outcome"
            return prob / 100.0, "gamma.outcomePrices.first_outcome"

        for key in ("lastTradePrice", "bestAsk", "bestBid"):
            if key in market:
                raw = _to_float(market[key])
                if raw > 0:
                    return (raw if raw <= 1 else raw / 100.0), f"gamma.{key}"

        return None, ""

    def _fallback_clob_price(self, market: Dict[str, Any]) -> Optional[float]:
        token_ids = _parse_json_list(market.get("clobTokenIds"))
        if not token_ids:
            return None

        token_id = str(token_ids[0])
        try:
            payload = self.http.get_json(f"{self.clob_base_url}/price", params={"token_id": token_id})
        except Exception:
            return None

        if isinstance(payload, dict):
            for key in ("price", "mid", "midpoint"):
                if key in payload:
                    value = _to_float(payload[key])
                    if value <= 1:
                        return value
                    return value / 100.0
        return None


def _parse_json_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


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
