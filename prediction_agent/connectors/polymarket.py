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

    def __init__(self, gamma_base_url: str, clob_base_url: str, limit: int = 1000, timeout: int = 15):
        self.gamma_base_url = gamma_base_url.rstrip("/")
        self.clob_base_url = clob_base_url.rstrip("/")
        self.limit = max(1, int(limit))
        self.http = HttpClient(timeout=timeout)

    def fetch_signals(self) -> List[PredictionSignal]:
        markets = self._fetch_markets_paginated()
        signals: List[PredictionSignal] = []
        for market in markets:
            try:
                event = _primary_event(market)
                slug = _to_clean_str(market.get("slug"))
                event_slug = _to_clean_str(market.get("eventSlug") or event.get("slug") or slug)
                question = _to_clean_str(market.get("question") or event.get("title"))
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

                url = f"https://polymarket.com/event/{event_slug}" if event_slug else "https://polymarket.com"
                event_title = _to_clean_str(market.get("eventTitle") or event.get("title"))
                event_category = _to_clean_str(market.get("eventCategory") or event.get("category"))
                sub_category = _to_clean_str(
                    market.get("subCategory") or event.get("subCategory") or event.get("subcategory")
                )
                category = _to_clean_str(market.get("category") or event_category)

                updated_at = _parse_dt(market.get("updatedAt") or event.get("updatedAt"))
                market_id = _to_clean_str(market.get("id") or market.get("conditionId") or slug)
                if not market_id:
                    continue

                tags = _extract_tags(market.get("tags"))
                if not tags:
                    tags = _extract_tags(event.get("tags"))

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
                            "eventId": _to_clean_str(event.get("id")),
                            "eventTicker": _to_clean_str(event.get("ticker")),
                            "endDate": market.get("endDate") or event.get("endDate"),
                            "category": category,
                            "eventCategory": event_category,
                            "subCategory": sub_category,
                            "eventTitle": event_title,
                            "tags": tags,
                            "probability_source": prob_source,
                            "outcomes": market.get("outcomes"),
                            "outcomePrices": market.get("outcomePrices"),
                            "active": market.get("active"),
                            "closed": market.get("closed"),
                            "archived": market.get("archived"),
                            "restricted": market.get("restricted"),
                        },
                    )
                )
            except Exception as exc:
                logger.debug("Failed to parse Polymarket market", extra={"error": str(exc)})
                continue

        return signals

    def _fetch_markets_paginated(self) -> List[Dict[str, Any]]:
        # Gamma supports `active=true&limit=...`; use offset paging when limit exceeds one page.
        page_size = min(1000, self.limit)
        offset = 0
        rows: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        while len(rows) < self.limit:
            remaining = self.limit - len(rows)
            params = {
                "active": True,
                "limit": min(page_size, remaining),
            }
            if offset > 0:
                params["offset"] = offset

            payload = self.http.get_json(f"{self.gamma_base_url}/markets", params=params)
            if not isinstance(payload, list):
                logger.warning("Unexpected Polymarket response type", extra={"type": type(payload).__name__})
                break
            if not payload:
                break

            added = 0
            for market in payload:
                market_id = _market_row_id(market)
                if market_id:
                    if market_id in seen_ids:
                        continue
                    seen_ids.add(market_id)
                rows.append(market)
                added += 1
                if len(rows) >= self.limit:
                    break

            if len(payload) < params["limit"]:
                break
            if added == 0:
                # Protect against infinite loops if API keeps returning duplicate pages.
                break
            offset += len(payload)

        return rows

    def _extract_probability_with_source(self, market: Dict[str, Any]) -> tuple[Optional[float], str]:
        outcomes = _parse_json_list(market.get("outcomes"))
        prices = _parse_json_list(market.get("outcomePrices"))

        if outcomes and prices and len(outcomes) == len(prices):
            for i, outcome in enumerate(outcomes):
                if str(outcome).strip().lower() in {"yes", "true", "will happen", "happen"}:
                    prob = _normalize_probability(prices[i])
                    if prob is not None:
                        return prob, "gamma.outcomePrices.yes"

            # Binary markets often store YES as first outcome.
            prob = _normalize_probability(prices[0])
            if prob is not None:
                return prob, "gamma.outcomePrices.first_outcome"

        for key in ("lastTradePrice", "bestAsk", "bestBid", "lastPrice", "price"):
            if key in market:
                prob = _normalize_probability(market[key])
                if prob is not None:
                    return prob, f"gamma.{key}"

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
                    value = _normalize_probability(payload[key])
                    if value is not None:
                        return value
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


def _to_clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_probability(value: Any) -> Optional[float]:
    prob = _to_float(value)
    if prob <= 0:
        return None
    if prob > 1:
        prob = prob / 100.0
    if prob < 0 or prob > 1:
        return None
    return prob


def _market_row_id(market: Dict[str, Any]) -> str:
    return _to_clean_str(market.get("id") or market.get("conditionId") or market.get("slug"))


def _primary_event(market: Dict[str, Any]) -> Dict[str, Any]:
    raw_events = market.get("events")
    if isinstance(raw_events, list):
        for event in raw_events:
            if isinstance(event, dict):
                return event
    if isinstance(raw_events, dict):
        return raw_events
    return {}


def _parse_dt(raw: Any) -> datetime:
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _extract_tags(raw_tags: Any) -> List[str]:
    if not raw_tags:
        return []
    if isinstance(raw_tags, str):
        parsed = _parse_json_list(raw_tags)
        return _extract_tags(parsed)
    if isinstance(raw_tags, list):
        out: List[str] = []
        for item in raw_tags:
            if isinstance(item, str):
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("slug", "name", "label", "tag"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        out.append(value.strip())
                        break
        return [x for x in out if x]
    return []
