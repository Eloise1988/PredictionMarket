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
        markets = self._fetch_events_markets_paginated()
        signals: List[PredictionSignal] = []
        for market in markets:
            try:
                event = _primary_event(market)
                if not _is_open_active_row(event):
                    continue
                if not _is_open_active_row(market):
                    continue

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

        signals.sort(key=lambda s: (_safe_timestamp(s.updated_at), s.liquidity), reverse=True)
        return signals

    def _fetch_events_markets_paginated(self) -> List[Dict[str, Any]]:
        # Pull event pages, flatten event markets, and keep active/open/non-archived rows only.
        page_size = min(1000, self.limit)
        offset = 0
        rows: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        while len(rows) < self.limit:
            remaining = self.limit - len(rows)
            params: List[tuple[str, Any]] = [
                ("active", "true"),
                ("archived", "false"),
                ("closed", "false"),
                ("featured_order", "true"),
                ("order", "volume"),
                ("ascending", "false"),
                ("limit", min(page_size, remaining)),
            ]
            if offset > 0:
                params.append(("offset", offset))

            payload = self.http.get_json(f"{self.gamma_base_url}/events/pagination", params=params)
            events = _extract_event_rows(payload)
            if not events:
                break

            added = 0
            for event in events:
                if not isinstance(event, dict):
                    continue
                if not _is_open_active_row(event):
                    continue

                for market in _extract_markets_from_event(event):
                    normalized = _normalize_event_market_row(event, market)
                    if not _is_open_active_row(normalized):
                        continue

                    market_id = _market_row_id(normalized)
                    if market_id:
                        if market_id in seen_ids:
                            continue
                        seen_ids.add(market_id)
                    rows.append(normalized)
                    added += 1
                    if len(rows) >= self.limit:
                        break

                if len(rows) >= self.limit:
                    break

            if len(events) < int(_get_param_value(params, "limit", 0)):
                break
            if added == 0:
                # Protect against infinite loops if API keeps returning duplicate pages.
                break
            offset += len(events)

        rows.sort(key=_market_sort_key, reverse=True)
        return rows[: self.limit]

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


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "y"}:
            return True
        if cleaned in {"false", "0", "no", "n", ""}:
            return False
    return None


def _is_open_active_row(row: Dict[str, Any]) -> bool:
    active = _to_bool(row.get("active"))
    closed = _to_bool(row.get("closed"))
    archived = _to_bool(row.get("archived"))
    if active is False:
        return False
    if closed is True:
        return False
    if archived is True:
        return False
    return True


def _extract_event_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []

    direct = payload.get("events")
    if isinstance(direct, list):
        return [x for x in direct if isinstance(x, dict)]

    for key in ("data", "items", "results", "rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict):
            nested = value.get("events") or value.get("items")
            if isinstance(nested, list):
                return [x for x in nested if isinstance(x, dict)]
    return []


def _get_param_value(params: List[tuple[str, Any]], key: str, default: Any) -> Any:
    for name, value in reversed(params):
        if name == key:
            return value
    return default


def _extract_markets_from_event(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    markets = event.get("markets")
    if isinstance(markets, list):
        return [x for x in markets if isinstance(x, dict)]
    if isinstance(markets, dict):
        return [markets]

    market = event.get("market")
    if isinstance(market, dict):
        return [market]
    if isinstance(market, list):
        return [x for x in market if isinstance(x, dict)]

    # Some responses may include market-like rows directly in the event object.
    if any(k in event for k in ("conditionId", "outcomes", "outcomePrices", "question", "clobTokenIds")):
        return [event]
    return []


def _normalize_event_market_row(event: Dict[str, Any], market: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(market)
    row["events"] = [event]

    if not _to_clean_str(row.get("eventSlug")):
        row["eventSlug"] = _to_clean_str(event.get("slug"))
    if not _to_clean_str(row.get("eventTitle")):
        row["eventTitle"] = _to_clean_str(event.get("title"))
    if not _to_clean_str(row.get("eventCategory")):
        row["eventCategory"] = _to_clean_str(event.get("category"))
    if not _to_clean_str(row.get("subCategory")):
        row["subCategory"] = _to_clean_str(event.get("subCategory") or event.get("subcategory"))
    if not _to_clean_str(row.get("question")):
        row["question"] = _to_clean_str(event.get("title"))
    if row.get("updatedAt") is None and event.get("updatedAt") is not None:
        row["updatedAt"] = event.get("updatedAt")
    if row.get("endDate") is None and event.get("endDate") is not None:
        row["endDate"] = event.get("endDate")
    if row.get("category") in (None, "") and event.get("category") is not None:
        row["category"] = event.get("category")
    if row.get("active") is None and event.get("active") is not None:
        row["active"] = event.get("active")
    if row.get("closed") is None and event.get("closed") is not None:
        row["closed"] = event.get("closed")
    if row.get("archived") is None and event.get("archived") is not None:
        row["archived"] = event.get("archived")
    return row


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


def _parse_dt_or_none(raw: Any) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, str) and raw:
        cleaned = raw.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(cleaned)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _safe_timestamp(value: datetime) -> float:
    dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _market_sort_key(market: Dict[str, Any]) -> tuple[float, float]:
    event = _primary_event(market)
    dt = (
        _parse_dt_or_none(market.get("updatedAt"))
        or _parse_dt_or_none(market.get("endDate"))
        or _parse_dt_or_none(market.get("createdAt"))
        or _parse_dt_or_none(event.get("updatedAt"))
        or _parse_dt_or_none(event.get("endDate"))
        or _parse_dt_or_none(event.get("createdAt"))
    )
    liquidity = _to_float(market.get("liquidityNum") or market.get("liquidity") or event.get("liquidity") or 0)
    return (_safe_timestamp(dt) if dt else 0.0, liquidity)


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
