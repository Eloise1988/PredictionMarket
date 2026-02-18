from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import websockets

from prediction_agent.clients.http_client import HttpClient
from prediction_agent.models import PredictionSignal
from prediction_agent.streaming.base import SignalHandler, SignalStreamer

logger = logging.getLogger(__name__)


@dataclass
class _MarketMeta:
    market_id: str
    question: str
    url: str
    liquidity: float
    volume_24h: float
    updated_at: datetime
    raw: Dict[str, Any]


class PolymarketWebsocketStreamer(SignalStreamer):
    source_name = "polymarket"

    def __init__(self, gamma_base_url: str, ws_url: str, market_limit: int = 100, timeout: int = 15):
        self.gamma_base_url = gamma_base_url.rstrip("/")
        self.ws_url = ws_url
        self.market_limit = market_limit
        self.http = HttpClient(timeout=timeout)

    async def stream(self, on_signal: SignalHandler) -> None:
        while True:
            try:
                token_map = self._load_token_map()
                if not token_map:
                    await asyncio.sleep(5)
                    continue

                subscribe_assets = list(token_map.keys())
                logger.info("Polymarket websocket subscribe", extra={"assets": len(subscribe_assets)})

                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20, max_size=2**20) as ws:
                    payload = {"type": "market", "assets_ids": subscribe_assets}
                    await ws.send(json.dumps(payload))

                    async for raw in ws:
                        for event in _to_event_list(raw):
                            signal = self._parse_event(event, token_map)
                            if signal:
                                await on_signal(signal)
            except Exception as exc:
                logger.warning("Polymarket websocket disconnected", extra={"error": str(exc)})
                await asyncio.sleep(3)

    def _load_token_map(self) -> Dict[str, _MarketMeta]:
        params = {"active": "true", "closed": "false", "limit": self.market_limit}
        markets = self.http.get_json(f"{self.gamma_base_url}/markets", params=params)
        if not isinstance(markets, list):
            return {}

        token_map: Dict[str, _MarketMeta] = {}
        for market in markets:
            if _to_bool(market.get("active")) is False:
                continue
            if _to_bool(market.get("closed")) is True:
                continue

            event = _primary_event(market)
            question = _to_clean_str(market.get("question") or event.get("title"))
            if not question:
                continue

            token_ids = _parse_json_list(market.get("clobTokenIds"))
            if not token_ids:
                continue

            yes_token_id = _to_clean_str(token_ids[0])
            market_id = _to_clean_str(market.get("id") or market.get("conditionId") or market.get("slug"))
            if not market_id:
                continue

            slug = _to_clean_str(market.get("slug"))
            event_slug = _to_clean_str(market.get("eventSlug") or event.get("slug") or slug)
            url = f"https://polymarket.com/event/{event_slug}" if event_slug else "https://polymarket.com"
            event_category = _to_clean_str(market.get("eventCategory") or event.get("category"))
            raw_meta = {
                "slug": slug,
                "eventSlug": event_slug,
                "eventId": _to_clean_str(event.get("id")),
                "eventTicker": _to_clean_str(event.get("ticker")),
                "eventTitle": _to_clean_str(market.get("eventTitle") or event.get("title")),
                "category": _to_clean_str(market.get("category") or event_category),
                "eventCategory": event_category,
                "subCategory": _to_clean_str(
                    market.get("subCategory") or event.get("subCategory") or event.get("subcategory")
                ),
                "tags": _extract_tags(market.get("tags")) or _extract_tags(event.get("tags")),
            }
            token_map[yes_token_id] = _MarketMeta(
                market_id=market_id,
                question=question,
                url=url,
                liquidity=_to_float(market.get("liquidityNum") or market.get("liquidity") or 0),
                volume_24h=_to_float(market.get("volume24hr") or market.get("volume24hrClob") or 0),
                updated_at=_parse_dt(market.get("updatedAt") or event.get("updatedAt")),
                raw=raw_meta,
            )

        return token_map

    def _parse_event(self, event: Dict[str, Any], token_map: Dict[str, _MarketMeta]) -> Optional[PredictionSignal]:
        token_id = str(
            event.get("asset_id")
            or event.get("assetId")
            or event.get("token_id")
            or event.get("tokenId")
            or ""
        )
        if not token_id:
            return None

        meta = token_map.get(token_id)
        if not meta:
            return None

        prob_yes = _extract_probability(event)
        if prob_yes is None:
            return None

        now = datetime.now(timezone.utc)
        raw = dict(meta.raw)
        raw["stream_event"] = event
        return PredictionSignal(
            source=self.source_name,
            market_id=meta.market_id,
            question=meta.question,
            url=meta.url,
            prob_yes=prob_yes,
            volume_24h=meta.volume_24h,
            liquidity=meta.liquidity,
            updated_at=now,
            raw=raw,
        )


def _to_event_list(raw: str) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        # Some websocket feeds nest updates under a key.
        if "events" in payload and isinstance(payload["events"], list):
            return [x for x in payload["events"] if isinstance(x, dict)]
        return [payload]
    return []


def _extract_probability(event: Dict[str, Any]) -> Optional[float]:
    for key in ("price", "mid", "midpoint", "last_trade_price", "best_bid", "best_ask"):
        if key in event:
            value = _normalize_probability(event[key])
            if value is not None:
                return value

    bid = _to_float(event.get("best_bid"))
    ask = _to_float(event.get("best_ask"))
    if bid > 0 and ask > 0:
        midpoint = _normalize_probability((bid + ask) / 2)
        if midpoint is not None:
            return midpoint

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


def _primary_event(market: Dict[str, Any]) -> Dict[str, Any]:
    raw_events = market.get("events")
    if isinstance(raw_events, list):
        for event in raw_events:
            if isinstance(event, dict):
                return event
    if isinstance(raw_events, dict):
        return raw_events
    return {}


def _extract_tags(raw_tags: Any) -> List[str]:
    if not raw_tags:
        return []
    if isinstance(raw_tags, str):
        return _extract_tags(_parse_json_list(raw_tags))
    if not isinstance(raw_tags, list):
        return []

    out: List[str] = []
    for item in raw_tags:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                out.append(cleaned)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("slug", "name", "label", "tag"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
                break
    return out


def _parse_dt(raw: Any) -> datetime:
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)
