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
        params = {"closed": "false", "active": "true", "limit": self.market_limit}
        markets = self.http.get_json(f"{self.gamma_base_url}/markets", params=params)
        if not isinstance(markets, list):
            return {}

        token_map: Dict[str, _MarketMeta] = {}
        for market in markets:
            question = (market.get("question") or "").strip()
            if not question:
                continue

            token_ids = _parse_json_list(market.get("clobTokenIds"))
            if not token_ids:
                continue

            yes_token_id = str(token_ids[0])
            market_id = str(market.get("id") or market.get("conditionId") or "")
            if not market_id:
                continue

            slug = market.get("slug") or ""
            url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com"
            token_map[yes_token_id] = _MarketMeta(
                market_id=market_id,
                question=question,
                url=url,
                liquidity=_to_float(market.get("liquidityNum") or market.get("liquidity") or 0),
                volume_24h=_to_float(market.get("volume24hr") or market.get("volume24hrClob") or 0),
                updated_at=_parse_dt(market.get("updatedAt")),
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
        return PredictionSignal(
            source=self.source_name,
            market_id=meta.market_id,
            question=meta.question,
            url=meta.url,
            prob_yes=prob_yes,
            volume_24h=meta.volume_24h,
            liquidity=meta.liquidity,
            updated_at=now,
            raw={"stream_event": event},
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
            value = _to_float(event[key])
            if value <= 0:
                continue
            return value if value <= 1 else value / 100.0

    bid = _to_float(event.get("best_bid"))
    ask = _to_float(event.get("best_ask"))
    if bid > 0 and ask > 0:
        midpoint = (bid + ask) / 2
        return midpoint if midpoint <= 1 else midpoint / 100.0

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
