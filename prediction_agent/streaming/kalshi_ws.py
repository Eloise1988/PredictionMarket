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


class KalshiWebsocketStreamer(SignalStreamer):
    source_name = "kalshi"

    def __init__(
        self,
        rest_base_url: str,
        ws_url: str,
        market_limit: int = 100,
        channel: str = "ticker",
        timeout: int = 15,
    ):
        self.rest_base_url = rest_base_url.rstrip("/")
        self.ws_url = ws_url
        self.market_limit = market_limit
        self.channel = channel
        self.http = HttpClient(timeout=timeout)

    async def stream(self, on_signal: SignalHandler) -> None:
        while True:
            try:
                market_map = self._load_market_map()
                if not market_map:
                    await asyncio.sleep(5)
                    continue

                tickers = list(market_map.keys())
                logger.info("Kalshi websocket subscribe", extra={"markets": len(tickers), "channel": self.channel})

                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20, max_size=2**20) as ws:
                    await self._subscribe(ws, tickers)

                    async for raw in ws:
                        for event in _to_event_list(raw):
                            signal = self._parse_event(event, market_map)
                            if signal:
                                await on_signal(signal)
            except Exception as exc:
                logger.warning("Kalshi websocket disconnected", extra={"error": str(exc)})
                await asyncio.sleep(3)

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol, market_tickers: List[str]) -> None:
        # Kalshi websocket docs changed channel names over time.
        # We send the current format and a compatibility fallback.
        primary = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": [self.channel],
                "market_tickers": market_tickers,
            },
        }
        fallback = {
            "id": 2,
            "type": "subscribe",
            "channel": self.channel,
            "market_tickers": market_tickers,
        }

        await ws.send(json.dumps(primary))
        await ws.send(json.dumps(fallback))

    def _load_market_map(self) -> Dict[str, _MarketMeta]:
        params = {"status": "open", "limit": self.market_limit}
        payload = self.http.get_json(f"{self.rest_base_url}/markets", params=params)
        markets = payload.get("markets", []) if isinstance(payload, dict) else []

        out: Dict[str, _MarketMeta] = {}
        for market in markets:
            ticker = str(market.get("ticker") or market.get("id") or "")
            if not ticker:
                continue

            question = (market.get("title") or market.get("name") or "").strip()
            if not question:
                continue

            event_ticker = market.get("event_ticker") or ""
            slug = market.get("slug") or ""
            if slug:
                url = f"https://kalshi.com/markets/{slug}"
            elif event_ticker:
                url = f"https://kalshi.com/markets/{event_ticker.lower()}"
            else:
                url = "https://kalshi.com/markets"

            out[ticker] = _MarketMeta(
                market_id=ticker,
                question=question,
                url=url,
                liquidity=_to_float(market.get("open_interest") or market.get("liquidity") or 0),
                volume_24h=_to_float(market.get("volume_24h") or market.get("volume") or 0),
            )

        return out

    def _parse_event(self, event: Dict[str, Any], market_map: Dict[str, _MarketMeta]) -> Optional[PredictionSignal]:
        row = _unwrap_event(event)

        ticker = str(row.get("market_ticker") or row.get("ticker") or row.get("market") or "")
        if not ticker:
            return None

        meta = market_map.get(ticker)
        if not meta:
            return None

        prob_yes = _extract_yes_probability(row)
        if prob_yes is None:
            return None

        return PredictionSignal(
            source=self.source_name,
            market_id=meta.market_id,
            question=meta.question,
            url=meta.url,
            prob_yes=prob_yes,
            volume_24h=meta.volume_24h,
            liquidity=meta.liquidity,
            updated_at=datetime.now(timezone.utc),
            raw={"stream_event": row},
        )


def _to_event_list(raw: str) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if "msg" in payload and isinstance(payload["msg"], dict):
            return [payload]
        if "data" in payload and isinstance(payload["data"], list):
            return [x for x in payload["data"] if isinstance(x, dict)]
        return [payload]
    return []


def _unwrap_event(event: Dict[str, Any]) -> Dict[str, Any]:
    msg = event.get("msg")
    if isinstance(msg, dict):
        return msg
    return event


def _extract_yes_probability(row: Dict[str, Any]) -> Optional[float]:
    for key in ("yes_price", "yes_last_price", "yes_bid", "yes_ask", "last_price"):
        if key in row and row[key] is not None:
            value = _to_float(row[key])
            if value <= 0:
                continue
            return value if value <= 1 else value / 100.0

    bid = _to_float(row.get("yes_bid"))
    ask = _to_float(row.get("yes_ask"))
    if bid > 0 and ask > 0:
        midpoint = (bid + ask) / 2
        return midpoint if midpoint <= 1 else midpoint / 100.0

    no_bid = _to_float(row.get("no_bid"))
    if no_bid > 0:
        no_prob = no_bid if no_bid <= 1 else no_bid / 100.0
        return max(0.0, min(1.0, 1.0 - no_prob))

    return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
