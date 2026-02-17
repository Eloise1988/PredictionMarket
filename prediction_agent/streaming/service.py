from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Dict, List

from prediction_agent.app import DecisionAgent
from prediction_agent.config import get_settings
from prediction_agent.models import PredictionSignal
from prediction_agent.streaming.kalshi_ws import KalshiWebsocketStreamer
from prediction_agent.streaming.polymarket_ws import PolymarketWebsocketStreamer
from prediction_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class RealtimeStreamingService:
    def __init__(self):
        self.settings = get_settings()
        self.agent = DecisionAgent()
        self.streamers = []

        if self.settings.polymarket_enabled and self.settings.polymarket_ws_enabled:
            self.streamers.append(
                PolymarketWebsocketStreamer(
                    gamma_base_url=self.settings.polymarket_gamma_base_url,
                    ws_url=self.settings.polymarket_ws_url,
                    market_limit=self.settings.polymarket_ws_market_limit,
                )
            )

        if self.settings.kalshi_enabled and self.settings.kalshi_ws_enabled:
            self.streamers.append(
                KalshiWebsocketStreamer(
                    rest_base_url=self.settings.kalshi_base_url,
                    ws_url=self.settings.kalshi_ws_url,
                    market_limit=self.settings.kalshi_ws_market_limit,
                    channel=self.settings.kalshi_ws_channel,
                )
            )

    async def run(self, dry_run: bool, runtime_seconds: int = 0) -> None:
        if not self.streamers:
            logger.warning("No streamers enabled. Check ws settings.")
            return

        queue: asyncio.Queue[PredictionSignal] = asyncio.Queue(maxsize=self.settings.streaming_max_buffer)

        async def on_signal(signal: PredictionSignal) -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    pass
            await queue.put(signal)

        tasks = [asyncio.create_task(s.stream(on_signal)) for s in self.streamers]
        started = time.monotonic()

        try:
            while True:
                await asyncio.sleep(self.settings.streaming_batch_seconds)
                batch = _drain_queue(queue)
                if batch:
                    logger.info("Processing streamed batch", extra={"count": len(batch)})
                    self.agent.process_signals(batch, dry_run=dry_run)

                if runtime_seconds > 0 and (time.monotonic() - started) >= runtime_seconds:
                    logger.info("Stopping stream due to runtime limit")
                    break
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def _drain_queue(queue: asyncio.Queue[PredictionSignal]) -> List[PredictionSignal]:
    items: List[PredictionSignal] = []
    while True:
        try:
            item = queue.get_nowait()
            queue.task_done()
            items.append(item)
        except asyncio.QueueEmpty:
            break

    dedup: Dict[str, PredictionSignal] = {}
    for sig in items:
        key = f"{sig.source}:{sig.market_id}"
        existing = dedup.get(key)
        if existing is None or sig.updated_at > existing.updated_at:
            dedup[key] = sig

    return list(dedup.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime websocket ingestion service")
    parser.add_argument("--dry-run", action="store_true", help="Do not send Telegram alerts")
    parser.add_argument("--runtime-seconds", type=int, default=0, help="Optional runtime cap for smoke tests")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    service = RealtimeStreamingService()
    asyncio.run(service.run(dry_run=args.dry_run, runtime_seconds=args.runtime_seconds))


if __name__ == "__main__":
    main()
