from __future__ import annotations

from typing import Awaitable, Callable

from prediction_agent.models import PredictionSignal

SignalHandler = Callable[[PredictionSignal], Awaitable[None]]


class SignalStreamer:
    source_name: str = "unknown"

    async def stream(self, on_signal: SignalHandler) -> None:
        raise NotImplementedError
