from __future__ import annotations

from typing import List

from prediction_agent.models import PredictionSignal


class PredictionConnector:
    source_name: str = "unknown"

    def fetch_signals(self) -> List[PredictionSignal]:
        raise NotImplementedError
