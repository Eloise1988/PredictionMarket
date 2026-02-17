from __future__ import annotations

import math
from datetime import datetime, timezone

from prediction_agent.models import EquitySnapshot, PredictionSignal


def valuation_score(snapshot: EquitySnapshot | None) -> float:
    if snapshot is None:
        return 0.5

    points = []

    if snapshot.pe_ratio is None:
        points.append(0.5)
    elif snapshot.pe_ratio <= 0:
        points.append(0.2)
    else:
        points.append(_clamp((40 - snapshot.pe_ratio) / 40, 0.0, 1.0))

    if snapshot.pb_ratio is None:
        points.append(0.5)
    elif snapshot.pb_ratio <= 0:
        points.append(0.2)
    else:
        points.append(_clamp((8 - snapshot.pb_ratio) / 8, 0.0, 1.0))

    if snapshot.ev_to_ebitda is None:
        points.append(0.5)
    elif snapshot.ev_to_ebitda <= 0:
        points.append(0.2)
    else:
        points.append(_clamp((25 - snapshot.ev_to_ebitda) / 25, 0.0, 1.0))

    return sum(points) / len(points)


def signal_quality(signal: PredictionSignal) -> float:
    liq = max(signal.liquidity, 0.0)
    vol = max(signal.volume_24h, 0.0)
    age_hours = max((datetime.now(timezone.utc) - signal.updated_at).total_seconds() / 3600.0, 0.0)

    liq_norm = _clamp(math.log10(1 + liq) / 6, 0.0, 1.0)
    vol_norm = _clamp(math.log10(1 + vol) / 6, 0.0, 1.0)
    freshness = _clamp(1.0 - age_hours / 24.0, 0.0, 1.0)

    return 0.45 * liq_norm + 0.35 * vol_norm + 0.20 * freshness


def clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
