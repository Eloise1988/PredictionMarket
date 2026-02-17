from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List

from prediction_agent.knowledge.ticker_profiles import get_ticker_background
from prediction_agent.models import CandidateIdea, EquitySnapshot, PredictionSignal


def build_candidate_ideas(
    signal: PredictionSignal,
    theme: str,
    equities_cfg: List[dict],
    snapshots: Dict[str, EquitySnapshot],
) -> List[CandidateIdea]:
    created_at = datetime.now(timezone.utc)
    probability_edge = abs(signal.prob_yes - 0.5) * 2
    signal_quality = _signal_quality(signal)

    ideas: List[CandidateIdea] = []
    for equity in equities_cfg:
        ticker = str(equity.get("ticker", "")).upper()
        if not ticker:
            continue
        exposure_direction = str(equity.get("direction", "long")).lower()
        exposure_sign = 1 if exposure_direction == "long" else -1
        event_sign = 1 if signal.prob_yes >= 0.5 else -1

        final_direction = "long" if event_sign * exposure_sign > 0 else "short"
        exposure_weight = _clamp(float(equity.get("weight", 0.5)), 0.05, 1.0)

        snap = snapshots.get(ticker)
        company_name = (snap.name if snap and snap.name else "").strip()
        sector = (snap.sector if snap and snap.sector else "").strip()
        background = get_ticker_background(ticker, company_name=company_name, sector=sector)
        valuation_raw = _valuation_score(snap)
        momentum_raw = _momentum_score(snap)

        valuation_score = valuation_raw if final_direction == "long" else 1.0 - valuation_raw
        momentum_score = momentum_raw if final_direction == "long" else 1.0 - momentum_raw

        score = (
            0.40 * probability_edge
            + 0.20 * signal_quality
            + 0.15 * exposure_weight
            + 0.15 * valuation_score
            + 0.10 * momentum_score
        )

        confidence = _clamp(0.55 * probability_edge + 0.30 * signal_quality + 0.15 * exposure_weight, 0.0, 1.0)

        rationale = (
            f"{theme}: market YES probability={signal.prob_yes:.1%}, edge={probability_edge:.2f}, "
            f"exposure={exposure_weight:.2f}, quality={signal_quality:.2f}."
        )

        ideas.append(
            CandidateIdea(
                ticker=ticker,
                direction=final_direction,
                score=_clamp(score, 0.0, 1.0),
                event_theme=theme,
                event_probability=signal.prob_yes,
                signal_quality=signal_quality,
                valuation_score=valuation_score,
                momentum_score=momentum_score,
                exposure_weight=exposure_weight,
                probability_edge=probability_edge,
                confidence=confidence,
                market_source=signal.source,
                market_id=signal.market_id,
                market_question=signal.question,
                market_url=signal.url,
                rationale=rationale,
                created_at=created_at,
                metadata={
                    "raw_exposure_direction": exposure_direction,
                    "snapshot_present": snap is not None,
                    "company_name": company_name,
                    "sector": sector,
                    "company_background": background,
                    "probability_source": str(signal.raw.get("probability_source", "")),
                },
            )
        )

    return ideas


def _signal_quality(signal: PredictionSignal) -> float:
    liq = max(signal.liquidity, 0.0)
    vol = max(signal.volume_24h, 0.0)
    age_hours = max((datetime.now(timezone.utc) - signal.updated_at).total_seconds() / 3600.0, 0.0)

    liq_norm = _clamp(math.log10(1 + liq) / 6, 0.0, 1.0)
    vol_norm = _clamp(math.log10(1 + vol) / 6, 0.0, 1.0)
    freshness = _clamp(1.0 - age_hours / 24.0, 0.0, 1.0)

    return 0.45 * liq_norm + 0.35 * vol_norm + 0.20 * freshness


def _valuation_score(snap: EquitySnapshot | None) -> float:
    if snap is None:
        return 0.5

    points = []

    if snap.pe_ratio is None:
        points.append(0.5)
    elif snap.pe_ratio <= 0:
        points.append(0.2)
    else:
        points.append(_clamp((40 - snap.pe_ratio) / 40, 0.0, 1.0))

    if snap.pb_ratio is None:
        points.append(0.5)
    elif snap.pb_ratio <= 0:
        points.append(0.2)
    else:
        points.append(_clamp((8 - snap.pb_ratio) / 8, 0.0, 1.0))

    if snap.ev_to_ebitda is None:
        points.append(0.5)
    elif snap.ev_to_ebitda <= 0:
        points.append(0.2)
    else:
        points.append(_clamp((25 - snap.ev_to_ebitda) / 25, 0.0, 1.0))

    return sum(points) / len(points)


def _momentum_score(snap: EquitySnapshot | None) -> float:
    if snap is None or snap.change_percent is None:
        return 0.5
    # +/-10% day move maps to 0..1 band and saturates outside.
    return _clamp((snap.change_percent + 0.10) / 0.20, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
