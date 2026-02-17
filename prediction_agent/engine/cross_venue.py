from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from prediction_agent.models import PredictionSignal
else:
    PredictionSignal = Any


@dataclass
class CrossVenueMatch:
    polymarket: PredictionSignal
    kalshi: PredictionSignal
    text_similarity: float
    probability_diff: float
    liquidity_sum: float


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "will",
    "before",
    "after",
    "into",
    "from",
    "that",
    "this",
    "have",
    "has",
    "had",
    "are",
    "was",
    "were",
    "would",
    "could",
    "should",
    "about",
    "next",
    "than",
    "over",
    "under",
    "between",
    "more",
    "less",
    "year",
}


def match_cross_venue_markets(
    polymarket_signals: List[PredictionSignal],
    kalshi_signals: List[PredictionSignal],
    min_similarity: float = 0.34,
) -> List[CrossVenueMatch]:
    pm = sorted(polymarket_signals, key=lambda s: (s.liquidity, s.volume_24h), reverse=True)
    ks = sorted(kalshi_signals, key=lambda s: (s.liquidity, s.volume_24h), reverse=True)

    ks_tokens = [_question_token_set(s.question) for s in ks]
    used_kalshi: set[int] = set()
    matches: List[CrossVenueMatch] = []

    for p in pm:
        p_tokens = _question_token_set(p.question)
        if not p_tokens:
            continue

        best_idx = -1
        best_similarity = 0.0
        for idx, k in enumerate(ks):
            if idx in used_kalshi:
                continue
            sim = _jaccard_similarity(p_tokens, ks_tokens[idx])
            if sim < min_similarity:
                continue
            if sim > best_similarity:
                best_similarity = sim
                best_idx = idx

        if best_idx < 0:
            continue

        used_kalshi.add(best_idx)
        k = ks[best_idx]
        matches.append(
            CrossVenueMatch(
                polymarket=p,
                kalshi=k,
                text_similarity=best_similarity,
                probability_diff=abs(p.prob_yes - k.prob_yes),
                liquidity_sum=p.liquidity + k.liquidity,
            )
        )

    matches.sort(
        key=lambda m: (m.liquidity_sum, m.text_similarity, m.probability_diff),
        reverse=True,
    )
    return matches


def _question_token_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union == 0:
        return 0.0
    return inter / union
