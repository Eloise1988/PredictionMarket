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
    min_similarity: float = 0.20,
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
            if not _is_semantically_compatible(p.question, k.question):
                continue
            sim = _similarity_score(p_tokens, ks_tokens[idx])
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
    out: set[str] = set()
    for tok in tokens:
        if len(tok) <= 2:
            continue
        tok = _TOKEN_ALIASES.get(tok, tok)
        if tok in _STOPWORDS:
            continue
        out.add(tok)
    return out


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union == 0:
        return 0.0
    return inter / union


def _overlap_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    denom = min(len(a), len(b))
    if denom == 0:
        return 0.0
    return inter / denom


def _similarity_score(a: set[str], b: set[str]) -> float:
    j = _jaccard_similarity(a, b)
    o = _overlap_similarity(a, b)
    return max(j, 0.70 * o + 0.30 * j)


def _is_semantically_compatible(a_text: str, b_text: str) -> bool:
    a = (a_text or "").lower()
    b = (b_text or "").lower()

    if _looks_like_rate_decision(a) and _looks_like_rate_decision(b):
        a_dir = _rate_direction(a)
        b_dir = _rate_direction(b)
        if a_dir and b_dir and a_dir != b_dir:
            return False

        a_bps = _basis_point_values(a)
        b_bps = _basis_point_values(b)
        if a_bps and b_bps and not a_bps.intersection(b_bps):
            return False

    return True


def _looks_like_rate_decision(text: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", text))
    return bool(tokens.intersection(_RATE_DECISION_TERMS))


def _rate_direction(text: str) -> str:
    tokens = set(re.findall(r"[a-z0-9]+", text))
    has_up = bool(tokens.intersection(_RATE_UP_TERMS))
    has_down = bool(tokens.intersection(_RATE_DOWN_TERMS))
    has_flat = bool(tokens.intersection(_RATE_HOLD_TERMS))

    if has_flat and not has_up and not has_down:
        return "hold"
    if has_up and not has_down:
        return "up"
    if has_down and not has_up:
        return "down"
    return ""


def _basis_point_values(text: str) -> set[int]:
    out: set[int] = set()
    for match in re.finditer(r"(\d+)\s*\+?\s*(?:bp|bps|basis\s+point(?:s)?)", text):
        try:
            out.add(int(match.group(1)))
        except (TypeError, ValueError):
            continue
    return out


_RATE_DECISION_TERMS = {
    "fed",
    "fomc",
    "rate",
    "rates",
    "hike",
    "hikes",
    "cut",
    "cuts",
    "increase",
    "decrease",
    "raise",
    "lower",
    "bps",
    "bp",
    "basis",
    "point",
    "points",
}

_RATE_UP_TERMS = {"hike", "hikes", "increase", "increases", "raise", "raises", "higher", "up"}
_RATE_DOWN_TERMS = {"cut", "cuts", "decrease", "decreases", "lower", "lowers", "reduce", "reduces", "down"}
_RATE_HOLD_TERMS = {"hold", "holds", "unchanged", "pause", "paused", "steady"}


_TOKEN_ALIASES = {
    "u": "us",
    "usa": "us",
    "america": "us",
    "american": "us",
    "federal": "fed",
    "reserve": "fed",
    "rates": "rate",
    "inflation": "cpi",
    "consumer": "cpi",
    "price": "cpi",
    "prices": "cpi",
    "gdp": "economy",
    "recession": "economy",
    "jobs": "employment",
    "unemployment": "employment",
    "btc": "bitcoin",
    "eth": "ethereum",
    "nomination": "nominate",
    "nominee": "nominate",
    "elections": "election",
    "presidential": "president",
    "decrease": "cut",
    "decreases": "cut",
    "lower": "cut",
    "lowers": "cut",
    "reduce": "cut",
    "reduces": "cut",
    "increase": "hike",
    "increases": "hike",
    "raise": "hike",
    "raises": "hike",
}
