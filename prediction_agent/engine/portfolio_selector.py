from __future__ import annotations

from collections import defaultdict
from typing import List

from prediction_agent.models import CandidateIdea


def select_top_ideas(
    ideas: List[CandidateIdea],
    top_n: int,
    min_score: float,
    max_per_ticker: int = 1,
    max_per_theme: int = 2,
) -> List[CandidateIdea]:
    ranked = sorted(ideas, key=lambda x: (x.score * x.confidence, x.score), reverse=True)
    ticker_counts = defaultdict(int)
    theme_counts = defaultdict(int)

    selected: List[CandidateIdea] = []
    for idea in ranked:
        if idea.score < min_score:
            continue
        if ticker_counts[idea.ticker] >= max_per_ticker:
            continue
        if theme_counts[idea.event_theme] >= max_per_theme:
            continue

        selected.append(idea)
        ticker_counts[idea.ticker] += 1
        theme_counts[idea.event_theme] += 1

        if len(selected) >= top_n:
            break

    return selected
