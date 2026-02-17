from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

from prediction_agent.models import CandidateIdea


def aggregate_consensus(ideas: List[CandidateIdea]) -> List[CandidateIdea]:
    buckets: Dict[Tuple[str, str, str], List[CandidateIdea]] = defaultdict(list)
    for idea in ideas:
        buckets[(idea.ticker, idea.direction, idea.event_theme)].append(idea)

    merged: List[CandidateIdea] = []
    for _, group in buckets.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        base = deepcopy(max(group, key=lambda i: i.score * i.confidence))
        weighted_score_num = 0.0
        weighted_score_den = 0.0
        weighted_prob_num = 0.0

        sources = set()
        markets = set()
        for g in group:
            weight = max(0.05, g.signal_quality * g.confidence)
            weighted_score_num += g.score * weight
            weighted_prob_num += g.event_probability * weight
            weighted_score_den += weight
            sources.add(g.market_source)
            markets.add(g.market_id)

        avg_score = weighted_score_num / weighted_score_den
        avg_prob = weighted_prob_num / weighted_score_den
        source_bonus = min(0.10, 0.03 * (len(sources) - 1))

        base.score = min(1.0, avg_score + source_bonus)
        base.event_probability = avg_prob
        base.confidence = min(1.0, base.confidence + source_bonus)
        base.metadata["consensus_source_count"] = len(sources)
        base.metadata["consensus_market_count"] = len(markets)
        base.metadata["consensus_group_size"] = len(group)
        base.rationale = (
            f"{base.rationale} Consensus confirmed across {len(markets)} market(s)"
            f" and {len(sources)} source(s)."
        )

        merged.append(base)

    return merged
