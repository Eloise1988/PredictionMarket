from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Protocol, Tuple


class SignalLike(Protocol):
    question: str
    prob_yes: float

_SPORTS_OR_ENTERTAINMENT_TERMS = (
    "fifa",
    "world cup",
    "stanley cup",
    "nba",
    "nfl",
    "mlb",
    "nhl",
    "uefa",
    "champions league",
    "olympic",
    "super bowl",
    "grand slam",
    "tournament",
    "playoff",
    "match",
    "goal",
)

_THEME_REQUIRE_ANY: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    # Require actor + conflict context to avoid matching sports markets about Iran.
    "us_iran_conflict": (
        ("iran", "tehran", "hormuz"),
        ("war", "strike", "attack", "military", "conflict", "missile", "sanction"),
    ),
    # Require explicit Fed/policy context to avoid substring collisions (e.g., "federal").
    "fed_rate_cut_cycle": (
        ("fed", "fomc", "federal reserve"),
        ("rate", "rate cut", "easing", "policy", "interest"),
    ),
}


class ThemeMatcher:
    def __init__(self, mapping_path: Path):
        self.mapping_path = mapping_path
        with mapping_path.open("r", encoding="utf-8") as f:
            self.mapping: Dict[str, Dict] = json.load(f)

    def match(self, signal: SignalLike, strict: bool = True) -> List[Tuple[str, Dict]]:
        text = (signal.question or "").lower().strip()
        if _is_sports_or_entertainment_market(text):
            return []

        matches: List[Tuple[str, Dict]] = []

        for theme, cfg in self.mapping.items():
            keywords = cfg.get("keywords", [])
            for kw in keywords:
                if _keyword_match(text, str(kw)):
                    if not _theme_context_ok(theme, text):
                        continue
                    matches.append((theme, cfg))
                    break

        if matches:
            return matches
        if strict:
            return []

        # If strict matching is off, capture high-conviction generalized signals.
        if signal.prob_yes >= 0.75 or signal.prob_yes <= 0.25:
            return [("general_macro", {"equities": []})]
        return []


def _is_sports_or_entertainment_market(text: str) -> bool:
    return any(_keyword_match(text, term) for term in _SPORTS_OR_ENTERTAINMENT_TERMS)


def _theme_context_ok(theme: str, text: str) -> bool:
    groups = _THEME_REQUIRE_ANY.get(theme)
    if not groups:
        return True
    for group in groups:
        if not any(_keyword_match(text, token) for token in group):
            return False
    return True


def _keyword_match(text: str, keyword: str) -> bool:
    kw = keyword.lower().strip()
    if not kw:
        return False
    # Use word boundaries to prevent false positives like "fed" in "federal" or
    # "oil" in "oilers".
    pattern = rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])"
    return re.search(pattern, text) is not None
