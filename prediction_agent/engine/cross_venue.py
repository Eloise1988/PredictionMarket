from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, List, Sequence

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
    match_score: float = 0.0
    match_type: str = "exact"
    differences: str = ""


def match_cross_venue_markets(
    polymarket_signals: List[PredictionSignal],
    kalshi_signals: List[PredictionSignal],
    min_similarity: float = 0.20,
    top_k: int = 20,
    use_llm_verifier: bool = False,
    llm_api_key: str = "",
    llm_model: str = "gpt-5-mini",
    llm_timeout_seconds: int = 20,
    max_llm_pairs: int = 80,
) -> List[CrossVenueMatch]:
    # Keep this matcher intentionally simple and deterministic.
    del top_k, use_llm_verifier, llm_api_key, llm_model, llm_timeout_seconds, max_llm_pairs

    pm = sorted(polymarket_signals, key=lambda s: (s.liquidity, s.volume_24h), reverse=True)
    ks = sorted(kalshi_signals, key=lambda s: (s.liquidity, s.volume_24h), reverse=True)

    ks_tokens = [_question_token_set(_signal_text(s)) for s in ks]
    used_kalshi: set[int] = set()
    matches: List[CrossVenueMatch] = []

    for p in pm:
        p_text = _signal_text(p)
        p_tokens = _question_token_set(p_text)
        if not p_tokens:
            continue

        best_idx = -1
        best_similarity = 0.0
        best_differences = ""
        for idx, k in enumerate(ks):
            if idx in used_kalshi:
                continue

            ok, diffs = _is_semantically_compatible(p, k)
            if not ok:
                continue

            sim = _similarity_score(p_tokens, ks_tokens[idx])
            if sim < min_similarity:
                continue
            if sim > best_similarity:
                best_similarity = sim
                best_idx = idx
                best_differences = diffs

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
                match_score=best_similarity,
                match_type="exact",
                differences=best_differences,
            )
        )

    matches.sort(
        key=lambda m: (m.liquidity_sum, m.text_similarity, m.probability_diff),
        reverse=True,
    )
    return matches


def _is_semantically_compatible(pm: PredictionSignal, ks: PredictionSignal) -> tuple[bool, str]:
    pm_text = _signal_text(pm)
    ks_text = _signal_text(ks)
    pm_tokens = _question_token_set(pm_text)
    ks_tokens = _question_token_set(ks_text)

    pm_entities = _extract_entities(pm_text, pm_tokens)
    ks_entities = _extract_entities(ks_text, ks_tokens)
    if pm_entities and ks_entities and pm_entities.isdisjoint(ks_entities):
        return False, "entity"

    pm_action = _action_groups(pm_text, pm_tokens)
    ks_action = _action_groups(ks_text, ks_tokens)
    if pm_action and ks_action and pm_action.isdisjoint(ks_action):
        return False, "action"

    pm_type = _event_type(pm_text, pm_tokens, pm_action)
    ks_type = _event_type(ks_text, ks_tokens, ks_action)
    if pm_type != "generic" and ks_type != "generic" and pm_type != ks_type:
        return False, "type"

    pm_dir = _rate_direction(pm_text, pm_tokens)
    ks_dir = _rate_direction(ks_text, ks_tokens)
    if pm_dir and ks_dir and pm_dir != ks_dir:
        return False, "direction"

    pm_bps = _basis_point_values(pm_text)
    ks_bps = _basis_point_values(ks_text)
    if pm_bps and ks_bps and pm_bps.isdisjoint(ks_bps):
        return False, "bps"

    pm_price = _price_targets(pm_text)
    ks_price = _price_targets(ks_text)
    if pm_price and ks_price and pm_price.isdisjoint(ks_price):
        return False, "price_threshold"

    pm_start, pm_end = _window_bounds(pm)
    ks_start, ks_end = _window_bounds(ks)
    if pm_end and ks_end and abs((pm_end - ks_end).days) > 2:
        return False, "end_date"
    if pm_start and ks_start and abs((pm_start - ks_start).days) > 35:
        return False, "start_date"

    # Fall back to textual month compatibility only when concrete end bounds are unavailable.
    if not (pm_end and ks_end):
        pm_months = _text_month_keys(pm_text)
        ks_months = _text_month_keys(ks_text)
        if pm_months and ks_months and pm_months.isdisjoint(ks_months):
            return False, "time_window"

    return True, ""


def _signal_text(signal: PredictionSignal) -> str:
    raw = signal.raw or {}
    # Exclude tags/categories here to avoid false entity leakage.
    return " ".join(
        [
            str(signal.question or ""),
            str(raw.get("eventTitle") or raw.get("event_title") or ""),
            str(raw.get("title") or ""),
            str(raw.get("subtitle") or raw.get("event_subtitle") or ""),
            str(raw.get("yes_sub_title") or raw.get("yes_subtitle") or ""),
            str(raw.get("no_sub_title") or raw.get("no_subtitle") or ""),
        ]
    ).lower()


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


def _extract_entities(text: str, tokens: set[str]) -> set[str]:
    padded = f" {_normalize_text(text)} "
    out: set[str] = set()
    for canonical, aliases in _ENTITY_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            if not alias_norm:
                continue
            if " " in alias_norm:
                if f" {alias_norm} " in padded:
                    out.add(canonical)
                    break
            elif alias_norm in tokens:
                out.add(canonical)
                break
    return out


def _action_groups(text: str, tokens: set[str]) -> set[str]:
    padded = f" {_normalize_text(text)} "
    out: set[str] = set()
    for group, aliases in _ACTION_GROUP_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            if not alias_norm:
                continue
            if " " in alias_norm:
                if f" {alias_norm} " in padded:
                    out.add(group)
                    break
            elif alias_norm in tokens:
                out.add(group)
                break
    return out


def _event_type(text: str, tokens: set[str], action_groups: set[str]) -> str:
    if "rate_decision" in action_groups:
        return "fed_rate_decision"
    if "clemency" in action_groups:
        return "clemency"
    if "leadership_outcome" in action_groups:
        return "leadership_outcome"
    if "leadership_change" in action_groups:
        return "leadership_change"
    if "military_action" in action_groups:
        return "military_action"
    if "price_target" in action_groups and bool(tokens.intersection({"bitcoin", "ethereum", "btc", "eth"})):
        return "crypto_price_target"
    return "generic"


def _rate_direction(text: str, tokens: set[str]) -> str:
    if not tokens.intersection({"fed", "fomc", "rate"}):
        return ""
    up = bool(tokens.intersection(_RATE_UP_TERMS))
    down = bool(tokens.intersection(_RATE_DOWN_TERMS))
    hold = bool(tokens.intersection(_RATE_HOLD_TERMS))
    if hold and not up and not down:
        return "hold"
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return ""


def _basis_point_values(text: str) -> set[int]:
    out: set[int] = set()
    for m in re.finditer(r"(\d+)\s*\+?\s*(?:bp|bps|basis\s+point(?:s)?)", text):
        try:
            out.add(int(m.group(1)))
        except (TypeError, ValueError):
            continue
    if _contains_any(text, _RATE_HOLD_TERMS):
        out.add(0)
    return out


def _price_targets(text: str) -> set[int]:
    out: set[int] = set()
    if not _contains_any(text, _PRICE_CONTEXT_TERMS):
        return out
    for m in re.finditer(r"(\$?\s*\d[\d,]*(?:\.\d+)?)\s*([kmb])?", text):
        raw_num = str(m.group(1) or "")
        suffix = str(m.group(2) or "").lower().strip()
        value = _parse_number(raw_num, suffix)
        if value is None:
            continue
        # Drop obvious date year mentions.
        if 1900 <= value <= 2100 and "$" not in raw_num and suffix == "":
            continue
        out.add(int(round(value)))
    return out


def _window_bounds(signal: PredictionSignal) -> tuple[date | None, date | None]:
    raw = signal.raw or {}
    start = _parse_date(
        raw.get("startDate")
        or raw.get("start_date")
        or raw.get("open_ts")
        or raw.get("acceptingOrdersTimestamp")
    )
    end = _parse_date(
        raw.get("endDate")
        or raw.get("end_date")
        or raw.get("close_time")
        or raw.get("close_ts")
        or raw.get("expected_expiration_ts")
        or raw.get("umaEndDate")
    )
    return start, end


def _text_month_keys(text: str) -> set[str]:
    out: set[str] = set()
    pattern = (
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"(?:\s+(\d{4}))?"
    )
    for m in re.finditer(pattern, text):
        mon = str(m.group(1) or "").lower()[:3]
        year_raw = str(m.group(2) or "").strip()
        month = _MONTHS.get(mon)
        if month is None:
            continue
        if year_raw.isdigit():
            out.add(f"{int(year_raw):04d}-{month:02d}")
        out.add(f"m-{month:02d}")
    return out


def _similarity_score(a: set[str], b: set[str]) -> float:
    j = _jaccard_similarity(a, b)
    o = _overlap_similarity(a, b)
    return max(j, 0.70 * o + 0.30 * j)


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


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    normalized = f" {_normalize_text(text)} "
    for term in terms:
        t = _normalize_text(term)
        if not t:
            continue
        if f" {t} " in normalized:
            return True
    return False


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _parse_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    raw = str(value).strip()
    if not raw:
        return None
    cleaned = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned).date()
    except ValueError:
        pass
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", cleaned)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _parse_number(raw: str, suffix: str) -> float | None:
    try:
        value = float(raw.replace("$", "").replace(",", "").strip())
    except ValueError:
        return None
    mult = 1.0
    if suffix == "k":
        mult = 1_000.0
    elif suffix == "m":
        mult = 1_000_000.0
    elif suffix == "b":
        mult = 1_000_000_000.0
    return value * mult


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
    "confirmed": "confirm",
}

_RATE_UP_TERMS = {"hike", "hikes", "increase", "increases", "raise", "raises", "higher", "up"}
_RATE_DOWN_TERMS = {"cut", "cuts", "decrease", "decreases", "lower", "lowers", "reduce", "reduces", "down"}
_RATE_HOLD_TERMS = {"hold", "holds", "unchanged", "pause", "paused", "steady", "maintain", "maintains"}
_PRICE_CONTEXT_TERMS = {
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "price",
    "reach",
    "hit",
    "above",
    "below",
    "over",
    "under",
    "close",
    "cross",
}

_ENTITY_ALIASES = {
    "bitcoin": {"bitcoin", "btc"},
    "ethereum": {"ethereum", "eth"},
    "fed": {"fed", "fomc", "federal reserve"},
    "us": {"us", "united states", "u s", "u.s"},
    "iran": {"iran", "iranian"},
    "trump": {"trump", "donald trump"},
    "ali_khamenei": {"ali khamenei", "khamenei"},
    "cpi": {"cpi", "inflation"},
    "epstein": {"epstein", "jeffrey epstein"},
    "venezuela": {"venezuela"},
}

_ACTION_GROUP_ALIASES = {
    "clemency": {"pardon", "pardons", "commute", "commutes", "reprieve", "reprieves"},
    "leadership_outcome": {"who will lead", "lead", "leader", "president", "prime minister"},
    "leadership_change": {"out", "removed", "remove", "resign", "resigns", "die", "dies", "deposed"},
    "rate_decision": {"fed", "fomc", "rate", "hike", "cut", "hold", "unchanged", "maintains"},
    "price_target": {"reach", "hit", "cross", "above", "below", "over", "under", "close"},
    "military_action": {"strike", "strikes", "attack", "attacks"},
    "disclosure_confirmation": {"confirm", "confirms", "confirmed", "announce", "announces", "alive"},
}

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
