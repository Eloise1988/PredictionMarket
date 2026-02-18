from __future__ import annotations

import calendar
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

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


@dataclass(frozen=True)
class _RangeConstraint:
    lower: Optional[float] = None
    lower_inclusive: bool = True
    upper: Optional[float] = None
    upper_inclusive: bool = True


@dataclass
class _MarketContext:
    tokens: set[str]
    resolution_dt: Optional[datetime]
    is_rate_market: bool
    rate_direction: str
    rate_bps_constraint: Optional[_RangeConstraint]
    crypto_asset: str
    crypto_direction: str
    crypto_target: Optional[float]


def match_cross_venue_markets(
    polymarket_signals: List[PredictionSignal],
    kalshi_signals: List[PredictionSignal],
    min_similarity: float = 0.20,
) -> List[CrossVenueMatch]:
    pm = sorted(polymarket_signals, key=lambda s: (s.liquidity, s.volume_24h), reverse=True)
    ks = sorted(kalshi_signals, key=lambda s: (s.liquidity, s.volume_24h), reverse=True)

    pm_contexts = [_build_market_context(s) for s in pm]
    ks_contexts = [_build_market_context(s) for s in ks]
    idf = _inverse_document_frequency([ctx.tokens for ctx in pm_contexts + ks_contexts])

    used_kalshi: set[int] = set()
    matches: List[CrossVenueMatch] = []

    for p_idx, p in enumerate(pm):
        p_ctx = pm_contexts[p_idx]
        if not p_ctx.tokens:
            continue

        best_idx = -1
        best_similarity = 0.0
        for k_idx, k in enumerate(ks):
            if k_idx in used_kalshi:
                continue

            k_ctx = ks_contexts[k_idx]
            if not _is_semantically_compatible(p_ctx, k_ctx):
                continue

            sim = _similarity_score(p_ctx.tokens, k_ctx.tokens, idf=idf)
            if sim < min_similarity:
                continue
            if sim > best_similarity:
                best_similarity = sim
                best_idx = k_idx

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


def _build_market_context(signal: PredictionSignal) -> _MarketContext:
    question = str(getattr(signal, "question", "") or "")
    tokens = _question_token_set(question)
    rate_direction = _rate_direction(question)
    if not rate_direction:
        rate_direction = _rate_direction_from_market_id(str(getattr(signal, "market_id", "") or ""))

    return _MarketContext(
        tokens=tokens,
        resolution_dt=_resolution_datetime(signal),
        is_rate_market=_looks_like_rate_decision(question),
        rate_direction=rate_direction,
        rate_bps_constraint=_basis_point_constraint(question),
        crypto_asset=_crypto_asset(tokens),
        crypto_direction=_crypto_direction(question),
        crypto_target=_crypto_target_price(question),
    )


def _is_semantically_compatible(a: _MarketContext, b: _MarketContext) -> bool:
    if not a.tokens or not b.tokens:
        return False

    # Require at least one meaningful shared token after normalization.
    if not a.tokens.intersection(b.tokens):
        return False

    if not _resolution_dates_compatible(a.resolution_dt, b.resolution_dt):
        return False

    if a.is_rate_market and b.is_rate_market and not _rate_markets_compatible(a, b):
        return False

    if not _crypto_markets_compatible(a, b):
        return False

    return True


def _resolution_dates_compatible(a: Optional[datetime], b: Optional[datetime]) -> bool:
    if a is None or b is None:
        return True
    if _is_month_boundary_equivalent(a, b):
        return True
    delta_days = abs((a - b).total_seconds()) / 86400.0
    return delta_days <= _MAX_RESOLUTION_GAP_DAYS


def _is_month_boundary_equivalent(a: datetime, b: datetime) -> bool:
    if a > b:
        a, b = b, a

    if b.day != 1:
        return False

    prev_year = b.year
    prev_month = b.month - 1
    if prev_month == 0:
        prev_month = 12
        prev_year -= 1

    prev_last_day = calendar.monthrange(prev_year, prev_month)[1]
    if a.year != prev_year or a.month != prev_month or a.day != prev_last_day:
        return False

    delta_days = abs((b - a).total_seconds()) / 86400.0
    return delta_days <= _MAX_MONTH_BOUNDARY_EQUIV_DAYS


def _rate_markets_compatible(a: _MarketContext, b: _MarketContext) -> bool:
    if a.rate_direction and b.rate_direction and a.rate_direction != b.rate_direction:
        return False

    if a.rate_bps_constraint and b.rate_bps_constraint:
        if not _constraints_overlap(a.rate_bps_constraint, b.rate_bps_constraint):
            return False

    return True


def _crypto_markets_compatible(a: _MarketContext, b: _MarketContext) -> bool:
    if not a.crypto_asset and not b.crypto_asset:
        return True

    if a.crypto_asset and b.crypto_asset and a.crypto_asset != b.crypto_asset:
        return False

    if not a.crypto_asset or not b.crypto_asset:
        return True

    if a.crypto_direction and b.crypto_direction and a.crypto_direction != b.crypto_direction:
        return False

    if a.crypto_target is not None and b.crypto_target is not None:
        rel_gap = abs(a.crypto_target - b.crypto_target) / max(a.crypto_target, b.crypto_target)
        if rel_gap > _MAX_CRYPTO_TARGET_REL_GAP:
            return False

    return True


def _question_token_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    out: set[str] = set()
    for tok in tokens:
        if tok.isdigit():
            continue
        if len(tok) <= 2 and tok not in _SHORT_ALLOWED_TOKENS:
            continue
        tok = _TOKEN_ALIASES.get(tok, tok)
        if tok in _STOPWORDS:
            continue
        out.add(tok)
    return out


def _inverse_document_frequency(token_sets: Iterable[set[str]]) -> Dict[str, float]:
    rows = [tokens for tokens in token_sets if tokens]
    if not rows:
        return {}

    total_docs = float(len(rows))
    doc_freq: Dict[str, int] = {}
    for tokens in rows:
        for tok in tokens:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    return {
        tok: math.log((1.0 + total_docs) / (1.0 + float(freq))) + 1.0
        for tok, freq in doc_freq.items()
    }


def _weighted_jaccard_similarity(a: set[str], b: set[str], idf: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0

    inter = a.intersection(b)
    union = a.union(b)
    inter_w = _token_weight_sum(inter, idf)
    union_w = _token_weight_sum(union, idf)
    if union_w <= 0.0:
        return 0.0
    return inter_w / union_w


def _weighted_overlap_similarity(a: set[str], b: set[str], idf: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0

    inter_w = _token_weight_sum(a.intersection(b), idf)
    denom = min(_token_weight_sum(a, idf), _token_weight_sum(b, idf))
    if denom <= 0.0:
        return 0.0
    return inter_w / denom


def _token_weight_sum(tokens: Iterable[str], idf: Dict[str, float]) -> float:
    return sum(idf.get(tok, 1.0) for tok in tokens)


def _similarity_score(a: set[str], b: set[str], idf: Optional[Dict[str, float]] = None) -> float:
    token_idf = idf or {}
    j = _weighted_jaccard_similarity(a, b, token_idf)
    o = _weighted_overlap_similarity(a, b, token_idf)
    return max(j, 0.75 * o + 0.25 * j)


def _resolution_datetime(signal: PredictionSignal) -> Optional[datetime]:
    raw = getattr(signal, "raw", {}) or {}
    if isinstance(raw, dict):
        for key in _RESOLUTION_RAW_KEYS:
            dt = _parse_datetime_value(raw.get(key))
            if dt is not None:
                return dt

    market_id = str(getattr(signal, "market_id", "") or "")
    dt = _parse_kalshi_ticker_datetime(market_id)
    if dt is not None:
        return dt

    question = str(getattr(signal, "question", "") or "")
    return _parse_resolution_datetime_from_text(question)


def _parse_datetime_value(value: object) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, (int, float)):
        return _parse_unix_timestamp(float(value))

    text = str(value).strip()
    if not text:
        return None

    if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", text):
        try:
            return _parse_unix_timestamp(float(text))
        except ValueError:
            return None

    iso = text
    if iso.endswith("Z"):
        iso = f"{iso[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(iso)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in _DATETIME_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


def _parse_unix_timestamp(value: float) -> Optional[datetime]:
    if value <= 0:
        return None

    ts = value
    if ts > 1e12:
        ts = ts / 1000.0

    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _parse_kalshi_ticker_datetime(market_id: str) -> Optional[datetime]:
    text = (market_id or "").upper()
    found: Optional[datetime] = None
    for match in re.finditer(r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})?", text):
        year = 2000 + int(match.group(1))
        month = _MONTH_TO_NUM.get(match.group(2), 0)
        if month <= 0:
            continue
        day = int(match.group(3)) if match.group(3) else 1
        try:
            found = datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            continue
    return found


def _parse_resolution_datetime_from_text(text: str) -> Optional[datetime]:
    lower = (text or "").lower()

    # Examples: "by March 1, 2026", "before March 1st 2026".
    date_match = re.search(
        rf"\b(?:by|before|on|until|through)?\s*({_MONTH_NAME_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(\d{{4}})\b",
        lower,
    )
    if date_match:
        month_name = date_match.group(1)
        day = int(date_match.group(2))
        year = int(date_match.group(3))
        month = _MONTH_NAME_TO_NUM.get(month_name, 0)
        if month > 0:
            try:
                return datetime(year, month, day, tzinfo=timezone.utc)
            except ValueError:
                return None

    # Examples: "by end of February 2026", "end of February".
    month_end_match = re.search(
        rf"\b(?:by|before|until|through)?\s*(?:the\s+)?end\s+of\s+(?:the\s+month\s+of\s+)?({_MONTH_NAME_PATTERN})(?:\s+(\d{{4}}))?\b",
        lower,
    )
    if month_end_match:
        month_name = month_end_match.group(1)
        year = _resolve_year_hint(month_end_match.group(2), lower)
        month = _MONTH_NAME_TO_NUM.get(month_name, 0)
        if year and month > 0:
            last_day = calendar.monthrange(year, month)[1]
            return datetime(year, month, last_day, tzinfo=timezone.utc)

    # Examples: "before July 2026", "by February 2026".
    month_year_match = re.search(rf"\b(by|before)?\s*({_MONTH_NAME_PATTERN})\s+(\d{{4}})\b", lower)
    if month_year_match:
        qualifier = str(month_year_match.group(1) or "").strip()
        month_name = month_year_match.group(2)
        year = int(month_year_match.group(3))
        month = _MONTH_NAME_TO_NUM.get(month_name, 0)
        if month <= 0:
            return None

        if qualifier == "by":
            last_day = calendar.monthrange(year, month)[1]
            return datetime(year, month, last_day, tzinfo=timezone.utc)
        return datetime(year, month, 1, tzinfo=timezone.utc)

    return None


def _resolve_year_hint(explicit_year: Optional[str], text: str) -> Optional[int]:
    if explicit_year:
        try:
            return int(explicit_year)
        except ValueError:
            return None

    fallback = re.search(r"\b(20\d{2})\b", text or "")
    if not fallback:
        return None
    return int(fallback.group(1))


def _looks_like_rate_decision(text: str) -> bool:
    tokens = _normalized_tokens(text)
    return bool(tokens.intersection(_RATE_DECISION_TERMS))


def _rate_direction(text: str) -> str:
    tokens = _normalized_tokens(text)
    has_up = bool(tokens.intersection(_RATE_UP_TERMS))
    has_down = bool(tokens.intersection(_RATE_DOWN_TERMS))
    has_flat = bool(tokens.intersection(_RATE_HOLD_TERMS))

    if "no" in tokens and "change" in tokens:
        has_flat = True

    if has_flat and not has_up and not has_down:
        return "hold"
    if has_up and not has_down:
        return "up"
    if has_down and not has_up:
        return "down"
    return ""


def _rate_direction_from_market_id(market_id: str) -> str:
    text = (market_id or "").upper()
    hold_code = re.search(r"-H0(?:[^0-9]|$)", text)
    if hold_code:
        return "hold"

    if re.search(r"-H\d+(?:[^0-9]|$)", text):
        return "up"
    if re.search(r"-C\d+(?:[^0-9]|$)", text):
        return "down"
    return ""


def _basis_point_constraint(text: str) -> Optional[_RangeConstraint]:
    lower_text = (text or "").lower()

    patterns = [
        (r"(\d+)\s*\+\s*(?:bp|bps|basis\s+point(?:s)?)", _constraint_ge),
        (r"(?:>=|at\s+least|minimum\s+of)\s*(\d+)\s*(?:bp|bps|basis\s+point(?:s)?)", _constraint_ge),
        (r"(?:>|more\s+than|greater\s+than|over)\s*(\d+)\s*(?:bp|bps|basis\s+point(?:s)?)", _constraint_gt),
        (r"(?:<=|at\s+most|up\s+to)\s*(\d+)\s*(?:bp|bps|basis\s+point(?:s)?)", _constraint_le),
        (r"(?:<|less\s+than|under)\s*(\d+)\s*(?:bp|bps|basis\s+point(?:s)?)", _constraint_lt),
    ]

    for pattern, builder in patterns:
        match = re.search(pattern, lower_text)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        return builder(value)

    values: List[float] = []
    for match in re.finditer(r"(\d+)\s*(?:bp|bps|basis\s+point(?:s)?)", lower_text):
        try:
            values.append(float(match.group(1)))
        except (TypeError, ValueError):
            continue

    if not values:
        return None

    uniq = sorted(set(values))
    if len(uniq) == 1:
        return _constraint_eq(uniq[0])
    return _RangeConstraint(lower=uniq[0], lower_inclusive=True, upper=uniq[-1], upper_inclusive=True)


def _constraint_eq(value: float) -> _RangeConstraint:
    return _RangeConstraint(lower=value, lower_inclusive=True, upper=value, upper_inclusive=True)


def _constraint_ge(value: float) -> _RangeConstraint:
    return _RangeConstraint(lower=value, lower_inclusive=True, upper=None, upper_inclusive=True)


def _constraint_gt(value: float) -> _RangeConstraint:
    return _RangeConstraint(lower=value, lower_inclusive=False, upper=None, upper_inclusive=True)


def _constraint_le(value: float) -> _RangeConstraint:
    return _RangeConstraint(lower=None, lower_inclusive=True, upper=value, upper_inclusive=True)


def _constraint_lt(value: float) -> _RangeConstraint:
    return _RangeConstraint(lower=None, lower_inclusive=True, upper=value, upper_inclusive=False)


def _constraints_overlap(a: _RangeConstraint, b: _RangeConstraint) -> bool:
    if a.lower is not None and b.upper is not None:
        if a.lower > b.upper:
            return False
        if a.lower == b.upper and (not a.lower_inclusive or not b.upper_inclusive):
            return False

    if b.lower is not None and a.upper is not None:
        if b.lower > a.upper:
            return False
        if b.lower == a.upper and (not b.lower_inclusive or not a.upper_inclusive):
            return False

    return True


def _crypto_asset(tokens: set[str]) -> str:
    for canonical, aliases in _CRYPTO_ASSET_ALIASES.items():
        if tokens.intersection(aliases):
            return canonical
    return ""


def _crypto_direction(text: str) -> str:
    tokens = _normalized_tokens(text)
    has_up = bool(tokens.intersection(_CRYPTO_UP_TERMS))
    has_down = bool(tokens.intersection(_CRYPTO_DOWN_TERMS))

    if has_up and not has_down:
        return "up"
    if has_down and not has_up:
        return "down"
    return ""


def _crypto_target_price(text: str) -> Optional[float]:
    lower = (text or "").lower()
    values: List[float] = []

    for match in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)([kmb])?", lower):
        parsed = _to_price_value(match.group(1), match.group(2))
        if parsed is not None:
            values.append(parsed)

    for match in re.finditer(r"\b([0-9]+(?:\.[0-9]+)?)\s*([kmb])\b", lower):
        parsed = _to_price_value(match.group(1), match.group(2))
        if parsed is not None:
            values.append(parsed)

    if not values:
        return None

    # Ignore small numbers that are not realistic crypto price targets.
    filtered = [value for value in values if value >= 1_000.0]
    if not filtered:
        return None

    direction = _crypto_direction(text)
    if direction == "down":
        return min(filtered)
    return max(filtered)


def _to_price_value(raw_value: str, suffix: Optional[str]) -> Optional[float]:
    clean = str(raw_value or "").replace(",", "").strip()
    if not clean:
        return None

    try:
        value = float(clean)
    except ValueError:
        return None

    suf = (suffix or "").lower()
    if suf == "k":
        value *= 1_000.0
    elif suf == "m":
        value *= 1_000_000.0
    elif suf == "b":
        value *= 1_000_000_000.0

    return value


def _normalized_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if not tok:
            continue
        out.add(_TOKEN_ALIASES.get(tok, tok))
    return out


_MAX_RESOLUTION_GAP_DAYS = 40.0
_MAX_MONTH_BOUNDARY_EQUIV_DAYS = 2.0
_MAX_CRYPTO_TARGET_REL_GAP = 0.025

_RESOLUTION_RAW_KEYS = (
    "endDate",
    "close_time",
    "close_ts",
    "expected_expiration_ts",
    "resolve_time",
    "resolveTime",
    "expires_at",
    "expiration",
)

_DATETIME_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d",
)

_MONTH_TO_NUM = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

_MONTH_NAME_TO_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_NAME_PATTERN = "january|february|march|april|may|june|july|august|september|october|november|december"

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "will",
    "would",
    "could",
    "should",
    "what",
    "when",
    "who",
    "where",
    "which",
    "during",
    "before",
    "after",
    "into",
    "from",
    "that",
    "this",
    "these",
    "those",
    "have",
    "has",
    "had",
    "are",
    "was",
    "were",
    "about",
    "next",
    "than",
    "over",
    "under",
    "between",
    "more",
    "less",
    "again",
    "then",
    "their",
    "there",
    "here",
    "month",
    "months",
    "year",
    "years",
    "end",
    "ending",
    "toward",
    "towards",
    "through",
    "across",
    "around",
    "today",
    "tomorrow",
    "yesterday",
    "say",
    "says",
    "said",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}

_SHORT_ALLOWED_TOKENS = {"ai", "us", "uk", "eu", "uae"}

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
    "maintain": "hold",
    "maintains": "hold",
    "maintained": "hold",
    "keeps": "hold",
    "keep": "hold",
    "stays": "hold",
    "remains": "hold",
}

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
    "hold",
    "maintain",
    "maintains",
    "bps",
    "bp",
    "basis",
    "point",
    "points",
}

_RATE_UP_TERMS = {"hike", "hikes", "increase", "increases", "raise", "raises", "higher", "up"}
_RATE_DOWN_TERMS = {"cut", "cuts", "decrease", "decreases", "lower", "lowers", "reduce", "reduces", "down"}
_RATE_HOLD_TERMS = {
    "hold",
    "holds",
    "unchanged",
    "pause",
    "paused",
    "steady",
    "maintain",
    "maintains",
    "keeps",
    "keep",
    "stays",
    "remain",
    "remains",
}

_CRYPTO_ASSET_ALIASES: Dict[str, set[str]] = {
    "bitcoin": {"bitcoin", "btc", "xbt"},
    "ethereum": {"ethereum", "eth"},
    "solana": {"solana", "sol"},
}

_CRYPTO_UP_TERMS = {
    "reach",
    "reaches",
    "hit",
    "hits",
    "cross",
    "crosses",
    "above",
    "over",
    "exceed",
    "exceeds",
    "touch",
    "touches",
    "climb",
    "climbs",
}

_CRYPTO_DOWN_TERMS = {
    "dip",
    "dips",
    "below",
    "under",
    "drop",
    "drops",
    "fall",
    "falls",
    "decline",
    "declines",
}
