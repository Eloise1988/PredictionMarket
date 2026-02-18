from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from prediction_agent.models import PredictionSignal
else:
    PredictionSignal = Any

logger = logging.getLogger(__name__)


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


@dataclass
class _EventSignature:
    signal: PredictionSignal
    market_key: str
    normalized_text: str
    tokens: set[str]
    entities: set[str]
    event_type: str
    direction: str
    threshold_tokens: set[str]
    bps_tokens: set[str]
    price_tokens: set[str]
    time_tokens: set[str]
    month_tokens: set[str]
    start_date: date | None
    end_date: date | None
    family_key: str
    canonical: str
    vector: list[float]


@dataclass
class _CandidateEdge:
    pm_key: str
    ks_key: str
    retrieval_score: float
    final_score: float
    verification: "_VerificationResult"


@dataclass
class _VerificationResult:
    same_underlying_event: bool
    same_resolution_criteria: bool
    match_score: float
    match_type: str
    differences: str = ""


@dataclass
class _CandidateIndex:
    by_entity: Dict[str, set[str]] = field(default_factory=dict)
    by_event_type: Dict[str, set[str]] = field(default_factory=dict)
    by_threshold: Dict[str, set[str]] = field(default_factory=dict)
    by_month: Dict[str, set[str]] = field(default_factory=dict)
    by_family: Dict[str, set[str]] = field(default_factory=dict)
    all_keys: set[str] = field(default_factory=set)


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
    pm = sorted(polymarket_signals, key=lambda s: (s.liquidity, s.volume_24h), reverse=True)
    ks = sorted(kalshi_signals, key=lambda s: (s.liquidity, s.volume_24h), reverse=True)
    if not pm or not ks:
        return []

    pm_signatures = [_build_signature(s, idx=i) for i, s in enumerate(pm)]
    ks_signatures = [_build_signature(s, idx=i) for i, s in enumerate(ks)]
    pm_by_key = {sig.market_key: sig for sig in pm_signatures}
    ks_by_key = {sig.market_key: sig for sig in ks_signatures}
    index = _build_candidate_index(ks_signatures)

    verifier = _LLMCandidateVerifier(
        enabled=use_llm_verifier,
        api_key=llm_api_key,
        model=llm_model,
        timeout_seconds=llm_timeout_seconds,
    )
    llm_budget = max(0, int(max_llm_pairs))
    top_k = max(1, int(top_k))

    candidate_edges: list[_CandidateEdge] = []
    llm_indices: list[int] = []
    for pm_sig in pm_signatures:
        candidates = _retrieve_top_candidates(
            pm_sig=pm_sig,
            ks_by_key=ks_by_key,
            index=index,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        for ks_key, retrieval_score in candidates:
            ks_sig = ks_by_key[ks_key]
            heur = _heuristic_verify(pm_sig, ks_sig, retrieval_score)
            if not heur.same_underlying_event:
                continue

            candidate_edges.append(
                _CandidateEdge(
                    pm_key=pm_sig.market_key,
                    ks_key=ks_key,
                    retrieval_score=retrieval_score,
                    final_score=0.0,
                    verification=heur,
                )
            )
            if verifier.enabled() and _should_call_llm(heur, retrieval_score):
                llm_indices.append(len(candidate_edges) - 1)

    if verifier.enabled() and llm_budget > 0 and llm_indices:
        llm_indices.sort(key=lambda idx: candidate_edges[idx].retrieval_score, reverse=True)
        selected_indices = llm_indices[:llm_budget]
        batch_pairs = [
            (
                str(idx),
                pm_by_key[candidate_edges[idx].pm_key],
                ks_by_key[candidate_edges[idx].ks_key],
            )
            for idx in selected_indices
        ]
        llm_results = verifier.verify_batch(batch_pairs)
        for idx in selected_indices:
            llm_res = llm_results.get(str(idx))
            if llm_res is None:
                continue
            edge = candidate_edges[idx]
            edge.verification = _merge_verification(edge.verification, llm_res)

    edges: list[_CandidateEdge] = []
    for edge in candidate_edges:
        verify = edge.verification
        if not verify.same_underlying_event or not verify.same_resolution_criteria:
            continue
        final_score = _clamp01(0.50 * edge.retrieval_score + 0.50 * verify.match_score)
        if final_score < min_similarity:
            continue
        edge.final_score = final_score
        edges.append(edge)

    # Greedy one-to-one assignment over sorted candidate edges.
    edges.sort(
        key=lambda e: (
            e.final_score,
            e.retrieval_score,
            pm_by_key[e.pm_key].signal.liquidity + ks_by_key[e.ks_key].signal.liquidity,
            -abs(pm_by_key[e.pm_key].signal.prob_yes - ks_by_key[e.ks_key].signal.prob_yes),
        ),
        reverse=True,
    )

    used_pm: set[str] = set()
    used_ks: set[str] = set()
    matches: list[CrossVenueMatch] = []

    for edge in edges:
        if edge.pm_key in used_pm or edge.ks_key in used_ks:
            continue

        pm_sig = pm_by_key[edge.pm_key]
        ks_sig = ks_by_key[edge.ks_key]
        used_pm.add(edge.pm_key)
        used_ks.add(edge.ks_key)
        matches.append(
            CrossVenueMatch(
                polymarket=pm_sig.signal,
                kalshi=ks_sig.signal,
                text_similarity=edge.final_score,
                probability_diff=abs(pm_sig.signal.prob_yes - ks_sig.signal.prob_yes),
                liquidity_sum=pm_sig.signal.liquidity + ks_sig.signal.liquidity,
                match_score=edge.verification.match_score,
                match_type=edge.verification.match_type,
                differences=edge.verification.differences,
            )
        )

    matches.sort(
        key=lambda m: (m.liquidity_sum, m.match_score, m.text_similarity, m.probability_diff),
        reverse=True,
    )
    return matches


def _build_signature(signal: PredictionSignal, idx: int) -> _EventSignature:
    raw = signal.raw or {}
    text_blob = " ".join(
        [
            str(signal.question or ""),
            str(raw.get("eventTitle") or raw.get("event_title") or ""),
            str(raw.get("title") or ""),
            str(raw.get("subtitle") or raw.get("event_subtitle") or ""),
            str(raw.get("category") or ""),
            str(raw.get("eventCategory") or ""),
            str(raw.get("subCategory") or ""),
            str(raw.get("eventTicker") or raw.get("event_ticker") or ""),
            str(raw.get("series_ticker") or ""),
            str(raw.get("slug") or ""),
            " ".join(str(x) for x in raw.get("tags", []) if isinstance(x, str)),
        ]
    )
    normalized_text = _normalize_text(text_blob)
    tokens = _question_token_set(normalized_text)
    entities = _extract_entities(normalized_text, tokens)
    event_type = _infer_event_type(normalized_text, tokens, entities)
    direction = _infer_direction(normalized_text, tokens, event_type)
    threshold_tokens, bps_tokens, price_tokens = _extract_threshold_tokens(normalized_text, event_type)
    time_tokens, month_tokens, start_dt, end_dt = _extract_time_features(signal, normalized_text)
    family_key = _extract_family_key(signal, normalized_text)
    canonical = _build_canonical_signature(
        event_type=event_type,
        entities=entities,
        direction=direction,
        threshold_tokens=threshold_tokens,
        month_tokens=month_tokens,
        family_key=family_key,
    )
    vector = _hashed_embedding(canonical)
    market_key = f"{signal.source}:{signal.market_id}:{idx}"
    return _EventSignature(
        signal=signal,
        market_key=market_key,
        normalized_text=normalized_text,
        tokens=tokens,
        entities=entities,
        event_type=event_type,
        direction=direction,
        threshold_tokens=threshold_tokens,
        bps_tokens=bps_tokens,
        price_tokens=price_tokens,
        time_tokens=time_tokens,
        month_tokens=month_tokens,
        start_date=start_dt,
        end_date=end_dt,
        family_key=family_key,
        canonical=canonical,
        vector=vector,
    )


def _build_candidate_index(signatures: Sequence[_EventSignature]) -> _CandidateIndex:
    idx = _CandidateIndex()
    for sig in signatures:
        idx.all_keys.add(sig.market_key)
        for entity in sig.entities:
            idx.by_entity.setdefault(entity, set()).add(sig.market_key)
        if sig.event_type:
            idx.by_event_type.setdefault(sig.event_type, set()).add(sig.market_key)
        for token in sig.threshold_tokens:
            idx.by_threshold.setdefault(token, set()).add(sig.market_key)
        for token in sig.month_tokens:
            idx.by_month.setdefault(token, set()).add(sig.market_key)
        if sig.family_key:
            idx.by_family.setdefault(sig.family_key, set()).add(sig.market_key)
    return idx


def _retrieve_top_candidates(
    pm_sig: _EventSignature,
    ks_by_key: Dict[str, _EventSignature],
    index: _CandidateIndex,
    top_k: int,
    min_similarity: float,
) -> list[tuple[str, float]]:
    candidate_keys: set[str] = set()
    for entity in pm_sig.entities:
        candidate_keys.update(index.by_entity.get(entity, set()))
    for token in pm_sig.threshold_tokens:
        candidate_keys.update(index.by_threshold.get(token, set()))
    for token in pm_sig.month_tokens:
        candidate_keys.update(index.by_month.get(token, set()))
    if pm_sig.family_key:
        candidate_keys.update(index.by_family.get(pm_sig.family_key, set()))
    if pm_sig.event_type:
        candidate_keys.update(index.by_event_type.get(pm_sig.event_type, set()))
    if not candidate_keys:
        candidate_keys = set(index.all_keys)

    rows: list[tuple[str, float]] = []
    min_floor = max(0.05, min_similarity * 0.55)
    for ks_key in candidate_keys:
        ks_sig = ks_by_key.get(ks_key)
        if ks_sig is None:
            continue
        if not _hard_compatibility(pm_sig, ks_sig):
            continue
        lexical = _similarity_score(pm_sig.tokens, ks_sig.tokens)
        semantic = _cosine_similarity(pm_sig.vector, ks_sig.vector)
        feature = _feature_alignment_score(pm_sig, ks_sig)
        retrieval = _clamp01(0.35 * lexical + 0.45 * semantic + 0.20 * feature)
        if retrieval < min_floor:
            continue
        rows.append((ks_key, retrieval))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def _hard_compatibility(pm_sig: _EventSignature, ks_sig: _EventSignature) -> bool:
    if pm_sig.event_type != ks_sig.event_type:
        if pm_sig.event_type != "generic" and ks_sig.event_type != "generic":
            return False
        shared_entities = pm_sig.entities.intersection(ks_sig.entities)
        same_family = bool(pm_sig.family_key and ks_sig.family_key and pm_sig.family_key == ks_sig.family_key)
        if not shared_entities and not same_family:
            return False
    elif pm_sig.event_type == "generic":
        shared_entities = pm_sig.entities.intersection(ks_sig.entities)
        same_family = bool(pm_sig.family_key and ks_sig.family_key and pm_sig.family_key == ks_sig.family_key)
        if not shared_entities and not same_family:
            return False

    if pm_sig.direction and ks_sig.direction and pm_sig.direction != ks_sig.direction:
        return False

    if pm_sig.bps_tokens and ks_sig.bps_tokens and pm_sig.bps_tokens.isdisjoint(ks_sig.bps_tokens):
        return False
    if pm_sig.price_tokens and ks_sig.price_tokens and pm_sig.price_tokens.isdisjoint(ks_sig.price_tokens):
        return False
    if pm_sig.entities and ks_sig.entities and pm_sig.entities.isdisjoint(ks_sig.entities):
        return False
    if not _time_windows_compatible(pm_sig, ks_sig):
        return False
    return True


def _time_windows_compatible(pm_sig: _EventSignature, ks_sig: _EventSignature) -> bool:
    if pm_sig.end_date and ks_sig.end_date:
        if abs((pm_sig.end_date - ks_sig.end_date).days) > 2:
            return False
    if pm_sig.start_date and ks_sig.start_date:
        # Accept slight publication offsets but reject materially different start windows.
        if abs((pm_sig.start_date - ks_sig.start_date).days) > 35:
            return False

    if pm_sig.month_tokens and ks_sig.month_tokens and pm_sig.month_tokens.isdisjoint(ks_sig.month_tokens):
        return False
    return True


def _heuristic_verify(
    pm_sig: _EventSignature,
    ks_sig: _EventSignature,
    retrieval_score: float,
) -> _VerificationResult:
    same_event = _hard_compatibility(pm_sig, ks_sig)
    if not same_event:
        return _VerificationResult(
            same_underlying_event=False,
            same_resolution_criteria=False,
            match_score=0.0,
            match_type="not_match",
            differences="different_event_or_constraints",
        )

    same_resolution = _same_resolution_criteria(pm_sig, ks_sig)
    feature = _feature_alignment_score(pm_sig, ks_sig)
    base_score = _clamp01(0.55 * retrieval_score + 0.45 * feature)
    if same_resolution:
        match_type = "exact"
        match_score = max(base_score, 0.60)
    else:
        match_type = "related"
        match_score = min(0.89, base_score)
    differences = _describe_differences(pm_sig, ks_sig)

    return _VerificationResult(
        same_underlying_event=True,
        same_resolution_criteria=same_resolution,
        match_score=match_score,
        match_type=match_type,
        differences=differences,
    )


def _same_resolution_criteria(pm_sig: _EventSignature, ks_sig: _EventSignature) -> bool:
    if pm_sig.direction and ks_sig.direction and pm_sig.direction != ks_sig.direction:
        return False
    if pm_sig.bps_tokens or ks_sig.bps_tokens:
        if pm_sig.bps_tokens != ks_sig.bps_tokens:
            return False
    if pm_sig.price_tokens or ks_sig.price_tokens:
        if pm_sig.price_tokens != ks_sig.price_tokens:
            return False

    if pm_sig.end_date and ks_sig.end_date:
        if abs((pm_sig.end_date - ks_sig.end_date).days) > 1:
            return False
    elif pm_sig.month_tokens and ks_sig.month_tokens:
        if pm_sig.month_tokens.isdisjoint(ks_sig.month_tokens):
            return False

    if pm_sig.start_date and ks_sig.start_date:
        if abs((pm_sig.start_date - ks_sig.start_date).days) > 14:
            return False
    return True


def _describe_differences(pm_sig: _EventSignature, ks_sig: _EventSignature) -> str:
    diffs: list[str] = []
    if pm_sig.direction and ks_sig.direction and pm_sig.direction != ks_sig.direction:
        diffs.append("direction")
    if pm_sig.bps_tokens != ks_sig.bps_tokens:
        if pm_sig.bps_tokens or ks_sig.bps_tokens:
            diffs.append("bps")
    if pm_sig.price_tokens != ks_sig.price_tokens:
        if pm_sig.price_tokens or ks_sig.price_tokens:
            diffs.append("price_threshold")
    if pm_sig.end_date and ks_sig.end_date and abs((pm_sig.end_date - ks_sig.end_date).days) > 1:
        diffs.append("end_date")
    if pm_sig.start_date and ks_sig.start_date and abs((pm_sig.start_date - ks_sig.start_date).days) > 14:
        diffs.append("start_date")
    if pm_sig.month_tokens and ks_sig.month_tokens and pm_sig.month_tokens.isdisjoint(ks_sig.month_tokens):
        diffs.append("time_window")
    if not diffs:
        return ""
    return ",".join(diffs)


def _should_call_llm(heuristic: _VerificationResult, retrieval_score: float) -> bool:
    if retrieval_score < 0.20:
        return False
    # LLM is mainly useful for ambiguous related vs exact.
    return not heuristic.same_resolution_criteria or heuristic.match_score < 0.85


def _merge_verification(heuristic: _VerificationResult, llm_result: _VerificationResult) -> _VerificationResult:
    same_event = heuristic.same_underlying_event and llm_result.same_underlying_event
    same_resolution = heuristic.same_resolution_criteria and llm_result.same_resolution_criteria
    if llm_result.same_underlying_event and llm_result.same_resolution_criteria:
        same_event = True
        same_resolution = True

    match_score = _clamp01(0.45 * heuristic.match_score + 0.55 * llm_result.match_score)
    match_type = llm_result.match_type or heuristic.match_type
    differences = llm_result.differences or heuristic.differences
    return _VerificationResult(
        same_underlying_event=same_event,
        same_resolution_criteria=same_resolution,
        match_score=match_score,
        match_type=match_type,
        differences=differences,
    )


class _LLMCandidateVerifier:
    def __init__(self, enabled: bool, api_key: str, model: str, timeout_seconds: int):
        self._enabled = bool(enabled and (api_key or "").strip())
        self.model = model
        self._client = None
        if not self._enabled:
            return
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key.strip(), timeout=timeout_seconds)
        except Exception:
            logger.warning("LLM verifier disabled: OpenAI client unavailable")
            self._enabled = False

    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    def verify_batch(
        self,
        pairs: Sequence[tuple[str, _EventSignature, _EventSignature]],
    ) -> dict[str, _VerificationResult]:
        if not self.enabled() or not pairs:
            return {}

        system = (
            "You judge whether each prediction-market pair represents the same underlying event and the same "
            "resolution criteria. Return strict JSON only: "
            "{\"results\":[{\"pair_id\":\"...\",\"same_underlying_event\":true|false,"
            "\"same_resolution_criteria\":true|false,\"match_type\":\"exact|subset|superset|related|not_match\","
            "\"match_score\":0..1,\"differences\":\"...\"}]}. "
            "Preserve pair_id exactly."
        )
        user_payload = {
            "pairs": [
                {
                    "pair_id": pair_id,
                    "polymarket": _pair_side_payload(pm_sig),
                    "kalshi": _pair_side_payload(ks_sig),
                }
                for pair_id, pm_sig, ks_sig in pairs
            ]
        }

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user_payload, separators=(",", ":"))},
                ],
            )
            payload = _extract_json_object((response.output_text or "").strip())
            if not isinstance(payload, dict):
                return {}
            rows = payload.get("results")
            if not isinstance(rows, list):
                return {}

            out: dict[str, _VerificationResult] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                pair_id = str(row.get("pair_id") or "").strip()
                if not pair_id:
                    continue
                out[pair_id] = _VerificationResult(
                    same_underlying_event=bool(row.get("same_underlying_event")),
                    same_resolution_criteria=bool(row.get("same_resolution_criteria")),
                    match_score=_clamp01(_to_float(row.get("match_score"))),
                    match_type=str(row.get("match_type") or "").strip() or "related",
                    differences=str(row.get("differences") or "").strip(),
                )
            return out
        except Exception as exc:
            logger.debug("LLM batch verification failed | pairs=%s error=%s", len(pairs), str(exc))
            return {}


def _pair_side_payload(sig: _EventSignature) -> dict[str, object]:
    return {
        "question": sig.signal.question,
        "event_type": sig.event_type,
        "entities": sorted(sig.entities),
        "direction": sig.direction,
        "thresholds": sorted(sig.threshold_tokens),
        "start_date": sig.start_date.isoformat() if sig.start_date else "",
        "end_date": sig.end_date.isoformat() if sig.end_date else "",
    }


def _extract_json_object(text: str) -> dict:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _feature_alignment_score(a: _EventSignature, b: _EventSignature) -> float:
    score = 0.0
    if a.event_type and b.event_type and a.event_type == b.event_type:
        score += 0.25
    if a.entities and b.entities:
        overlap = len(a.entities.intersection(b.entities)) / max(1, min(len(a.entities), len(b.entities)))
        score += 0.35 * _clamp01(overlap)
    if a.direction and b.direction and a.direction == b.direction:
        score += 0.10
    if a.threshold_tokens and b.threshold_tokens:
        if not a.threshold_tokens.isdisjoint(b.threshold_tokens):
            score += 0.20
    elif not a.threshold_tokens and not b.threshold_tokens:
        score += 0.08
    if a.month_tokens and b.month_tokens and not a.month_tokens.isdisjoint(b.month_tokens):
        score += 0.10
    return _clamp01(score)


def _build_canonical_signature(
    event_type: str,
    entities: set[str],
    direction: str,
    threshold_tokens: set[str],
    month_tokens: set[str],
    family_key: str,
) -> str:
    return " | ".join(
        [
            f"type={event_type or 'generic'}",
            f"entities={','.join(sorted(entities))}",
            f"direction={direction or 'none'}",
            f"thresholds={','.join(sorted(threshold_tokens))}",
            f"time={','.join(sorted(month_tokens))}",
            f"family={family_key}",
        ]
    )


def _extract_time_features(signal: PredictionSignal, normalized_text: str) -> tuple[set[str], set[str], date | None, date | None]:
    raw = signal.raw or {}
    start_dt = _parse_date_any(
        raw.get("startDate")
        or raw.get("start_date")
        or raw.get("open_ts")
        or raw.get("acceptingOrdersTimestamp")
    )
    end_dt = _parse_date_any(
        raw.get("endDate")
        or raw.get("end_date")
        or raw.get("close_time")
        or raw.get("close_ts")
        or raw.get("expected_expiration_ts")
        or raw.get("umaEndDate")
    )

    month_tokens: set[str] = set()
    time_tokens: set[str] = set()

    if start_dt:
        month_tokens.add(f"month:{start_dt.year:04d}-{start_dt.month:02d}")
        time_tokens.add(f"start:{start_dt.isoformat()}")
    if end_dt:
        month_tokens.add(f"month:{end_dt.year:04d}-{end_dt.month:02d}")
        time_tokens.add(f"end:{end_dt.isoformat()}")

    for found in _extract_textual_date_mentions(normalized_text):
        dt = found["date"]
        relation = found["relation"]
        if dt:
            month_tokens.add(f"month:{dt.year:04d}-{dt.month:02d}")
            time_tokens.add(f"date:{dt.isoformat()}")
            if relation:
                time_tokens.add(f"{relation}:{dt.isoformat()}")
            if relation == "before" and dt.day == 1:
                prev = dt - timedelta(days=1)
                month_tokens.add(f"month:{prev.year:04d}-{prev.month:02d}")

    return time_tokens, month_tokens, start_dt, end_dt


def _extract_textual_date_mentions(text: str) -> list[dict]:
    out: list[dict] = []
    pattern = (
        r"\b(?:(before|by|on|after|in)\s+)?"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"(?:\s+(\d{1,2}))?"
        r"(?:,?\s+(\d{4}))?\b"
    )
    for match in re.finditer(pattern, text):
        relation = str(match.group(1) or "").strip().lower()
        month_name = str(match.group(2) or "").strip().lower()
        day_raw = str(match.group(3) or "").strip()
        year_raw = str(match.group(4) or "").strip()
        month_num = _MONTHS.get(month_name[:3])
        if month_num is None:
            continue
        day = int(day_raw) if day_raw.isdigit() else 1
        year = int(year_raw) if year_raw.isdigit() else 0
        if year < 1900 or year > 2100:
            continue
        try:
            dt = date(year, month_num, day)
        except ValueError:
            continue
        out.append({"date": dt, "relation": relation})
    return out


def _extract_threshold_tokens(text: str, event_type: str) -> tuple[set[str], set[str], set[str]]:
    threshold_tokens: set[str] = set()
    bps_tokens = _extract_bps_tokens(text)
    price_tokens = _extract_price_tokens(text, event_type=event_type)
    threshold_tokens.update(bps_tokens)
    threshold_tokens.update(price_tokens)

    if event_type == "fed_rate_decision" and not bps_tokens and _contains_any_token(text, _RATE_HOLD_TERMS):
        token = "bps:eq:0"
        bps_tokens.add(token)
        threshold_tokens.add(token)

    return threshold_tokens, bps_tokens, price_tokens


def _extract_bps_tokens(text: str) -> set[str]:
    out: set[str] = set()
    pattern = (
        r"(?:(at\s+least|more\s+than|less\s+than|above|over|under|below|>=|<=|>|<)\s+)?"
        r"(\d{1,3})"
        r"\s*(\+)?\s*(?:bp|bps|basis\s+point(?:s)?)"
    )
    for match in re.finditer(pattern, text):
        prefix = str(match.group(1) or "").strip().lower()
        value_raw = str(match.group(2) or "").strip()
        plus = str(match.group(3) or "").strip()
        if not value_raw.isdigit():
            continue
        value = int(value_raw)
        qualifier = "eq"
        if plus or prefix in {"at least", "more than", "above", "over", ">=", ">"}:
            qualifier = "ge"
        elif prefix in {"less than", "under", "below", "<=", "<"}:
            qualifier = "le"
        out.add(f"bps:{qualifier}:{value}")
    return out


def _extract_price_tokens(text: str, event_type: str) -> set[str]:
    out: set[str] = set()
    likely_price_market = event_type == "crypto_price_target" or _contains_any_token(
        text,
        {"bitcoin", "btc", "ethereum", "eth", "nasdaq", "sp500", "s&p", "gold", "oil", "wti", "brent", "index"},
    )
    pattern = r"(\$?\s*\d[\d,]*(?:\.\d+)?)\s*([kmb])?"
    for match in re.finditer(pattern, text):
        number_text = str(match.group(1) or "")
        suffix = str(match.group(2) or "").lower().strip()
        if not likely_price_market and "$" not in number_text and suffix not in {"k", "m", "b"}:
            continue
        span_start, span_end = match.span()
        context = text[max(0, span_start - 24): min(len(text), span_end + 24)]
        if not _contains_any_token(context, _PRICE_CONTEXT_TERMS):
            continue
        value = _parse_number_token(number_text, suffix=suffix)
        if value is None:
            continue
        if 1900 <= value <= 2100 and "$" not in number_text and not suffix:
            continue
        qualifier = "eq"
        prefix = text[max(0, span_start - 16):span_start]
        if re.search(r"(at least|more than|over|above|>=|>)\s*$", prefix):
            qualifier = "ge"
        elif re.search(r"(less than|under|below|<=|<)\s*$", prefix):
            qualifier = "le"
        out.add(f"usd:{qualifier}:{int(round(value))}")
    return out


def _extract_entities(text: str, tokens: set[str]) -> set[str]:
    padded = f" {text} "
    entities: set[str] = set()
    for canonical, aliases in _ENTITY_ALIASES.items():
        for alias in aliases:
            if f" {alias} " in padded:
                entities.add(canonical)
                break

    for tok in tokens:
        if tok in _DOMAIN_ENTITY_TOKENS:
            entities.add(tok)
    return entities


def _infer_event_type(text: str, tokens: set[str], entities: set[str]) -> str:
    if ("fed" in entities or "fomc" in entities or "fed" in tokens) and (
        "rate" in tokens or _contains_any_token(text, _RATE_UP_TERMS.union(_RATE_DOWN_TERMS).union(_RATE_HOLD_TERMS))
    ):
        return "fed_rate_decision"
    if entities.intersection({"bitcoin", "btc", "ethereum", "eth"}) and _contains_any_token(text, _PRICE_CONTEXT_TERMS):
        return "crypto_price_target"
    if entities.intersection({"ali_khamenei", "khamenei"}) and _contains_any_token(text, {"out", "removed", "die", "dies"}):
        return "leadership_status"
    if entities.intersection({"iran"}) and entities.intersection({"us"}) and _contains_any_token(text, {"strike", "strikes"}):
        return "military_strike"
    if _contains_any_token(text, {"election", "nominate", "nomination"}):
        return "election_outcome"
    if _contains_any_token(text, {"cpi", "inflation"}):
        return "inflation_release"
    return "generic"


def _infer_direction(text: str, tokens: set[str], event_type: str) -> str:
    has_up = bool(tokens.intersection(_RATE_UP_TERMS))
    has_down = bool(tokens.intersection(_RATE_DOWN_TERMS))
    has_hold = bool(tokens.intersection(_RATE_HOLD_TERMS))

    if event_type == "fed_rate_decision":
        if has_hold and not has_up and not has_down:
            return "hold"
        if has_up and not has_down:
            return "up"
        if has_down and not has_up:
            return "down"

    if event_type == "crypto_price_target":
        if _contains_any_token(text, {"below", "under", "fall", "falls"}):
            return "down"
        if _contains_any_token(text, {"above", "over", "reach", "hit", "exceed"}):
            return "up"

    if event_type == "leadership_status":
        if _contains_any_token(text, {"out", "removed", "die", "dies"}):
            return "out"
    return ""


def _extract_family_key(signal: PredictionSignal, normalized_text: str) -> str:
    raw = signal.raw or {}
    base = (
        str(raw.get("eventTitle") or raw.get("event_title") or "").strip()
        or str(raw.get("series_title") or "").strip()
        or str(raw.get("eventSlug") or raw.get("slug") or "").strip()
        or str(signal.question or "")
    )
    cleaned = _normalize_text(base)
    cleaned = re.sub(
        r"\b(yes|no|will|by|before|after|in|march|april|may|june|july|august|september|october|november|december|"
        r"january|february|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|bps?|bp|basis|point|points|\d+)\b",
        " ",
        cleaned,
    )
    key = _normalize_text(cleaned)
    if not key:
        key = " ".join(sorted(_question_token_set(normalized_text))[:8])
    parts = key.split()
    return " ".join(parts[:10])


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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _contains_any_token(text: str, terms: set[str]) -> bool:
    if not text:
        return False
    padded = f" {_normalize_text(text)} "
    for term in terms:
        norm = _normalize_text(term)
        if not norm:
            continue
        if f" {norm} " in padded:
            return True
    return False


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


def _hashed_embedding(text: str, dims: int = 256) -> list[float]:
    vec = [0.0] * dims
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        idx = h % dims
        sign = -1.0 if ((h >> 11) & 1) else 1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        return [v / norm for v in vec]
    return vec


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return _clamp01(dot / (math.sqrt(norm_a) * math.sqrt(norm_b)))


def _parse_date_any(value: Any) -> date | None:
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
    # Fall back to YYYY-MM-DD prefix.
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", cleaned)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    return None


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_number_token(raw: str, suffix: str = "") -> float | None:
    cleaned = (raw or "").replace("$", "").replace(",", "").strip().lower()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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

_ENTITY_ALIASES = {
    "bitcoin": {"bitcoin", "btc"},
    "ethereum": {"ethereum", "eth"},
    "fed": {"fed", "fomc", "federal reserve"},
    "us": {"us", "united states", "u s"},
    "iran": {"iran", "iranian"},
    "trump": {"trump", "donald trump"},
    "ali_khamenei": {"ali khamenei", "khamenei"},
    "cpi": {"cpi", "inflation"},
}

_DOMAIN_ENTITY_TOKENS = {
    "bitcoin",
    "ethereum",
    "fed",
    "fomc",
    "cpi",
    "iran",
    "trump",
    "khamenei",
    "tesla",
    "openai",
    "nvidia",
}
