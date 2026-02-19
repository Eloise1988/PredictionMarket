from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from openai import OpenAI

from prediction_agent.models import PredictionSignal

logger = logging.getLogger(__name__)


@dataclass
class MarketStockCandidate:
    ticker: str
    direction_if_yes: str
    linkage_score: float
    rationale: str


@dataclass
class CrossVenueStrongMatch:
    polymarket_id: str
    kalshi_id: str
    strength: float
    rationale: str


class LLMMarketMapper:
    def __init__(self, api_key: str, model: str, timeout_seconds: int = 20):
        self.api_key = api_key.strip()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = OpenAI(api_key=self.api_key, timeout=timeout_seconds) if self.api_key else None
        self.last_cross_venue_llm_raw: str = ""
        self.last_cross_venue_llm_error: str = ""

    def enabled(self) -> bool:
        return self._client is not None

    def map_market(self, signal: PredictionSignal, max_tickers: int = 8, min_linkage_score: float = 0.35) -> List[MarketStockCandidate]:
        if not self.enabled():
            return []

        system = (
            "You are an event-driven equity analyst. "
            "Given a prediction-market question, return the U.S.-listed stocks or ETFs most directly impacted "
            "IF the event resolves YES. "
            "Do not use meme picks; prefer liquid, investable instruments. "
            "Return strict JSON only with schema: "
            "{\"candidates\":[{\"ticker\":\"AAPL\",\"direction_if_yes\":\"long|short\",\"linkage_score\":0.0-1.0,\"rationale\":\"...\"}]}"
        )
        user = (
            f"question={signal.question}\n"
            f"source={signal.source}\n"
            f"prob_yes={signal.prob_yes:.4f}\n"
            f"liquidity={signal.liquidity:.2f}\n"
            f"volume_24h={signal.volume_24h:.2f}\n"
            f"max_candidates={max_tickers}"
        )

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw_text = (response.output_text or "").strip()
            payload = _extract_json(raw_text)
            return _parse_candidates(payload, max_tickers=max_tickers, min_linkage_score=min_linkage_score)
        except Exception as exc:
            logger.warning("LLM market mapper failed: %s | q=%s", str(exc), signal.question[:180])
            return []

    def select_diverse_markets(
        self,
        signals: List[PredictionSignal],
        target_count: int,
    ) -> List[str]:
        if not self.enabled() or not signals or target_count <= 0:
            return []

        market_rows = []
        for rank, s in enumerate(signals, start=1):
            raw = s.raw or {}
            market_rows.append(
                {
                    "rank_by_liquidity": rank,
                    "market_id": s.market_id,
                    "question": s.question,
                    "liquidity": round(float(s.liquidity), 2),
                    "prob_yes": round(float(s.prob_yes), 4),
                    "event_title": str(raw.get("eventTitle") or ""),
                    "event_slug": str(raw.get("eventSlug") or raw.get("slug") or ""),
                    "category": str(raw.get("category") or ""),
                    "event_category": str(raw.get("eventCategory") or ""),
                    "sub_category": str(raw.get("subCategory") or ""),
                    "tags": [str(x) for x in raw.get("tags", []) if isinstance(x, str)][:8],
                }
            )

        system = (
            "You are selecting a diverse subset of prediction markets for downstream stock-impact analysis. "
            "Pick markets that are materially different events, avoid near-duplicate variants of the same parent event "
            "(for example multiple candidate outcomes for one nomination race), and prefer higher-liquidity contracts. "
            "Hard constraints: select only finance/macro/economy/markets-related contracts; avoid sports/entertainment contracts. "
            "Select at most one contract per parent event/event_slug/event_title family. "
            "Return strict JSON only with schema: {\"market_ids\":[\"id1\",\"id2\"]}. "
            "Use only market_id values provided by user."
        )
        user = json.dumps(
            {
                "target_count": int(target_count),
                "markets": market_rows,
            },
            separators=(",", ":"),
        )

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            payload = _extract_json((response.output_text or "").strip())
            ids = payload.get("market_ids", []) if isinstance(payload, dict) else []
            if not isinstance(ids, list):
                return []
            allowed = {s.market_id for s in signals}
            out: List[str] = []
            seen: set[str] = set()
            for raw_id in ids:
                market_id = str(raw_id).strip()
                if not market_id or market_id not in allowed or market_id in seen:
                    continue
                seen.add(market_id)
                out.append(market_id)
                if len(out) >= target_count:
                    break
            return out
        except Exception as exc:
            logger.warning("LLM market selector failed: %s", str(exc))
            return []

    def select_strong_cross_venue_matches(
        self,
        candidate_pairs: List[Dict[str, object]],
        min_strength: float = 0.85,
        max_matches: int = 100,
    ) -> List[CrossVenueStrongMatch]:
        if not self.enabled() or not candidate_pairs or max_matches <= 0:
            return []

        normalized: List[Dict[str, object]] = []
        allowed_pairs: set[Tuple[str, str]] = set()
        for row in candidate_pairs:
            if not isinstance(row, dict):
                continue
            polymarket_id = str(row.get("polymarket_id") or "").strip()
            kalshi_id = str(row.get("kalshi_id") or "").strip()
            polymarket_question = str(row.get("polymarket_question") or "").strip()
            kalshi_question = str(row.get("kalshi_question") or "").strip()
            if not polymarket_id or not kalshi_id:
                continue
            if not polymarket_question or not kalshi_question:
                continue

            normalized.append(
                {
                    "rank": int(_to_float(row.get("rank"))),
                    "polymarket_id": polymarket_id,
                    "kalshi_id": kalshi_id,
                    "polymarket_question": polymarket_question,
                    "kalshi_question": kalshi_question,
                    "heuristic_similarity": round(_clamp(_to_float(row.get("heuristic_similarity")), 0.0, 1.0), 4),
                    "probability_diff_pp": round(_to_float(row.get("probability_diff_pp")), 3),
                }
            )
            allowed_pairs.add((polymarket_id, kalshi_id))

        if not normalized:
            return []

        system = (
            "You are matching prediction-market contracts across exchanges. "
            "Keep ONLY direct and strong equivalents where the underlying event and resolution condition are materially the same. "
            "Reject broad topical similarity, different outcomes, different strike levels, or different entities. "
            "Treat deadline boundary equivalents as the same event when appropriate, e.g., "
            "'before March 1, 2026' and 'by end of February 2026'. "
            "Use only provided candidate rows; do not invent ids. "
            "Return strict JSON only with schema: "
            "{\"matches\":[{\"polymarket_id\":\"...\",\"kalshi_id\":\"...\",\"strength\":0.0-1.0,\"rationale\":\"...\"}]}"
        )
        user = json.dumps(
            {
                "min_strength": round(_clamp(float(min_strength), 0.0, 1.0), 3),
                "candidates": normalized,
            },
            separators=(",", ":"),
        )

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            payload = _extract_json((response.output_text or "").strip())
            return _parse_cross_venue_matches(
                payload,
                allowed_pairs=allowed_pairs,
                min_strength=min_strength,
                max_matches=max_matches,
            )
        except Exception as exc:
            logger.warning("LLM cross-venue strong matcher failed: %s", str(exc))
            return []

    def select_strong_cross_venue_matches_from_lists(
        self,
        polymarket_markets: List[Dict[str, object]],
        kalshi_markets: List[Dict[str, object]],
        min_strength: float = 0.85,
        max_matches: int = 100,
    ) -> List[CrossVenueStrongMatch]:
        self.last_cross_venue_llm_raw = ""
        self.last_cross_venue_llm_error = ""
        if not self.enabled() or not polymarket_markets or not kalshi_markets or max_matches <= 0:
            return []

        pm_rows: List[Dict[str, object]] = []
        ks_rows: List[Dict[str, object]] = []
        allowed_pm_ids: set[str] = set()
        allowed_ka_ids: set[str] = set()

        for row in polymarket_markets:
            if not isinstance(row, dict):
                continue
            market_id = str(row.get("market_id") or "").strip()
            question = str(row.get("question") or "").strip()
            if not market_id or not question:
                continue
            allowed_pm_ids.add(market_id)
            pm_rows.append(
                {
                    "rank": int(_to_float(row.get("rank"))),
                    "market_id": market_id,
                    "question": question,
                }
            )

        for row in kalshi_markets:
            if not isinstance(row, dict):
                continue
            market_id = str(row.get("market_id") or "").strip()
            question = str(row.get("question") or "").strip()
            if not market_id or not question:
                continue
            allowed_ka_ids.add(market_id)
            ks_rows.append(
                {
                    "rank": int(_to_float(row.get("rank"))),
                    "market_id": market_id,
                    "question": question,
                }
            )

        if not pm_rows or not ks_rows:
            return []

        system = (
            "You are matching prediction-market contracts across exchanges. "
            "Input gives two pre-match lists: Polymarket and Kalshi. "
            "Return ALL direct / strong equivalents where both contracts resolve on materially the same event condition. "
            "Reject broad topical similarity, different outcomes, different strike levels, and different entities. "
            "Treat deadline boundary equivalents as same when appropriate, e.g., "
            "'before March 1, 2026' can match 'by end of February 2026'. "
            "Use only provided IDs. Prefer one-to-one mappings. "
            "Return strict JSON only with schema: "
            "{\"matches\":[{\"polymarket_id\":\"...\",\"kalshi_id\":\"...\",\"strength\":0.0-1.0,\"rationale\":\"...\"}]}"
        )
        user = _build_cross_venue_text_list_prompt(
            min_strength=min_strength,
            polymarket_rows=pm_rows,
            kalshi_rows=ks_rows,
        )

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw_text = (response.output_text or "").strip()
            self.last_cross_venue_llm_raw = raw_text
            payload = _extract_json(raw_text)
            return _parse_cross_venue_matches_from_lists(
                payload,
                allowed_polymarket_ids=allowed_pm_ids,
                allowed_kalshi_ids=allowed_ka_ids,
                min_strength=min_strength,
                max_matches=max_matches,
            )
        except Exception as exc:
            self.last_cross_venue_llm_error = str(exc)
            logger.warning("LLM cross-venue strong matcher (list mode) failed: %s", str(exc))
            return []


def _extract_json(text: str) -> dict:
    if not text:
        return {}

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}

    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def _parse_candidates(payload: dict, max_tickers: int, min_linkage_score: float) -> List[MarketStockCandidate]:
    rows = payload.get("candidates", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []

    out: List[MarketStockCandidate] = []
    seen: set[str] = set()

    for row in rows:
        if not isinstance(row, dict):
            continue

        ticker = str(row.get("ticker") or "").upper().strip()
        if not re.match(r"^[A-Z]{1,6}$", ticker):
            continue

        direction = str(row.get("direction_if_yes") or "").lower().strip()
        if direction not in {"long", "short"}:
            continue

        linkage_score = _clamp(_to_float(row.get("linkage_score")), 0.0, 1.0)
        if linkage_score < min_linkage_score:
            continue

        rationale = str(row.get("rationale") or "").strip()
        if ticker in seen:
            continue
        seen.add(ticker)

        out.append(
            MarketStockCandidate(
                ticker=ticker,
                direction_if_yes=direction,
                linkage_score=linkage_score,
                rationale=rationale,
            )
        )

        if len(out) >= max_tickers:
            break

    return out


def _parse_cross_venue_matches(
    payload: dict,
    allowed_pairs: set[Tuple[str, str]],
    min_strength: float,
    max_matches: int,
) -> List[CrossVenueStrongMatch]:
    rows = payload.get("matches", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []

    out: List[CrossVenueStrongMatch] = []
    seen_pairs: set[Tuple[str, str]] = set()
    used_pm: set[str] = set()
    used_ka: set[str] = set()
    min_s = _clamp(float(min_strength), 0.0, 1.0)

    for row in rows:
        if not isinstance(row, dict):
            continue

        polymarket_id = str(row.get("polymarket_id") or "").strip()
        kalshi_id = str(row.get("kalshi_id") or "").strip()
        pair = (polymarket_id, kalshi_id)
        if not polymarket_id or not kalshi_id:
            continue
        if pair not in allowed_pairs:
            continue
        if pair in seen_pairs:
            continue
        if polymarket_id in used_pm or kalshi_id in used_ka:
            continue

        strength = _clamp(_to_float(row.get("strength")), 0.0, 1.0)
        if strength < min_s:
            continue

        rationale = str(row.get("rationale") or "").strip()
        seen_pairs.add(pair)
        used_pm.add(polymarket_id)
        used_ka.add(kalshi_id)
        out.append(
            CrossVenueStrongMatch(
                polymarket_id=polymarket_id,
                kalshi_id=kalshi_id,
                strength=strength,
                rationale=rationale,
            )
        )
        if len(out) >= max_matches:
            break

    return out


def _parse_cross_venue_matches_from_lists(
    payload: dict,
    allowed_polymarket_ids: set[str],
    allowed_kalshi_ids: set[str],
    min_strength: float,
    max_matches: int,
) -> List[CrossVenueStrongMatch]:
    rows = payload.get("matches", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []

    out: List[CrossVenueStrongMatch] = []
    seen_pairs: set[Tuple[str, str]] = set()
    used_pm: set[str] = set()
    used_ka: set[str] = set()
    min_s = _clamp(float(min_strength), 0.0, 1.0)

    for row in rows:
        if not isinstance(row, dict):
            continue

        polymarket_id = str(row.get("polymarket_id") or "").strip()
        kalshi_id = str(row.get("kalshi_id") or "").strip()
        pair = (polymarket_id, kalshi_id)
        if not polymarket_id or not kalshi_id:
            continue
        if polymarket_id not in allowed_polymarket_ids or kalshi_id not in allowed_kalshi_ids:
            continue
        if pair in seen_pairs:
            continue
        if polymarket_id in used_pm or kalshi_id in used_ka:
            continue

        strength = _clamp(_to_float(row.get("strength")), 0.0, 1.0)
        if strength < min_s:
            continue

        rationale = str(row.get("rationale") or "").strip()
        seen_pairs.add(pair)
        used_pm.add(polymarket_id)
        used_ka.add(kalshi_id)
        out.append(
            CrossVenueStrongMatch(
                polymarket_id=polymarket_id,
                kalshi_id=kalshi_id,
                strength=strength,
                rationale=rationale,
            )
        )
        if len(out) >= max_matches:
            break

    return out


def _build_cross_venue_text_list_prompt(
    min_strength: float,
    polymarket_rows: List[Dict[str, object]],
    kalshi_rows: List[Dict[str, object]],
) -> str:
    min_s = round(_clamp(float(min_strength), 0.0, 1.0), 3)
    pm_lines = "\n".join(_format_market_line(row) for row in polymarket_rows)
    ks_lines = "\n".join(_format_market_line(row) for row in kalshi_rows)
    return (
        f"Minimum strength: {min_s}\n\n"
        "Polymarket list (format: [id] question):\n"
        f"{pm_lines}\n\n"
        "Kalshi list (format: [id] question):\n"
        f"{ks_lines}\n\n"
        "Return JSON only."
    )


def _format_market_line(row: Dict[str, object]) -> str:
    market_id = str(row.get("market_id") or "").strip()
    question = str(row.get("question") or "").strip()
    if len(question) > _LLM_STRONG_MAX_QUESTION_CHARS:
        question = f"{question[:_LLM_STRONG_MAX_QUESTION_CHARS].rstrip()}..."
    return f"[{market_id}] {question}"


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


_LLM_STRONG_MAX_QUESTION_CHARS = 220
