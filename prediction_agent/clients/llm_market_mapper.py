from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List

from openai import OpenAI

from prediction_agent.models import PredictionSignal

logger = logging.getLogger(__name__)


@dataclass
class MarketStockCandidate:
    ticker: str
    direction_if_yes: str
    linkage_score: float
    rationale: str


class LLMMarketMapper:
    def __init__(self, api_key: str, model: str, timeout_seconds: int = 20):
        self.api_key = api_key.strip()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = OpenAI(api_key=self.api_key, timeout=timeout_seconds) if self.api_key else None

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
        for s in signals:
            raw = s.raw or {}
            market_rows.append(
                {
                    "market_id": s.market_id,
                    "question": s.question,
                    "liquidity": round(float(s.liquidity), 2),
                    "prob_yes": round(float(s.prob_yes), 4),
                    "event_title": str(raw.get("eventTitle") or ""),
                    "event_slug": str(raw.get("eventSlug") or raw.get("slug") or ""),
                }
            )

        system = (
            "You are selecting a diverse subset of prediction markets for downstream stock-impact analysis. "
            "Pick markets that are materially different events, avoid near-duplicate variants of the same parent event "
            "(for example multiple candidate outcomes for one nomination race), and prefer higher-liquidity contracts. "
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


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
