from __future__ import annotations

import json
import logging
from typing import List

from openai import OpenAI

from prediction_agent.models import CandidateIdea

logger = logging.getLogger(__name__)


class LLMBriefer:
    def __init__(self, api_key: str, model: str, timeout_seconds: int = 20):
        self.api_key = api_key.strip()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = OpenAI(api_key=self.api_key, timeout=timeout_seconds) if self.api_key else None

    def enabled(self) -> bool:
        return self._client is not None

    def summarize(self, ideas: List[CandidateIdea]) -> str:
        if not ideas:
            return ""
        if not self.enabled():
            return _fallback_summary(ideas)

        compact = [
            {
                "ticker": i.ticker,
                "direction": i.direction,
                "score": round(i.score, 3),
                "theme": i.event_theme,
                "event_probability": round(i.event_probability, 3),
                "question": i.market_question,
                "source": i.market_source,
                "url": i.market_url,
                "company_name": str(i.metadata.get("company_name", "")),
                "sector": str(i.metadata.get("sector", "")),
                "background": str(i.metadata.get("company_background", "")),
            }
            for i in ideas
        ]

        system = (
            "You are a risk-aware macro + equities analyst. "
            "Return a concise Telegram-ready summary with: "
            "(1) top ideas, (2) event rationale, (3) risk controls. "
            "In the top-ideas section, include one short background clause "
            "for each ticker/ETF explaining what it is and why exposure fits the event. "
            "Never guarantee returns and never claim certainty."
        )
        user = f"Ideas JSON:\n{json.dumps(compact, ensure_ascii=True)}"

        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = (response.output_text or "").strip()
            return text if text else _fallback_summary(ideas)
        except Exception as exc:
            logger.warning("OpenAI summary failed", extra={"error": str(exc)})
            return _fallback_summary(ideas)


def _fallback_summary(ideas: List[CandidateIdea]) -> str:
    lines = ["Prediction-market-driven equity ideas (decision support, not financial advice):"]
    for i in ideas:
        background = str(i.metadata.get("company_background", "")).strip()
        if background:
            lines.append(f"- Background: {background}")
        lines.append(
            f"- {i.direction.upper()} {i.ticker} | score={i.score:.2f} | theme={i.event_theme} | "
            f"P(event)={i.event_probability:.1%} | {i.market_source}: {i.market_question}"
        )
    lines.append("Use position sizing, stop-loss rules, and max theme exposure limits.")
    return "\n".join(lines)
