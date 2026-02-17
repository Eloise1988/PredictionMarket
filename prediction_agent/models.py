from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionSignal(BaseModel):
    source: str
    market_id: str
    question: str
    url: str = ""
    prob_yes: float = Field(ge=0.0, le=1.0)
    volume_24h: float = 0.0
    liquidity: float = 0.0
    updated_at: datetime
    raw: Dict[str, Any] = Field(default_factory=dict)


class EquitySnapshot(BaseModel):
    ticker: str
    name: str = ""
    sector: str = ""
    price: Optional[float] = None
    change_percent: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    beta: Optional[float] = None
    market_cap: Optional[float] = None
    fetched_at: datetime
    raw: Dict[str, Any] = Field(default_factory=dict)


class CandidateIdea(BaseModel):
    ticker: str
    direction: str
    score: float
    event_theme: str
    event_probability: float
    signal_quality: float
    valuation_score: float
    momentum_score: float
    exposure_weight: float
    probability_edge: float
    confidence: float
    market_source: str
    market_id: str
    market_question: str
    market_url: str
    rationale: str = ""
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertPayload(BaseModel):
    digest: str
    ideas: List[CandidateIdea]
    created_at: datetime
    llm_summary: str = ""
