from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional


@dataclass
class BacktestTrade:
    created_at: datetime
    ticker: str
    direction: str
    score: float
    confidence: float
    event_theme: str
    entry_date: date
    exit_date: date
    entry_close: float
    exit_close: float
    raw_return: float
    net_return: float


@dataclass
class BacktestMetrics:
    trade_count: int
    win_rate: float
    avg_return: float
    median_return: float
    cumulative_return: float
    sharpe_like: float
    max_drawdown: float


@dataclass
class FoldResult:
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    best_threshold: float
    train_metrics: BacktestMetrics
    val_metrics: BacktestMetrics


@dataclass
class CalibrationReport:
    horizon_days: int
    transaction_cost_bps: float
    tested_thresholds: List[float]
    folds: List[FoldResult]
    recommended_threshold: Optional[float]
    overall_validation_metrics: Optional[BacktestMetrics]
    validation_trades: List[BacktestTrade] = field(default_factory=list)
