from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median
from typing import Dict, List

from prediction_agent.backtest.engine import BacktestEngine, IdeaLike
from prediction_agent.backtest.models import BacktestTrade, CalibrationReport, FoldResult
from prediction_agent.backtest.stats import objective


def walk_forward_calibrate(
    ideas: List[IdeaLike],
    series_by_ticker: Dict,
    engine: BacktestEngine,
    thresholds: List[float],
    train_days: int,
    val_days: int,
    min_trades: int,
) -> CalibrationReport:
    if not ideas:
        return CalibrationReport(
            horizon_days=engine.horizon_days,
            transaction_cost_bps=engine.transaction_cost_bps,
            tested_thresholds=thresholds,
            folds=[],
            recommended_threshold=None,
            overall_validation_metrics=None,
            validation_trades=[],
        )

    ideas_sorted = sorted(ideas, key=lambda x: x.created_at)
    start = ideas_sorted[0].created_at
    end = ideas_sorted[-1].created_at

    folds: List[FoldResult] = []
    val_trades_all: List[BacktestTrade] = []

    train_start = start
    while True:
        train_end = train_start + timedelta(days=train_days)
        val_start = train_end + timedelta(seconds=1)
        val_end = val_start + timedelta(days=val_days)
        if val_end > end:
            break

        train_ideas = [x for x in ideas_sorted if train_start <= x.created_at <= train_end]
        val_ideas = [x for x in ideas_sorted if val_start <= x.created_at <= val_end]
        if not train_ideas or not val_ideas:
            train_start = train_start + timedelta(days=val_days)
            continue

        best_threshold = _choose_threshold(train_ideas, series_by_ticker, engine, thresholds, min_trades)

        train_trades = engine.simulate(train_ideas, series_by_ticker, min_score=best_threshold)
        val_trades = engine.simulate(val_ideas, series_by_ticker, min_score=best_threshold)

        train_metrics = engine.metrics(train_trades)
        val_metrics = engine.metrics(val_trades)
        val_trades_all.extend(val_trades)

        folds.append(
            FoldResult(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                best_threshold=best_threshold,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
        )

        train_start = train_start + timedelta(days=val_days)

    recommended = median([f.best_threshold for f in folds]) if folds else None
    overall = engine.metrics(val_trades_all) if val_trades_all else None

    return CalibrationReport(
        horizon_days=engine.horizon_days,
        transaction_cost_bps=engine.transaction_cost_bps,
        tested_thresholds=thresholds,
        folds=folds,
        recommended_threshold=float(recommended) if recommended is not None else None,
        overall_validation_metrics=overall,
        validation_trades=val_trades_all,
    )


def _choose_threshold(
    ideas: List[IdeaLike],
    series_by_ticker: Dict,
    engine: BacktestEngine,
    thresholds: List[float],
    min_trades: int,
) -> float:
    best_threshold = thresholds[0]
    best_obj = -1e18

    for threshold in thresholds:
        trades = engine.simulate(ideas, series_by_ticker, min_score=threshold)
        metrics = engine.metrics(trades)
        score = objective(metrics, min_trades=min_trades)
        if score > best_obj:
            best_obj = score
            best_threshold = threshold

    return best_threshold
