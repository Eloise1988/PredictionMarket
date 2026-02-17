from __future__ import annotations

import math
from statistics import median
from typing import Iterable, List

from prediction_agent.backtest.models import BacktestMetrics


def compute_metrics(net_returns: Iterable[float], horizon_days: int) -> BacktestMetrics:
    values = list(net_returns)
    if not values:
        return BacktestMetrics(
            trade_count=0,
            win_rate=0.0,
            avg_return=0.0,
            median_return=0.0,
            cumulative_return=0.0,
            sharpe_like=0.0,
            max_drawdown=0.0,
        )

    count = len(values)
    wins = sum(1 for r in values if r > 0)
    avg = sum(values) / count
    med = float(median(values))

    equity_curve = _equity_curve(values)
    cumulative = equity_curve[-1] - 1.0
    max_dd = _max_drawdown(equity_curve)

    variance = sum((r - avg) ** 2 for r in values) / max(1, count - 1)
    stdev = math.sqrt(variance)
    sharpe_like = 0.0
    if stdev > 0:
        annualizer = math.sqrt(252 / max(1, horizon_days))
        sharpe_like = (avg / stdev) * annualizer

    return BacktestMetrics(
        trade_count=count,
        win_rate=wins / count,
        avg_return=avg,
        median_return=med,
        cumulative_return=cumulative,
        sharpe_like=sharpe_like,
        max_drawdown=max_dd,
    )


def objective(metrics: BacktestMetrics, min_trades: int) -> float:
    if metrics.trade_count < min_trades:
        return -1e9

    # Encourage stable positive expectancy and penalize drawdown.
    return (
        0.50 * metrics.avg_return
        + 0.25 * metrics.win_rate
        + 0.20 * metrics.sharpe_like / 10.0
        - 0.35 * abs(metrics.max_drawdown)
    )


def _equity_curve(returns: List[float]) -> List[float]:
    eq = [1.0]
    value = 1.0
    for r in returns:
        value *= 1.0 + r
        eq.append(value)
    return eq


def _max_drawdown(equity_curve: List[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        if peak <= 0:
            continue
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)
    return max_dd
