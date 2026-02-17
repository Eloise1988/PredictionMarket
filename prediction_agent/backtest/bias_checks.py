from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev
from typing import Iterable, List

from prediction_agent.backtest.models import BacktestTrade, CalibrationReport


@dataclass
class BiasCheckResult:
    name: str
    status: str
    detail: str


def run_bias_checks(report: CalibrationReport, validation_trades: Iterable[BacktestTrade]) -> List[BiasCheckResult]:
    trades = list(validation_trades)
    checks: List[BiasCheckResult] = []

    if report.overall_validation_metrics and report.overall_validation_metrics.trade_count < 30:
        checks.append(
            BiasCheckResult(
                name="sample_size",
                status="warn",
                detail="Validation sample is small (<30 trades). Results are fragile.",
            )
        )
    else:
        checks.append(
            BiasCheckResult(
                name="sample_size",
                status="ok",
                detail="Validation sample size is adequate for first-pass calibration.",
            )
        )

    thresholds = [f.best_threshold for f in report.folds]
    if len(thresholds) >= 2 and pstdev(thresholds) > 0.08:
        checks.append(
            BiasCheckResult(
                name="threshold_stability",
                status="warn",
                detail="Chosen threshold varies strongly across folds; possible regime instability.",
            )
        )
    else:
        checks.append(
            BiasCheckResult(
                name="threshold_stability",
                status="ok",
                detail="Threshold selection appears stable across folds.",
            )
        )

    if trades:
        top_theme_count = 0
        by_theme = {}
        for t in trades:
            by_theme[t.event_theme] = by_theme.get(t.event_theme, 0) + 1
            top_theme_count = max(top_theme_count, by_theme[t.event_theme])

        concentration = top_theme_count / len(trades)
        if concentration > 0.45:
            checks.append(
                BiasCheckResult(
                    name="theme_concentration",
                    status="warn",
                    detail="Validation trades are concentrated in one theme (>45%). Diversification may be overstated.",
                )
            )
        else:
            checks.append(
                BiasCheckResult(
                    name="theme_concentration",
                    status="ok",
                    detail="Validation trade distribution is not heavily concentrated in one theme.",
                )
            )

    if report.overall_validation_metrics and report.overall_validation_metrics.max_drawdown < -0.20:
        checks.append(
            BiasCheckResult(
                name="drawdown",
                status="warn",
                detail="Modeled drawdown exceeds 20%; risk controls likely insufficient.",
            )
        )
    else:
        checks.append(
            BiasCheckResult(
                name="drawdown",
                status="ok",
                detail="Modeled drawdown is within 20% threshold.",
            )
        )

    return checks


def format_bias_checks(checks: List[BiasCheckResult]) -> str:
    lines = ["Bias Diagnostics"]
    for c in checks:
        lines.append(f"- [{c.status.upper()}] {c.name}: {c.detail}")
    return "\n".join(lines)
