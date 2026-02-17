from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from prediction_agent.backtest.models import BacktestMetrics, CalibrationReport


def calibration_report_to_doc(report: CalibrationReport) -> Dict:
    return {
        "created_at": datetime.now(timezone.utc),
        "horizon_days": report.horizon_days,
        "transaction_cost_bps": report.transaction_cost_bps,
        "tested_thresholds": report.tested_thresholds,
        "recommended_threshold": report.recommended_threshold,
        "validation_trade_count": len(report.validation_trades),
        "overall_validation_metrics": _metrics_to_dict(report.overall_validation_metrics)
        if report.overall_validation_metrics
        else None,
        "folds": [
            {
                "train_start": f.train_start,
                "train_end": f.train_end,
                "val_start": f.val_start,
                "val_end": f.val_end,
                "best_threshold": f.best_threshold,
                "train_metrics": _metrics_to_dict(f.train_metrics),
                "val_metrics": _metrics_to_dict(f.val_metrics),
            }
            for f in report.folds
        ],
    }


def format_report_text(report: CalibrationReport) -> str:
    lines: List[str] = []
    lines.append("Backtest + Walk-Forward Calibration")
    lines.append("")
    lines.append(f"Horizon: {report.horizon_days} trading days")
    lines.append(f"Transaction cost: {report.transaction_cost_bps:.1f} bps/side")
    if report.recommended_threshold is not None:
        lines.append(f"Recommended score threshold: {report.recommended_threshold:.2f}")
    else:
        lines.append("Recommended score threshold: unavailable (insufficient folds)")

    if report.overall_validation_metrics:
        m = report.overall_validation_metrics
        lines.append("")
        lines.extend(_metrics_lines("Overall validation", m))

    lines.append("")
    lines.append(f"Fold count: {len(report.folds)}")
    for i, fold in enumerate(report.folds, start=1):
        lines.append(
            f"Fold {i}: threshold={fold.best_threshold:.2f}, "
            f"train_trades={fold.train_metrics.trade_count}, val_trades={fold.val_metrics.trade_count}, "
            f"val_avg={fold.val_metrics.avg_return:.3%}, val_win={fold.val_metrics.win_rate:.1%}"
        )

    lines.append("")
    lines.append("Bias controls: chronological walk-forward split, next-day entry, and explicit transaction costs.")
    return "\n".join(lines)


def _metrics_to_dict(metrics: BacktestMetrics) -> Dict:
    return {
        "trade_count": metrics.trade_count,
        "win_rate": metrics.win_rate,
        "avg_return": metrics.avg_return,
        "median_return": metrics.median_return,
        "cumulative_return": metrics.cumulative_return,
        "sharpe_like": metrics.sharpe_like,
        "max_drawdown": metrics.max_drawdown,
    }


def _metrics_lines(label: str, m: BacktestMetrics) -> List[str]:
    return [
        f"{label}: trades={m.trade_count}",
        f"  win_rate={m.win_rate:.1%}",
        f"  avg_return={m.avg_return:.3%}",
        f"  median_return={m.median_return:.3%}",
        f"  cumulative_return={m.cumulative_return:.3%}",
        f"  sharpe_like={m.sharpe_like:.2f}",
        f"  max_drawdown={m.max_drawdown:.2%}",
    ]
