from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

from prediction_agent.backtest.calibration import walk_forward_calibrate
from prediction_agent.backtest.bias_checks import format_bias_checks, run_bias_checks
from prediction_agent.backtest.engine import BacktestEngine
from prediction_agent.backtest.price_loader import PriceHistoryLoader
from prediction_agent.backtest.reporting import calibration_report_to_doc, format_report_text
from prediction_agent.clients.alpha_vantage import AlphaVantageClient
from prediction_agent.config import get_settings
from prediction_agent.storage.mongo import MongoStore
from prediction_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def run_backtest(start: Optional[datetime], end: Optional[datetime], save_report: bool) -> str:
    settings = get_settings()
    store = MongoStore(settings.mongodb_uri, settings.mongodb_db)
    alpha = AlphaVantageClient(settings.alpha_vantage_api_key, settings.alpha_vantage_base_url)

    ideas = store.get_candidate_ideas(start=start, end=end)
    if not ideas:
        return "No candidate ideas found in MongoDB for the selected window."

    tickers = sorted({i.ticker.upper() for i in ideas})

    first_idea_dt = min(i.created_at for i in ideas)
    last_idea_dt = max(i.created_at for i in ideas)
    history_start = first_idea_dt.date() - timedelta(days=45)
    history_end = last_idea_dt.date() + timedelta(days=settings.backtest_horizon_days + 45)

    loader = PriceHistoryLoader(store, alpha)
    series_by_ticker: Dict[date, float]
    all_series: Dict[str, Dict[date, float]] = {}

    for ticker in tickers:
        series_by_ticker = loader.load_series(ticker, start_date=history_start, end_date=history_end)
        if series_by_ticker:
            all_series[ticker] = series_by_ticker

    ideas = [i for i in ideas if i.ticker.upper() in all_series]
    if not ideas:
        return "No ideas have enough price history for backtest in the selected window."

    engine = BacktestEngine(
        transaction_cost_bps=settings.backtest_transaction_cost_bps,
        horizon_days=settings.backtest_horizon_days,
    )
    thresholds = _build_thresholds(
        settings.backtest_threshold_min,
        settings.backtest_threshold_max,
        settings.backtest_threshold_step,
    )

    report = walk_forward_calibrate(
        ideas=ideas,
        series_by_ticker=all_series,
        engine=engine,
        thresholds=thresholds,
        train_days=settings.calibration_train_days,
        val_days=settings.calibration_val_days,
        min_trades=settings.backtest_min_trades,
    )

    bias_text = format_bias_checks(run_bias_checks(report, report.validation_trades))
    text = f"{format_report_text(report)}\n\n{bias_text}"
    if save_report:
        store.save_backtest_report(calibration_report_to_doc(report))

    return text


def _build_thresholds(min_thr: float, max_thr: float, step: float) -> List[float]:
    if step <= 0:
        step = 0.02
    if min_thr > max_thr:
        min_thr, max_thr = max_thr, min_thr

    out: List[float] = []
    x = min_thr
    while x <= max_thr + 1e-9:
        out.append(round(x, 4))
        x += step
    return out


def _parse_utc_date(raw: Optional[str], end_of_day: bool = False) -> Optional[datetime]:
    if not raw:
        return None
    d = date.fromisoformat(raw)
    if end_of_day:
        return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest and calibrate idea threshold")
    parser.add_argument("--start", type=str, default=None, help="UTC start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="UTC end date YYYY-MM-DD")
    parser.add_argument("--save-report", action="store_true", help="Store report in MongoDB")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    start = _parse_utc_date(args.start, end_of_day=False)
    end = _parse_utc_date(args.end, end_of_day=True)

    text = run_backtest(start=start, end=end, save_report=args.save_report)
    print(text)


if __name__ == "__main__":
    main()
