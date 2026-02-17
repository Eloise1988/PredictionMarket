from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from prediction_agent.clients.alpha_vantage import AlphaVantageClient
from prediction_agent.clients.llm_briefer import LLMBriefer
from prediction_agent.clients.telegram import TelegramClient
from prediction_agent.config import get_settings
from prediction_agent.connectors.kalshi import KalshiConnector
from prediction_agent.connectors.polymarket import PolymarketConnector
from prediction_agent.engine.consensus import aggregate_consensus
from prediction_agent.engine.portfolio_selector import select_top_ideas
from prediction_agent.engine.scoring import build_candidate_ideas
from prediction_agent.engine.theme_matcher import ThemeMatcher
from prediction_agent.models import AlertPayload, CandidateIdea, EquitySnapshot, PredictionSignal
from prediction_agent.storage.mongo import MongoStore
from prediction_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class DecisionAgent:
    def __init__(self):
        self.settings = get_settings()
        self.store = MongoStore(self.settings.mongodb_uri, self.settings.mongodb_db)

        self.connectors = []
        if self.settings.polymarket_enabled:
            self.connectors.append(
                PolymarketConnector(
                    gamma_base_url=self.settings.polymarket_gamma_base_url,
                    clob_base_url=self.settings.polymarket_clob_base_url,
                    limit=self.settings.polymarket_limit,
                )
            )
        if self.settings.kalshi_enabled:
            self.connectors.append(
                KalshiConnector(base_url=self.settings.kalshi_base_url, limit=self.settings.kalshi_limit)
            )

        self.theme_matcher = ThemeMatcher(Path(__file__).parent / "knowledge" / "theme_map.json")
        self.alpha_vantage = AlphaVantageClient(
            api_key=self.settings.alpha_vantage_api_key,
            base_url=self.settings.alpha_vantage_base_url,
        )
        self.telegram = TelegramClient(
            bot_token=self.settings.telegram_bot_token,
            chat_id=self.settings.telegram_chat_id,
        )
        self.briefer = LLMBriefer(
            api_key=self.settings.openai_api_key,
            model=self.settings.openai_model,
            timeout_seconds=self.settings.openai_timeout_seconds,
        )

    def run_cycle(self, dry_run: bool = False) -> List[CandidateIdea]:
        signals = self._fetch_signals()
        if not signals:
            logger.info("No prediction signals fetched")
            return []

        return self.process_signals(signals, dry_run=dry_run)

    def process_signals(self, signals: List[PredictionSignal], dry_run: bool = False) -> List[CandidateIdea]:
        if not signals:
            return []

        filtered_signals = self._apply_signal_filters(signals)
        if not filtered_signals:
            logger.info("No signal passed filtering gates")
            return []

        logger.info("Processing prediction signals", extra={"count": len(filtered_signals)})
        self.store.save_signals(filtered_signals)

        tickers_to_load: set[str] = set()
        themed_signals: List[tuple[PredictionSignal, str, List[dict]]] = []

        for signal in filtered_signals:
            matches = self.theme_matcher.match(signal, strict=self.settings.strict_theme_filter)
            for theme, cfg in matches:
                equities = cfg.get("equities", [])
                if not equities:
                    continue
                themed_signals.append((signal, theme, equities))
                for e in equities:
                    if "ticker" in e:
                        tickers_to_load.add(str(e["ticker"]).upper())

        if not themed_signals:
            logger.info("No themed signals matched current mapping")
            return []

        snapshots = self._load_equity_snapshots(sorted(tickers_to_load), dry_run=dry_run)

        ideas: List[CandidateIdea] = []
        for signal, theme, equities in themed_signals:
            ideas.extend(build_candidate_ideas(signal, theme, equities, snapshots))

        if not ideas:
            logger.info("No candidate ideas generated")
            return []

        ideas = aggregate_consensus(ideas)

        selected = select_top_ideas(
            ideas,
            top_n=self.settings.alert_top_n,
            min_score=self.settings.alert_min_score,
            max_per_ticker=1,
            max_per_theme=2,
        )

        if not selected:
            logger.info("No idea passed score threshold", extra={"threshold": self.settings.alert_min_score})
            return []

        self.store.save_ideas(selected)

        digest = self.store.ideas_digest(selected)
        if self.store.has_recent_alert(digest, self.settings.alert_cooldown_minutes):
            logger.info("Skipping alert due to cooldown", extra={"digest": digest})
            return selected

        summary = self.briefer.summarize(selected)
        alert = AlertPayload(
            digest=digest,
            ideas=selected,
            created_at=datetime.now(timezone.utc),
            llm_summary=summary,
        )

        if dry_run:
            logger.info("Dry run mode: not sending Telegram", extra={"digest": digest})
        else:
            sent = self.telegram.send_message(_format_telegram_message(alert))
            logger.info("Alert send attempted", extra={"sent": sent, "digest": digest})
            if sent:
                self.store.save_alert(alert)
            else:
                logger.warning("Alert was not saved because Telegram send failed", extra={"digest": digest})
        return selected

    def _fetch_signals(self) -> List[PredictionSignal]:
        all_signals: List[PredictionSignal] = []
        for connector in self.connectors:
            try:
                signals = connector.fetch_signals()
                all_signals.extend(signals)
            except Exception as exc:
                logger.warning(
                    "Connector failed",
                    extra={"source": getattr(connector, "source_name", "unknown"), "error": str(exc)},
                )

        return self._apply_signal_filters(all_signals)

    def _apply_signal_filters(self, signals: List[PredictionSignal]) -> List[PredictionSignal]:
        dedup: Dict[str, PredictionSignal] = {}
        for signal in signals:
            key = f"{signal.source}:{signal.market_id}"
            existing = dedup.get(key)
            if existing is None or signal.updated_at > existing.updated_at:
                dedup[key] = signal

        ranked = sorted(
            dedup.values(),
            key=lambda s: (abs(s.prob_yes - 0.5), s.liquidity + s.volume_24h),
            reverse=True,
        )

        filtered: List[PredictionSignal] = []
        for signal in ranked:
            edge = abs(signal.prob_yes - 0.5) * 2
            if edge < self.settings.min_probability_edge:
                continue
            if (
                signal.liquidity < self.settings.min_signal_liquidity
                and signal.volume_24h < self.settings.min_signal_volume_24h
            ):
                continue
            filtered.append(signal)

        return filtered[: self.settings.max_signals_per_cycle]

    def _load_equity_snapshots(self, tickers: List[str], dry_run: bool) -> Dict[str, EquitySnapshot]:
        snapshots: Dict[str, EquitySnapshot] = {}
        for ticker in tickers:
            cached = self.store.get_cached_equity(ticker, max_age_seconds=900)
            if cached:
                snapshots[ticker] = cached
                continue

            # During dry-runs we avoid spending API quota.
            if dry_run:
                continue

            snap = self.alpha_vantage.fetch_snapshot(ticker)
            if snap:
                snapshots[ticker] = snap
                self.store.upsert_equity(snap)

        return snapshots


def _format_telegram_message(alert: AlertPayload) -> str:
    lines = ["Prediction Market Equity Signals", ""]
    for i, idea in enumerate(alert.ideas, start=1):
        psrc = str(idea.metadata.get("probability_source", "")).strip()
        lines.append(
            f"{i}. {idea.direction.upper()} {idea.ticker} | score {idea.score:.2f} | "
            f"theme {idea.event_theme} | P(event) {idea.event_probability:.1%}"
        )
        lines.append(f"   Source: {idea.market_source} | {idea.market_question}")
        if psrc:
            lines.append(f"   P(source): {psrc}")
        lines.append(f"   Link: {idea.market_url}")
        background = str(idea.metadata.get("company_background", "")).strip()
        if background:
            lines.append(f"   Background: {background}")

    if alert.llm_summary:
        lines.append("")
        lines.append(alert.llm_summary)

    lines.append("")
    lines.append("Risk: Decision support only, not investment advice.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction market equity signal agent")
    parser.add_argument("--once", action="store_true", help="Run a single decision cycle")
    parser.add_argument("--dry-run", action="store_true", help="Do not send Telegram message")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    agent = DecisionAgent()

    if args.once:
        ideas = agent.run_cycle(dry_run=args.dry_run)
        logger.info("Cycle complete", extra={"ideas": len(ideas)})
        return

    while True:
        start = time.time()
        try:
            ideas = agent.run_cycle(dry_run=args.dry_run)
            logger.info("Cycle complete", extra={"ideas": len(ideas)})
        except Exception as exc:
            logger.exception("Cycle failed", extra={"error": str(exc)})

        elapsed = time.time() - start
        sleep_for = max(1, settings.loop_interval_seconds - int(elapsed))
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
