from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from prediction_agent.clients.alpha_vantage import AlphaVantageClient
from prediction_agent.clients.llm_briefer import LLMBriefer
from prediction_agent.clients.llm_market_mapper import LLMMarketMapper, MarketStockCandidate
from prediction_agent.clients.telegram import TelegramClient
from prediction_agent.config import get_settings
from prediction_agent.connectors.kalshi import KalshiConnector
from prediction_agent.connectors.polymarket import PolymarketConnector
from prediction_agent.engine.consensus import aggregate_consensus
from prediction_agent.engine.portfolio_selector import select_top_ideas
from prediction_agent.engine.valuation import clamp01, signal_quality, valuation_score
from prediction_agent.knowledge.ticker_profiles import get_ticker_background
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
        if self.settings.kalshi_enabled and not self.settings.polymarket_only_mode:
            self.connectors.append(
                KalshiConnector(base_url=self.settings.kalshi_base_url, limit=self.settings.kalshi_limit)
            )

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
        self.market_mapper = LLMMarketMapper(
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
            logger.info("No signal passed liquidity/probability gates")
            return []

        logger.info("Processing prediction signals (eligible=%s)", len(filtered_signals))
        self.store.save_signals(filtered_signals)

        if not self.market_mapper.enabled():
            logger.warning("Market mapper disabled because OPENAI_API_KEY is missing")
            return []

        candidates_by_market: List[Tuple[PredictionSignal, List[MarketStockCandidate]]] = []
        tickers_to_load: set[str] = set()

        markets_for_llm = filtered_signals[: self.settings.max_markets_for_llm]
        for signal in markets_for_llm:
            candidates = self.market_mapper.map_market(
                signal,
                max_tickers=self.settings.llm_map_max_tickers,
                min_linkage_score=self.settings.llm_min_linkage_score,
            )
            if not candidates:
                continue

            candidates_by_market.append((signal, candidates))
            for c in candidates:
                tickers_to_load.add(c.ticker)

        if not candidates_by_market:
            logger.info("No stock candidates returned by LLM for eligible markets")
            self._log_market_samples(filtered_signals)
            return []

        snapshots = self._load_equity_snapshots(sorted(tickers_to_load), dry_run=dry_run)

        ideas: List[CandidateIdea] = []
        for signal, candidates in candidates_by_market:
            idea = self._select_best_valuation_idea(signal, candidates, snapshots)
            if idea is not None:
                ideas.append(idea)

        if not ideas:
            logger.info("No candidate ideas generated after valuation selection")
            return []

        ideas = aggregate_consensus(ideas)

        selected = select_top_ideas(
            ideas,
            top_n=self.settings.alert_top_n,
            min_score=self.settings.alert_min_score,
            max_per_ticker=1,
            max_per_theme=self.settings.alert_top_n,
        )

        if not selected:
            logger.info("No idea passed final score threshold", extra={"threshold": self.settings.alert_min_score})
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

    def _select_best_valuation_idea(
        self,
        signal: PredictionSignal,
        candidates: List[MarketStockCandidate],
        snapshots: Dict[str, EquitySnapshot],
    ) -> CandidateIdea | None:
        prob_yes = signal.prob_yes
        event_sign = 1 if prob_yes >= 0.5 else -1
        edge = abs(prob_yes - 0.5) * 2
        quality = signal_quality(signal)

        scored_rows = []
        for c in candidates:
            snap = snapshots.get(c.ticker)
            val_raw = valuation_score(snap)
            has_metrics = _has_valuation_metrics(snap)

            final_direction = c.direction_if_yes if event_sign > 0 else _invert_direction(c.direction_if_yes)
            val_for_direction = val_raw if final_direction == "long" else 1.0 - val_raw

            total_score = clamp01(
                0.35 * edge
                + 0.35 * val_for_direction
                + 0.20 * quality
                + 0.10 * clamp01(c.linkage_score)
            )

            scored_rows.append((has_metrics, val_for_direction, total_score, c, final_direction, snap))

        if not scored_rows:
            return None

        if not any(x[0] for x in scored_rows):
            logger.info("Skipping market due to missing valuation metrics for LLM candidates: %s", signal.question[:160])
            return None

        # User-requested behavior: pick the best-valuation stock from each market's candidate list.
        scored_rows = [x for x in scored_rows if x[0]]
        scored_rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
        _, best_val, best_score, best_candidate, final_direction, snap = scored_rows[0]

        company_name = (snap.name if snap and snap.name else "").strip()
        sector = (snap.sector if snap and snap.sector else "").strip()
        background = get_ticker_background(best_candidate.ticker, company_name=company_name, sector=sector)

        confidence = clamp01(0.60 * edge + 0.25 * best_candidate.linkage_score + 0.15 * quality)

        rationale = (
            f"LLM mapped market to impacted stocks; selected highest valuation-fit candidate "
            f"({best_candidate.ticker}) with direction={final_direction}. "
            f"LLM rationale: {best_candidate.rationale}"
        )

        return CandidateIdea(
            ticker=best_candidate.ticker,
            direction=final_direction,
            score=best_score,
            event_theme="dynamic_llm_impact",
            event_probability=prob_yes,
            signal_quality=quality,
            valuation_score=best_val,
            momentum_score=0.5,
            exposure_weight=clamp01(best_candidate.linkage_score),
            probability_edge=edge,
            confidence=confidence,
            market_source=signal.source,
            market_id=signal.market_id,
            market_question=signal.question,
            market_url=signal.url,
            rationale=rationale,
            created_at=datetime.now(timezone.utc),
            metadata={
                "selection_method": "best_valuation_from_llm_candidates",
                "llm_direction_if_yes": best_candidate.direction_if_yes,
                "llm_linkage_score": best_candidate.linkage_score,
                "llm_rationale": best_candidate.rationale,
                "company_name": company_name,
                "sector": sector,
                "company_background": background,
                "probability_source": str(signal.raw.get("probability_source", "")),
                "llm_candidate_count": len(candidates),
            },
        )

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

        return all_signals

    def _apply_signal_filters(self, signals: List[PredictionSignal]) -> List[PredictionSignal]:
        dedup: Dict[str, PredictionSignal] = {}
        for signal in signals:
            if self.settings.polymarket_only_mode and signal.source != "polymarket":
                continue

            key = f"{signal.source}:{signal.market_id}"
            existing = dedup.get(key)
            if existing is None or signal.updated_at > existing.updated_at:
                dedup[key] = signal

        ranked = sorted(
            dedup.values(),
            key=lambda s: (s.liquidity + s.volume_24h),
            reverse=True,
        )

        filtered: List[PredictionSignal] = []
        for signal in ranked:
            p = signal.prob_yes
            if not (p <= self.settings.probability_low_threshold or p >= self.settings.probability_high_threshold):
                continue

            if signal.liquidity < self.settings.min_signal_liquidity:
                continue

            if signal.volume_24h < self.settings.min_signal_volume_24h:
                continue

            edge = abs(p - 0.5) * 2
            if edge < self.settings.min_probability_edge:
                continue

            filtered.append(signal)

            if len(filtered) >= self.settings.max_signals_per_cycle:
                break

        return filtered

    def _log_market_samples(self, signals: List[PredictionSignal]) -> None:
        sample = signals[:12]
        for s in sample:
            logger.info(
                "Eligible market sample | source=%s prob_yes=%.4f liq=%.0f vol24h=%.0f q=%s",
                s.source,
                s.prob_yes,
                s.liquidity,
                s.volume_24h,
                s.question[:180],
            )

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


def _invert_direction(direction: str) -> str:
    d = (direction or "").lower().strip()
    if d == "long":
        return "short"
    if d == "short":
        return "long"
    return "long"


def _has_valuation_metrics(snapshot: EquitySnapshot | None) -> bool:
    if snapshot is None:
        return False
    return any(
        x is not None
        for x in (
            snapshot.pe_ratio,
            snapshot.pb_ratio,
            snapshot.ev_to_ebitda,
        )
    )


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
