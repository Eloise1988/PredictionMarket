from __future__ import annotations

import argparse
import logging
import math
import re
import time
from dataclasses import dataclass, field
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
from prediction_agent.engine.cross_venue import match_cross_venue_markets
from prediction_agent.engine.portfolio_selector import select_top_ideas
from prediction_agent.engine.valuation import clamp01, signal_quality, valuation_score
from prediction_agent.knowledge.ticker_profiles import get_ticker_background
from prediction_agent.models import AlertPayload, CandidateIdea, EquitySnapshot, PredictionSignal
from prediction_agent.storage.mongo import MongoStore
from prediction_agent.utils.logging import configure_logging

logger = logging.getLogger(__name__)

_ARB_BUDGET_USD = 1000.0
_ARB_SLIPPAGE_BPS = 15.0
_ARB_SPREAD_IMPACT = 0.20
_ARB_DEFAULT_POLYMARKET_FEE_BPS = 25.0


@dataclass
class FilterDiagnostics:
    total_input: int = 0
    after_source_filter: int = 0
    after_dedup: int = 0
    after_finance_filter: int = 0
    rejected_liquidity: int = 0
    rejected_volume: int = 0
    rejected_probability: int = 0
    rejected_edge: int = 0
    passed: int = 0
    sample_rows: List[str] = field(default_factory=list)


class DecisionAgent:
    def __init__(self):
        self.settings = get_settings()
        self.store = MongoStore(self.settings.mongodb_uri, self.settings.mongodb_db)

        self.connectors = []
        if self.settings.polymarket_enabled:
            effective_polymarket_limit = self.settings.polymarket_limit
            if self.settings.finance_only_mode:
                effective_polymarket_limit = max(
                    effective_polymarket_limit,
                    self.settings.polymarket_min_scan_markets,
                )
            if effective_polymarket_limit != self.settings.polymarket_limit:
                logger.info(
                    "Raising Polymarket scan depth to satisfy target selection | configured=%s effective=%s",
                    self.settings.polymarket_limit,
                    effective_polymarket_limit,
                )
            self.connectors.append(
                PolymarketConnector(
                    gamma_base_url=self.settings.polymarket_gamma_base_url,
                    clob_base_url=self.settings.polymarket_clob_base_url,
                    limit=effective_polymarket_limit,
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

    def show_finance_table(self, limit: int = 0) -> None:
        # Backward-compatible command name: this is now the Polymarket target-universe table.
        self.show_source_table(source="polymarket", limit=limit)

    def show_source_table(self, source: str, limit: int = 0) -> None:
        source = (source or "").strip().lower()
        if source not in {"polymarket", "kalshi"}:
            logger.warning("Unsupported source for table: %s", source)
            return

        source_signals = self._fetch_source_signals(source)
        if not source_signals:
            logger.info("No %s signals fetched", source)
            return

        dedup: Dict[str, PredictionSignal] = {}
        for signal in source_signals:
            key = f"{signal.source}:{signal.market_id}"
            existing = dedup.get(key)
            if existing is None or signal.updated_at > existing.updated_at:
                dedup[key] = signal

        ranked = sorted(dedup.values(), key=lambda s: (s.updated_at, s.liquidity, s.volume_24h), reverse=True)
        universe = [s for s in ranked if _is_finance_signal(s)]
        passed_universe = [s for s in universe if self._gate_reason(s) == "passed"]
        passed_total = len(passed_universe)
        if limit > 0:
            passed_universe = passed_universe[:limit]

        print("rank | liq_usd | vol24h_usd | yes_px | no_px | prob_yes | category | gate | source | market_id | link | question")
        print("-" * 260)
        for idx, s in enumerate(passed_universe, start=1):
            gate = "passed"
            category = _signal_category(s)
            link = _signal_link(s)
            yes_px, no_px = _signal_yes_no_prices(s)
            print(
                f"{idx:>4} | "
                f"{s.liquidity:>10.0f} | "
                f"{s.volume_24h:>10.0f} | "
                f"{_format_price_cents(yes_px):>6} | "
                f"{_format_price_cents(no_px):>6} | "
                f"{s.prob_yes*100:>7.2f}% | "
                f"{category:<11} | "
                f"{gate:<11} | "
                f"{s.source:<10} | "
                f"{s.market_id:<12} | "
                f"{link[:84]:<84} | "
                f"{(s.question or '').strip()[:160]}"
            )

        logger.info(
            "Source table complete | source=%s total_input=%s source_filtered=%s universe_ranked=%s passed=%s shown=%s",
            source,
            len(source_signals),
            len(dedup),
            len(universe),
            passed_total,
            len(passed_universe),
        )

    def show_cross_venue_table(self, limit: int = 0, min_similarity: float | None = None) -> None:
        source_signals = self._fetch_cross_venue_signals()
        pm_all = source_signals.get("polymarket", [])
        ks_all = source_signals.get("kalshi", [])
        polymarket_universe = [s for s in pm_all if _is_finance_signal(s)]
        kalshi_universe = [s for s in ks_all if _is_finance_signal(s)]
        polymarket_signals = [s for s in polymarket_universe if self._gate_reason(s) == "passed"]
        kalshi_signals = [s for s in kalshi_universe if self._gate_reason(s) == "passed"]

        if not polymarket_signals or not kalshi_signals:
            logger.info(
                "Cross-venue table unavailable | polymarket_total=%s polymarket_universe=%s polymarket_passed=%s kalshi_total=%s kalshi_universe=%s kalshi_passed=%s",
                len(pm_all),
                len(polymarket_universe),
                len(polymarket_signals),
                len(ks_all),
                len(kalshi_universe),
                len(kalshi_signals),
            )
            return

        threshold = self.settings.cross_venue_min_similarity if min_similarity is None else min_similarity
        matches = match_cross_venue_markets(
            polymarket_signals=polymarket_signals,
            kalshi_signals=kalshi_signals,
            min_similarity=max(0.0, min(1.0, float(threshold))),
        )
        if not matches and threshold > 0.10:
            relaxed = max(0.10, round(float(threshold) * 0.55, 2))
            matches = match_cross_venue_markets(
                polymarket_signals=polymarket_signals,
                kalshi_signals=kalshi_signals,
                min_similarity=relaxed,
            )
            if matches:
                logger.info(
                    "Cross-venue match fallback used | original_min_similarity=%.2f relaxed_min_similarity=%.2f",
                    float(threshold),
                    relaxed,
                )
        total_matches = len(matches)

        rows = []
        for m in matches:
            edge_hint = _cross_venue_edge_hint(m.polymarket.prob_yes, m.kalshi.prob_yes)
            pm_yes, pm_no = _signal_yes_no_prices(m.polymarket)
            ka_yes, ka_no = _signal_yes_no_prices(m.kalshi)
            arb = _cross_venue_arbitrage_metrics(
                m.polymarket,
                m.kalshi,
                budget_usd=_ARB_BUDGET_USD,
                slippage_bps=_ARB_SLIPPAGE_BPS,
                spread_impact=_ARB_SPREAD_IMPACT,
            )
            rows.append(
                {
                    "match": m,
                    "edge_hint": edge_hint,
                    "pm_yes": pm_yes,
                    "pm_no": pm_no,
                    "ka_yes": ka_yes,
                    "ka_no": ka_no,
                    "arb_flag": "yes" if arb.get("is_arb") else "no",
                    "arb_pnl": float(arb.get("net_pnl", 0.0)),
                }
            )

        # Rank table by highest net $1k arbitrage payoff.
        rows.sort(
            key=lambda r: (
                float(r["arb_pnl"]),
                float(r["match"].liquidity_sum),
                float(r["match"].text_similarity),
            ),
            reverse=True,
        )
        if limit > 0:
            rows = rows[:limit]

        print(
            "rank | liq_sum_usd | yes_pm | no_pm | yes_ka | no_ka | prob_diff_pp | arb | arb_pnl_1k_net | edge_hint | sim | cat_pm | cat_ka | polymarket_id | kalshi_id | polymarket_question | kalshi_question"
        )
        print("-" * 260)
        for idx, row in enumerate(rows, start=1):
            m = row["match"]
            edge_hint = str(row["edge_hint"])
            pm_yes = float(row["pm_yes"])
            pm_no = float(row["pm_no"])
            ka_yes = float(row["ka_yes"])
            ka_no = float(row["ka_no"])
            arb_flag = str(row["arb_flag"])
            arb_pnl = float(row["arb_pnl"])
            print(
                f"{idx:>4} | "
                f"{m.liquidity_sum:>11.0f} | "
                f"{_format_price_cents(pm_yes):>6} | "
                f"{_format_price_cents(pm_no):>6} | "
                f"{_format_price_cents(ka_yes):>6} | "
                f"{_format_price_cents(ka_no):>6} | "
                f"{m.probability_diff*100:>12.2f} | "
                f"{arb_flag:<3} | "
                f"{arb_pnl:>10.2f} | "
                f"{edge_hint:<20} | "
                f"{m.text_similarity:>4.2f} | "
                f"{_signal_category(m.polymarket):<7} | "
                f"{_signal_category(m.kalshi):<7} | "
                f"{m.polymarket.market_id:<12} | "
                f"{m.kalshi.market_id:<14} | "
                f"{(m.polymarket.question or '').strip()[:80]} | "
                f"{(m.kalshi.question or '').strip()[:80]}"
            )

        logger.info(
            "Cross-venue table complete | polymarket_total=%s polymarket_universe=%s polymarket_passed=%s kalshi_total=%s kalshi_universe=%s kalshi_passed=%s matches=%s shown=%s min_similarity=%.2f",
            len(pm_all),
            len(polymarket_universe),
            len(polymarket_signals),
            len(ks_all),
            len(kalshi_universe),
            len(kalshi_signals),
            total_matches,
            len(rows),
            max(0.0, min(1.0, float(threshold))),
        )

    def show_cross_venue_question_lists(self, limit: int = 0, min_similarity: float | None = None) -> None:
        source_signals = self._fetch_cross_venue_signals()
        pm_all = source_signals.get("polymarket", [])
        ks_all = source_signals.get("kalshi", [])
        polymarket_universe = [s for s in pm_all if _is_finance_signal(s)]
        kalshi_universe = [s for s in ks_all if _is_finance_signal(s)]
        polymarket_signals = [s for s in polymarket_universe if self._gate_reason(s) == "passed"]
        kalshi_signals = [s for s in kalshi_universe if self._gate_reason(s) == "passed"]

        if not polymarket_signals or not kalshi_signals:
            logger.info(
                "Cross-venue question lists unavailable | polymarket_total=%s polymarket_universe=%s polymarket_passed=%s kalshi_total=%s kalshi_universe=%s kalshi_passed=%s",
                len(pm_all),
                len(polymarket_universe),
                len(polymarket_signals),
                len(ks_all),
                len(kalshi_universe),
                len(kalshi_signals),
            )
            return

        threshold = self.settings.cross_venue_min_similarity if min_similarity is None else min_similarity
        matches = match_cross_venue_markets(
            polymarket_signals=polymarket_signals,
            kalshi_signals=kalshi_signals,
            min_similarity=max(0.0, min(1.0, float(threshold))),
        )
        if not matches and threshold > 0.10:
            relaxed = max(0.10, round(float(threshold) * 0.55, 2))
            matches = match_cross_venue_markets(
                polymarket_signals=polymarket_signals,
                kalshi_signals=kalshi_signals,
                min_similarity=relaxed,
            )
            if matches:
                logger.info(
                    "Cross-venue question-list fallback used | original_min_similarity=%.2f relaxed_min_similarity=%.2f",
                    float(threshold),
                    relaxed,
                )
        total_matches = len(matches)

        rows = []
        for m in matches:
            arb = _cross_venue_arbitrage_metrics(
                m.polymarket,
                m.kalshi,
                budget_usd=_ARB_BUDGET_USD,
                slippage_bps=_ARB_SLIPPAGE_BPS,
                spread_impact=_ARB_SPREAD_IMPACT,
            )
            rows.append(
                {
                    "match": m,
                    "arb_pnl": float(arb.get("net_pnl", 0.0)),
                }
            )

        rows.sort(
            key=lambda r: (
                float(r["arb_pnl"]),
                float(r["match"].liquidity_sum),
                float(r["match"].text_similarity),
            ),
            reverse=True,
        )
        if limit > 0:
            rows = rows[:limit]

        print("polymarket_questions")
        print("-" * 200)
        for idx, row in enumerate(rows, start=1):
            m = row["match"]
            print(f"{idx:>4} | {m.polymarket.market_id:<18} | {(m.polymarket.question or '').strip()}")

        print("")
        print("kalshi_questions")
        print("-" * 200)
        for idx, row in enumerate(rows, start=1):
            m = row["match"]
            print(f"{idx:>4} | {m.kalshi.market_id:<18} | {(m.kalshi.question or '').strip()}")

        logger.info(
            "Cross-venue question lists complete | polymarket_total=%s polymarket_universe=%s polymarket_passed=%s kalshi_total=%s kalshi_universe=%s kalshi_passed=%s matches=%s shown=%s min_similarity=%.2f",
            len(pm_all),
            len(polymarket_universe),
            len(polymarket_signals),
            len(ks_all),
            len(kalshi_universe),
            len(kalshi_signals),
            total_matches,
            len(rows),
            max(0.0, min(1.0, float(threshold))),
        )

    def process_signals(self, signals: List[PredictionSignal], dry_run: bool = False) -> List[CandidateIdea]:
        if not signals:
            return []

        filtered_signals, diag = self._apply_signal_filters(signals)
        if not filtered_signals:
            logger.info(
                "No signal passed liquidity/probability gates | "
                "total_input=%s source_filtered=%s dedup=%s finance_candidates=%s "
                "rejected_liquidity=%s rejected_volume=%s rejected_probability=%s rejected_edge=%s passed=%s",
                diag.total_input,
                diag.after_source_filter,
                diag.after_dedup,
                diag.after_finance_filter,
                diag.rejected_liquidity,
                diag.rejected_volume,
                diag.rejected_probability,
                diag.rejected_edge,
                diag.passed,
            )
            for row in diag.sample_rows:
                logger.info("Gate detail | %s", row)
            return []

        markets_for_llm = self._select_markets_for_mapping(filtered_signals)
        if not markets_for_llm:
            logger.info("No markets selected for LLM mapping after diversification")
            return []

        logger.info(
            "Processing prediction signals (passed_pool=%s selected_for_mapping=%s)",
            len(filtered_signals),
            len(markets_for_llm),
        )
        self._log_eligible_markets(markets_for_llm)
        self.store.save_signals(markets_for_llm)

        if not self.market_mapper.enabled():
            logger.warning("Market mapper disabled because OPENAI_API_KEY is missing")
            return []

        candidates_by_market: List[Tuple[PredictionSignal, List[MarketStockCandidate]]] = []
        tickers_to_load: set[str] = set()

        for signal in markets_for_llm:
            candidates = self.market_mapper.map_market(
                signal,
                max_tickers=self.settings.llm_map_max_tickers,
                min_linkage_score=self.settings.llm_min_linkage_score,
            )
            if not candidates:
                logger.info("LLM candidates | none | question=%s", signal.question[:220])
                continue
            self._log_llm_candidates(signal, candidates)

            candidates_by_market.append((signal, candidates))
            for c in candidates:
                tickers_to_load.add(c.ticker)

        if not candidates_by_market:
            logger.info("No stock candidates returned by LLM for eligible markets")
            self._log_market_samples(filtered_signals)
            return []

        snapshots = self._load_equity_snapshots(sorted(tickers_to_load))

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

    def _fetch_source_signals(self, source: str) -> List[PredictionSignal]:
        source = (source or "").strip().lower()
        if source == "polymarket":
            if not self.settings.polymarket_enabled:
                return []
            connector = PolymarketConnector(
                gamma_base_url=self.settings.polymarket_gamma_base_url,
                clob_base_url=self.settings.polymarket_clob_base_url,
                limit=max(self.settings.polymarket_limit, self.settings.polymarket_min_scan_markets),
            )
            return connector.fetch_signals()

        if source == "kalshi":
            if not self.settings.kalshi_enabled:
                return []
            connector = KalshiConnector(
                base_url=self.settings.kalshi_base_url,
                limit=max(self.settings.kalshi_limit, self.settings.kalshi_min_scan_markets),
            )
            return connector.fetch_signals()

        return []

    def _fetch_cross_venue_signals(self) -> Dict[str, List[PredictionSignal]]:
        out: Dict[str, List[PredictionSignal]] = {"polymarket": [], "kalshi": []}

        if self.settings.polymarket_enabled:
            try:
                connector = PolymarketConnector(
                    gamma_base_url=self.settings.polymarket_gamma_base_url,
                    clob_base_url=self.settings.polymarket_clob_base_url,
                    limit=max(self.settings.polymarket_limit, self.settings.polymarket_min_scan_markets),
                )
                out["polymarket"] = connector.fetch_signals()
            except Exception as exc:
                logger.warning("Cross-venue fetch failed for polymarket", extra={"error": str(exc)})

        if self.settings.kalshi_enabled:
            try:
                connector = KalshiConnector(
                    base_url=self.settings.kalshi_base_url,
                    limit=max(self.settings.kalshi_limit, self.settings.kalshi_min_scan_markets),
                )
                out["kalshi"] = connector.fetch_signals()
            except Exception as exc:
                logger.warning("Cross-venue fetch failed for kalshi", extra={"error": str(exc)})

        return out

    def _apply_signal_filters(self, signals: List[PredictionSignal]) -> tuple[List[PredictionSignal], FilterDiagnostics]:
        diag = FilterDiagnostics(total_input=len(signals))

        dedup: Dict[str, PredictionSignal] = {}
        for signal in signals:
            if self.settings.polymarket_only_mode and signal.source != "polymarket":
                continue

            key = f"{signal.source}:{signal.market_id}"
            existing = dedup.get(key)
            if existing is None or signal.updated_at > existing.updated_at:
                dedup[key] = signal

        diag.after_source_filter = len(dedup)

        ranked = sorted(
            dedup.values(),
            key=lambda s: (s.updated_at, s.liquidity, s.volume_24h),
            reverse=True,
        )
        diag.after_dedup = len(ranked)

        candidates = [s for s in ranked if (not self.settings.finance_only_mode) or _is_finance_signal(s)]
        diag.after_finance_filter = len(candidates)

        filtered: List[PredictionSignal] = []
        for signal in candidates:
            reason = self._gate_reason(signal, diag=diag)

            if len(diag.sample_rows) < 30:
                diag.sample_rows.append(
                    "source="
                    f"{signal.source} liq={signal.liquidity:.0f} vol24h={signal.volume_24h:.0f} "
                    f"prob_yes={signal.prob_yes*100:.1f}% reason={reason} q={signal.question[:180]}"
                )

            if reason != "passed":
                continue

            filtered.append(signal)
            if self.settings.finance_only_mode and len(filtered) >= self.settings.finance_passed_pool_size:
                break
            if not self.settings.finance_only_mode and len(filtered) >= self.settings.max_signals_per_cycle:
                break

        diag.passed = len(filtered)
        return filtered, diag

    def _gate_reason(self, signal: PredictionSignal, diag: FilterDiagnostics | None = None) -> str:
        if signal.liquidity < self.settings.min_signal_liquidity:
            if diag is not None:
                diag.rejected_liquidity += 1
            return "liquidity"
        if signal.volume_24h < self.settings.min_signal_volume_24h:
            if diag is not None:
                diag.rejected_volume += 1
            return "volume"

        if self.settings.enable_probability_gate:
            p = signal.prob_yes
            if not (p <= self.settings.probability_low_threshold or p >= self.settings.probability_high_threshold):
                if diag is not None:
                    diag.rejected_probability += 1
                return "probability"
            edge = abs(p - 0.5) * 2
            if edge < self.settings.min_probability_edge:
                if diag is not None:
                    diag.rejected_edge += 1
                return "edge"

        return "passed"

    def _select_markets_for_mapping(self, passed_signals: List[PredictionSignal]) -> List[PredictionSignal]:
        if not passed_signals:
            return []

        if not self.settings.finance_only_mode:
            return passed_signals[: self.settings.max_markets_for_llm]

        target = max(1, self.settings.top_liquidity_finance_markets)
        pool = passed_signals[: max(target, self.settings.finance_passed_pool_size)]

        selected: List[PredictionSignal] = []
        if self.settings.llm_select_diverse_markets and self.market_mapper.enabled():
            llm_pool = pool[: max(target, self.settings.llm_market_selection_pool)]
            selected_ids = self.market_mapper.select_diverse_markets(llm_pool, target_count=target)
            if selected_ids:
                id_to_signal = {s.market_id: s for s in llm_pool}
                for market_id in selected_ids:
                    signal = id_to_signal.get(market_id)
                    if signal is None:
                        continue
                    selected.append(signal)
                logger.info(
                    "LLM market selector raw picks | count=%s ids=%s",
                    len(selected),
                    ",".join(s.market_id for s in selected)[:1000],
                )

        if self.settings.diversify_markets and selected:
            selected = self._select_diverse_markets_deterministic(
                selected,
                target_count=len(selected),
                already_selected=[],
            )

        selected_ids = {s.market_id for s in selected}
        remaining = [s for s in pool if s.market_id not in selected_ids]

        if len(selected) < target:
            selected.extend(
                self._select_diverse_markets_deterministic(
                    remaining,
                    target_count=target - len(selected),
                    already_selected=selected,
                )
            )

        if len(selected) < target:
            selected_ids = {s.market_id for s in selected}
            for signal in remaining:
                if signal.market_id in selected_ids:
                    continue
                selected.append(signal)
                selected_ids.add(signal.market_id)
                if len(selected) >= target:
                    break

        logger.info(
            "Market selection summary | passed_pool=%s pool_used=%s target=%s selected=%s llm_enabled=%s",
            len(passed_signals),
            len(pool),
            target,
            len(selected),
            self.settings.llm_select_diverse_markets and self.market_mapper.enabled(),
        )
        return selected[:target]

    def _select_diverse_markets_deterministic(
        self,
        signals: List[PredictionSignal],
        target_count: int,
        already_selected: List[PredictionSignal] | None = None,
    ) -> List[PredictionSignal]:
        if target_count <= 0:
            return []

        selected = list(already_selected or [])
        event_counts: Dict[str, int] = {}
        selected_token_sets: List[set[str]] = []
        selected_ids = {s.market_id for s in selected}

        for s in selected:
            event_key = _event_group_key(s)
            event_counts[event_key] = event_counts.get(event_key, 0) + 1
            selected_token_sets.append(_question_token_set(s.question))

        out: List[PredictionSignal] = []
        for signal in signals:
            if signal.market_id in selected_ids:
                continue

            if self.settings.diversify_markets:
                event_key = _event_group_key(signal)
                if event_counts.get(event_key, 0) >= max(1, self.settings.diversify_max_per_event):
                    continue

                tokens = _question_token_set(signal.question)
                if tokens and selected_token_sets:
                    max_sim = max(_jaccard_similarity(tokens, prior) for prior in selected_token_sets)
                    if max_sim >= self.settings.diversify_text_similarity:
                        continue

            out.append(signal)
            selected_ids.add(signal.market_id)

            if self.settings.diversify_markets:
                event_key = _event_group_key(signal)
                event_counts[event_key] = event_counts.get(event_key, 0) + 1
                selected_token_sets.append(_question_token_set(signal.question))

            if len(out) >= target_count:
                break

        return out

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

    def _log_eligible_markets(self, signals: List[PredictionSignal]) -> None:
        for i, s in enumerate(signals, start=1):
            logger.info(
                "Bet %d | source=%s | liquidity=%.0f | prob_yes=%.1f%% | question=%s",
                i,
                s.source,
                s.liquidity,
                s.prob_yes * 100.0,
                s.question[:220],
            )

    def _log_llm_candidates(self, signal: PredictionSignal, candidates: List[MarketStockCandidate]) -> None:
        rows = ", ".join(
            f"{c.ticker}({c.direction_if_yes},link={c.linkage_score:.2f})"
            for c in candidates
        )
        logger.info(
            "LLM candidates | prob_yes=%.1f%% | question=%s | candidates=%s",
            signal.prob_yes * 100.0,
            signal.question[:180],
            rows[:1000],
        )

    def _load_equity_snapshots(self, tickers: List[str]) -> Dict[str, EquitySnapshot]:
        snapshots: Dict[str, EquitySnapshot] = {}
        for ticker in tickers:
            cached = self.store.get_cached_equity(ticker, max_age_seconds=900)
            if cached:
                snapshots[ticker] = cached
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


_TARGET_CATEGORY_ORDER = (
    "geopolitics",
    "politics",
    "economy",
    "finance",
    "tech",
)

_FINANCE_TERMS = (
    "fed",
    "fomc",
    "interest rate",
    "rate cut",
    "rate hike",
    "federal funds",
    "treasury",
    "yield",
    "bond",
    "stock",
    "equity",
    "etf",
    "s&p",
    "nasdaq",
    "dow",
    "sp500",
    "spx",
    "nas100",
    "russell",
    "vix",
    "dxy",
    "dollar",
    "bitcoin",
    "btc",
    "ethereum",
    "eth",
    "crypto",
    "solana",
    "sol",
    "commodity",
    "gold",
    "silver",
    "oil",
    "brent",
    "wti",
    "gasoline",
    "crude",
    "earnings",
)

_ECONOMY_TERMS = (
    "economy",
    "economic",
    "inflation",
    "cpi",
    "pce",
    "gdp",
    "recession",
    "unemployment",
    "jobless",
    "job growth",
    "payrolls",
)

_POLITICS_TERMS = (
    "election",
    "president",
    "senate",
    "house",
    "congress",
    "white house",
    "supreme court",
    "cabinet",
    "nominate",
    "nomination",
    "policy",
    "government",
    "fiscal",
    "spending bill",
    "executive order",
    "prime minister",
    "parliament",
    "party control",
    "midterm",
    "pardon",
)

_GEOPOLITICS_TERMS = (
    "war",
    "conflict",
    "ceasefire",
    "sanction",
    "tariff",
    "trade war",
    "nato",
    "iran",
    "israel",
    "gaza",
    "ukraine",
    "russia",
    "china",
    "taiwan",
    "north korea",
    "south china sea",
    "strait",
    "missile",
    "invasion",
)

_TECH_TERMS = (
    "ai",
    "artificial intelligence",
    "semiconductor",
    "chip",
    "nvidia",
    "openai",
    "microsoft",
    "google",
    "meta",
    "apple",
    "amazon",
    "tesla",
    "software",
    "cloud",
    "cybersecurity",
    "robotaxi",
    "market cap",
    "fdv",
    "launch",
)

_SPORTS_TERMS = (
    "world cup",
    "fifa",
    "premier league",
    "english premier league",
    "la liga",
    "serie a",
    "bundesliga",
    "division",
    "stanley cup",
    "nba",
    "nfl",
    "mlb",
    "nhl",
    "super bowl",
    "olympic",
    "champions league",
    "tournament",
    "playoff",
    "match",
    "soccer",
    "football club",
    "wins the",
    "win the",
    "to win the",
    "championship",
)

_ENTERTAINMENT_TERMS = (
    "academy awards",
    "oscars",
    "best picture",
    "grammy",
    "emmy",
    "golden globe",
    "movie",
    "film",
    "actor",
    "actress",
)

_EXCLUDED_CATEGORY_TERMS = (
    "sports",
    "sport",
    "soccer",
    "football",
    "basketball",
    "baseball",
    "hockey",
    "tennis",
    "golf",
    "mma",
    "boxing",
    "esports",
    "entertainment",
    "movie",
    "film",
    "music",
    "celebrity",
    "television",
    "award",
)

_GEOPOLITICS_CATEGORY_TERMS = ("geopolitics", "geopolitical", "international", "war", "conflict", "world affairs")
_POLITICS_CATEGORY_TERMS = ("politics", "political", "government", "election", "policy", "current affairs")
_ECONOMY_CATEGORY_TERMS = ("economy", "economic", "macro", "inflation", "employment", "rates")
_FINANCE_CATEGORY_TERMS = ("finance", "financial", "markets", "market", "business", "crypto", "stocks", "equities")
_TECH_CATEGORY_TERMS = ("technology", "tech", "ai", "artificial intelligence")


def _is_finance_signal(signal: PredictionSignal) -> bool:
    # Legacy function name retained. It now means: "in target universe".
    return _signal_category(signal) in _TARGET_CATEGORY_ORDER


def _signal_category(signal: PredictionSignal) -> str:
    raw_text, question = _signal_text_blob(signal)
    combined = f"{raw_text} {question}".strip()
    if not combined:
        return "other"

    explicit_category = _signal_explicit_category(signal)
    if explicit_category == "excluded":
        return "excluded"

    if _contains_any(combined, _SPORTS_TERMS):
        return "excluded"
    if _contains_any(combined, _ENTERTAINMENT_TERMS):
        return "excluded"
    if _looks_like_competition_market(question):
        return "excluded"

    if explicit_category:
        return explicit_category

    # Prefer explicit metadata categories when present.
    if raw_text.strip():
        if _contains_any(raw_text, ("geopolitics", "geopolitical", "international", "war")):
            return "geopolitics"
        if _contains_any(raw_text, ("politics", "political", "government", "election")):
            return "politics"
        if _contains_any(raw_text, ("economy", "economic", "macro", "inflation", "employment")):
            return "economy"
        if _contains_any(raw_text, ("finance", "markets", "business", "crypto", "rates")):
            return "finance"
        if _contains_any(raw_text, ("technology", "tech", "ai", "software")):
            return "tech"

    if _contains_any(combined, _GEOPOLITICS_TERMS):
        return "geopolitics"
    if _contains_any(combined, _POLITICS_TERMS):
        return "politics"
    if _contains_any(combined, _ECONOMY_TERMS):
        return "economy"
    if _contains_any(combined, _FINANCE_TERMS):
        return "finance"
    if _contains_any(combined, _TECH_TERMS):
        return "tech"
    return "other"


def _signal_explicit_category(signal: PredictionSignal) -> str:
    raw = signal.raw or {}
    category_blob = " ".join(
        [
            str(raw.get("category") or ""),
            str(raw.get("eventCategory") or ""),
            str(raw.get("subCategory") or ""),
            " ".join(str(x) for x in raw.get("tags", []) if isinstance(x, str)),
        ]
    )
    if not category_blob.strip():
        return ""
    if _contains_any(category_blob, _EXCLUDED_CATEGORY_TERMS):
        return "excluded"
    if _contains_any(category_blob, _GEOPOLITICS_CATEGORY_TERMS):
        return "geopolitics"
    if _contains_any(category_blob, _POLITICS_CATEGORY_TERMS):
        return "politics"
    if _contains_any(category_blob, _ECONOMY_CATEGORY_TERMS):
        return "economy"
    if _contains_any(category_blob, _FINANCE_CATEGORY_TERMS):
        return "finance"
    if _contains_any(category_blob, _TECH_CATEGORY_TERMS):
        return "tech"
    return ""


def _signal_text_blob(signal: PredictionSignal) -> tuple[str, str]:
    raw = signal.raw or {}
    raw_text = " ".join(
        [
            str(raw.get("category") or ""),
            str(raw.get("eventCategory") or ""),
            str(raw.get("subCategory") or ""),
            str(raw.get("eventTitle") or ""),
            str(raw.get("eventSlug") or ""),
            str(raw.get("eventTicker") or ""),
            str(raw.get("slug") or ""),
            str(raw.get("event_ticker") or ""),
            str(raw.get("series_ticker") or ""),
            str(raw.get("ticker") or ""),
            str(raw.get("title") or ""),
            str(raw.get("subtitle") or ""),
            str(raw.get("yes_sub_title") or ""),
            str(raw.get("no_sub_title") or ""),
            " ".join(str(x) for x in raw.get("tags", []) if isinstance(x, str)),
        ]
    ).lower()
    question = (signal.question or "").lower()
    return raw_text, question


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = _normalize_text(text)
    tokens = set(normalized.split())
    for term in terms:
        term_norm = _normalize_text(term)
        if not term_norm:
            continue
        if " " in term_norm:
            if f" {term_norm} " in f" {normalized} ":
                return True
        else:
            if term_norm in tokens:
                return True
    return False


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _format_price_cents(prob: float) -> str:
    cents = max(0.0, min(100.0, float(prob) * 100.0))
    if abs(cents - round(cents)) < 0.05:
        return f"{int(round(cents))}c"
    return f"{cents:.1f}c"


def _signal_yes_no_prices(signal: PredictionSignal) -> tuple[float, float]:
    raw = signal.raw or {}

    yes_px = (
        _to_prob(raw.get("yes_price"))
        or _to_prob(raw.get("yes_ask_dollars"))
        or _to_prob(raw.get("yes_ask"))
        or _to_prob(raw.get("last_price_dollars"))
        or _to_prob(raw.get("last_price"))
        or _to_prob(raw.get("yes_bid_dollars"))
        or _to_prob(raw.get("yes_bid"))
    )
    if yes_px is None:
        yes_px = max(0.0, min(1.0, float(signal.prob_yes)))

    no_px = (
        _to_prob(raw.get("no_price"))
        or _to_prob(raw.get("no_ask_dollars"))
        or _to_prob(raw.get("no_ask"))
    )
    if no_px is None:
        yes_bid = _to_prob(raw.get("yes_bid_dollars")) or _to_prob(raw.get("yes_bid"))
        if yes_bid is not None:
            no_px = max(0.0, min(1.0, 1.0 - yes_bid))
    if no_px is None:
        no_px = max(0.0, min(1.0, 1.0 - yes_px))

    return yes_px, no_px


def _to_prob(value: object) -> float | None:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if raw < 0:
        return None
    if raw <= 1:
        return max(0.0, min(1.0, raw))
    if raw <= 100:
        return max(0.0, min(1.0, raw / 100.0))
    return None


def _to_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "1", "yes", "y"}:
            return True
        if cleaned in {"false", "0", "no", "n", ""}:
            return False
    return None


def _signal_link(signal: PredictionSignal) -> str:
    raw = signal.raw or {}
    src = (signal.source or "").lower().strip()

    if src == "polymarket":
        if signal.url:
            return signal.url
        event_slug = str(raw.get("eventSlug") or raw.get("event_slug") or raw.get("slug") or "").strip()
        if event_slug:
            return f"https://polymarket.com/event/{event_slug}"
        return "https://polymarket.com"

    if src == "kalshi":
        if signal.url:
            return signal.url
        slug = str(raw.get("slug") or "").strip()
        if slug:
            return f"https://kalshi.com/markets/{slug}"
        event_ticker = str(raw.get("event_ticker") or "").strip()
        if event_ticker:
            return f"https://kalshi.com/markets/{event_ticker.lower()}"
        return "https://kalshi.com/markets"

    return signal.url or ""


def _looks_like_competition_market(question: str) -> bool:
    q = (question or "").lower()
    return bool(
        re.search(r"\bwin\b.*\b(league|cup|division|title|championship)\b", q)
        or re.search(r"\b(beat|defeat)\b", q)
        or "vs " in q
        or " versus " in q
    )


def _event_group_key(signal: PredictionSignal) -> str:
    raw = signal.raw or {}
    base = (
        str(raw.get("eventTitle") or "").strip()
        or str(raw.get("eventSlug") or "").strip()
        or str(raw.get("slug") or "").strip()
        or signal.question
    )
    norm = re.sub(r"[^a-z0-9]+", " ", base.lower()).strip()
    norm = re.sub(r"\s+", " ", norm)
    return f"{signal.source}:{norm[:180]}"


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "will",
    "before",
    "after",
    "into",
    "from",
    "that",
    "this",
    "have",
    "has",
    "had",
    "are",
    "was",
    "were",
    "would",
    "could",
    "should",
    "about",
    "next",
    "than",
    "over",
    "under",
    "between",
    "more",
    "less",
    "year",
}


def _question_token_set(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union == 0:
        return 0.0
    return inter / union


def _cross_venue_edge_hint(prob_pm: float, prob_kalshi: float) -> str:
    # Diagnostic spread hint only; not execution advice.
    diff = float(prob_pm) - float(prob_kalshi)
    if diff >= 0.02:
        return "yes_kalshi/no_pm"
    if diff <= -0.02:
        return "yes_pm/no_kalshi"
    return "aligned"


def _cross_venue_arbitrage_metrics(
    polymarket_signal: PredictionSignal,
    kalshi_signal: PredictionSignal,
    budget_usd: float,
    slippage_bps: float = 0.0,
    spread_impact: float = 0.0,
) -> dict[str, float | bool | str]:
    leg_a = _cross_venue_leg_metrics(
        polymarket_signal,
        "yes",
        kalshi_signal,
        "no",
        budget_usd=budget_usd,
        slippage_bps=slippage_bps,
        spread_impact=spread_impact,
        leg_name="buy_pm_yes_buy_ka_no",
    )
    leg_b = _cross_venue_leg_metrics(
        polymarket_signal,
        "no",
        kalshi_signal,
        "yes",
        budget_usd=budget_usd,
        slippage_bps=slippage_bps,
        spread_impact=spread_impact,
        leg_name="buy_pm_no_buy_ka_yes",
    )

    candidates = [leg for leg in (leg_a, leg_b) if leg.get("available")]
    if not candidates:
        return {
            "is_arb": False,
            "net_pnl": 0.0,
            "fees_total": 0.0,
            "slippage_cost": 0.0,
            "gross_payout": 0.0,
            "total_cost": 0.0,
            "contracts": 0.0,
            "leg": "no_executable_lock",
        }

    best = max(candidates, key=lambda leg: float(leg.get("net_pnl", -1e9)))
    net_pnl = float(best.get("net_pnl", 0.0))
    return {
        "is_arb": net_pnl > 0.0,
        "net_pnl": net_pnl,
        "fees_total": float(best.get("fees_total", 0.0)),
        "slippage_cost": float(best.get("slippage_cost", 0.0)),
        "gross_payout": float(best.get("gross_payout", 0.0)),
        "total_cost": float(best.get("total_cost", 0.0)),
        "contracts": float(best.get("contracts", 0.0)),
        "leg": str(best.get("leg", "")),
    }


def _cross_venue_leg_metrics(
    pm_signal: PredictionSignal,
    pm_side: str,
    ka_signal: PredictionSignal,
    ka_side: str,
    budget_usd: float,
    slippage_bps: float,
    spread_impact: float,
    leg_name: str,
) -> dict[str, float | str]:
    pm_ask, pm_bid = _signal_side_quote(pm_signal, pm_side)
    ka_ask, ka_bid = _signal_side_quote(ka_signal, ka_side)

    if pm_ask is None or ka_ask is None:
        return {
            "leg": leg_name,
            "available": False,
            "contracts": 0.0,
            "total_cost": 0.0,
            "gross_payout": 0.0,
            "fees_total": 0.0,
            "slippage_cost": 0.0,
            "net_pnl": -1e9,
        }

    pm_exec = _effective_buy_price(
        ask_price=pm_ask,
        bid_price=pm_bid,
        slippage_bps=slippage_bps,
        spread_impact=spread_impact,
    )
    ka_exec = _effective_buy_price(
        ask_price=ka_ask,
        bid_price=ka_bid,
        slippage_bps=slippage_bps,
        spread_impact=spread_impact,
    )

    if pm_exec < 0.0 or ka_exec < 0.0:
        return {
            "leg": leg_name,
            "available": False,
            "contracts": 0.0,
            "total_cost": 0.0,
            "gross_payout": 0.0,
            "fees_total": 0.0,
            "slippage_cost": 0.0,
            "net_pnl": -1e9,
        }

    def _total_cost_usd(contracts: float) -> tuple[float, float]:
        notional = contracts * (pm_exec + ka_exec)
        fees = _estimate_market_fee_usd(pm_signal, contracts, pm_exec) + _estimate_market_fee_usd(
            ka_signal,
            contracts,
            ka_exec,
        )
        return notional + fees, fees

    unit_notional = pm_exec + ka_exec
    if unit_notional <= 0:
        return {
            "leg": leg_name,
            "available": False,
            "contracts": 0.0,
            "total_cost": 0.0,
            "gross_payout": 0.0,
            "fees_total": 0.0,
            "slippage_cost": 0.0,
            "net_pnl": -1e9,
        }

    lo = 0.0
    hi = max(0.0, float(budget_usd) / unit_notional)
    for _ in range(50):
        mid = (lo + hi) / 2.0
        total, _ = _total_cost_usd(mid)
        if total <= budget_usd:
            lo = mid
        else:
            hi = mid

    contracts = lo
    total_cost, fees_total = _total_cost_usd(contracts)
    gross_payout = contracts
    slippage_cost = contracts * max(0.0, (pm_exec - pm_ask)) + contracts * max(0.0, (ka_exec - ka_ask))
    net_pnl = gross_payout - total_cost
    return {
        "leg": leg_name,
        "available": True,
        "contracts": contracts,
        "total_cost": total_cost,
        "gross_payout": gross_payout,
        "fees_total": fees_total,
        "slippage_cost": slippage_cost,
        "net_pnl": net_pnl,
    }


def _signal_side_quote(signal: PredictionSignal, side: str) -> tuple[float | None, float | None]:
    raw = signal.raw or {}
    side_clean = (side or "").strip().lower()

    yes_order_ask = _first_non_none(
        _to_prob(raw.get("yes_ask_dollars")),
        _to_prob(raw.get("yes_ask")),
        _to_prob(raw.get("bestAsk")),
    )
    yes_order_bid = _first_non_none(
        _to_prob(raw.get("yes_bid_dollars")),
        _to_prob(raw.get("yes_bid")),
        _to_prob(raw.get("bestBid")),
    )

    if side_clean == "yes":
        ask = _first_non_none(yes_order_ask, _to_prob(raw.get("yes_price")))
        bid = yes_order_bid
        return _clamp_prob_or_none(ask), _clamp_prob_or_none(bid)

    explicit_no_ask = _first_non_none(
        _to_prob(raw.get("no_ask_dollars")),
        _to_prob(raw.get("no_ask")),
        _to_prob(raw.get("no_price")),
    )
    explicit_no_bid = _first_non_none(_to_prob(raw.get("no_bid_dollars")), _to_prob(raw.get("no_bid")))

    derived_no_ask = _clamp_prob(1.0 - yes_order_bid) if yes_order_bid is not None else None
    derived_no_bid = _clamp_prob(1.0 - yes_order_ask) if yes_order_ask is not None else None

    ask = _first_non_none(explicit_no_ask, derived_no_ask)
    bid = _first_non_none(explicit_no_bid, derived_no_bid)
    return _clamp_prob_or_none(ask), _clamp_prob_or_none(bid)


def _effective_buy_price(ask_price: float, bid_price: float | None, slippage_bps: float, spread_impact: float) -> float:
    ask = _clamp_prob(ask_price)
    bid = _clamp_prob_or_none(bid_price)
    spread = max(0.0, ask - bid) if bid is not None else 0.0
    slip_bps = max(0.0, float(slippage_bps))
    spread_weight = max(0.0, float(spread_impact))
    bps_cost = ask * (slip_bps / 10_000.0)
    spread_cost = spread * spread_weight
    return _clamp_prob(ask + bps_cost + spread_cost)


def _estimate_market_fee_usd(signal: PredictionSignal, contracts: float, execution_price: float) -> float:
    src = (signal.source or "").strip().lower()
    if src == "kalshi":
        return _estimate_kalshi_fee_usd(signal, contracts, execution_price)
    if src == "polymarket":
        return _estimate_polymarket_fee_usd(signal, contracts, execution_price)
    return 0.0


def _estimate_kalshi_fee_usd(signal: PredictionSignal, contracts: float, execution_price: float) -> float:
    if contracts <= 0:
        return 0.0
    raw = signal.raw or {}
    fee_type = str(raw.get("fee_type") or raw.get("feeType") or "quadratic").strip().lower()
    multiplier = _safe_float(raw.get("fee_multiplier") or raw.get("feeMultiplier"), default=1.0)
    if multiplier <= 0:
        multiplier = 1.0

    p = _clamp_prob(execution_price)
    if fee_type in {"", "quadratic"}:
        fee = 0.07 * multiplier * contracts * p * (1.0 - p)
        # Kalshi fee is charged in cents.
        return math.ceil(fee * 100.0) / 100.0

    fee_bps = _safe_float(
        raw.get("fee_bps") or raw.get("feeBps") or raw.get("taker_fee_bps") or raw.get("takerFeeBps"),
        default=0.0,
    )
    if fee_bps <= 0:
        return 0.0
    notional = contracts * p
    return notional * fee_bps / 10_000.0


def _estimate_polymarket_fee_usd(signal: PredictionSignal, contracts: float, execution_price: float) -> float:
    if contracts <= 0:
        return 0.0
    raw = signal.raw or {}
    enabled = _to_bool(raw.get("feesEnabled"))
    if enabled is False:
        return 0.0

    fee_bps = _safe_float(
        raw.get("feeRateBps")
        or raw.get("fee_rate_bps")
        or raw.get("takerFeeBps")
        or raw.get("taker_fee_bps"),
        default=0.0,
    )
    if fee_bps <= 0 and enabled is True:
        fee_bps = _ARB_DEFAULT_POLYMARKET_FEE_BPS
    if fee_bps <= 0:
        return 0.0

    notional = contracts * _clamp_prob(execution_price)
    return notional * fee_bps / 10_000.0


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_non_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _clamp_prob(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp_prob_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return _clamp_prob(value)


def _cross_venue_arbitrage_flag(pm_yes: float, pm_no: float, ka_yes: float, ka_no: float) -> str:
    # Two-leg cross-venue arb (ignores fees/slippage):
    # 1) Buy PM yes + Kalshi no
    # 2) Buy PM no + Kalshi yes
    leg_a = float(pm_yes) + float(ka_no)
    leg_b = float(pm_no) + float(ka_yes)
    return "yes" if min(leg_a, leg_b) < 1.0 else "no"


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
    parser.add_argument(
        "--show-finance-table",
        action="store_true",
        help="Print only passed target-universe markets (finance/economy/politics/geopolitics/tech), then exit",
    )
    parser.add_argument(
        "--show-polymarket-table",
        action="store_true",
        help="Print only passed Polymarket target-universe markets, then exit",
    )
    parser.add_argument(
        "--show-kalshi-table",
        action="store_true",
        help="Print only passed Kalshi target-universe markets, then exit",
    )
    parser.add_argument(
        "--show-cross-venue-table",
        action="store_true",
        help="Print matched passed markets across Polymarket and Kalshi, with yes/no prices and links",
    )
    parser.add_argument(
        "--show-cross-venue-question-lists",
        action="store_true",
        help="Print two lists from matched passed markets: Polymarket questions and Kalshi questions",
    )
    parser.add_argument(
        "--table-limit",
        type=int,
        default=0,
        help="Optional max rows for source-table commands (0 = all)",
    )
    parser.add_argument(
        "--cross-min-similarity",
        type=float,
        default=None,
        help="Optional text-similarity threshold for cross-venue matching (0-1)",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    agent = DecisionAgent()

    if args.show_polymarket_table:
        agent.show_source_table(source="polymarket", limit=max(0, args.table_limit))
        return

    if args.show_kalshi_table:
        agent.show_source_table(source="kalshi", limit=max(0, args.table_limit))
        return

    if args.show_finance_table:
        agent.show_finance_table(limit=max(0, args.table_limit))
        return

    if args.show_cross_venue_table:
        agent.show_cross_venue_table(limit=max(0, args.table_limit), min_similarity=args.cross_min_similarity)
        return

    if args.show_cross_venue_question_lists:
        agent.show_cross_venue_question_lists(limit=max(0, args.table_limit), min_similarity=args.cross_min_similarity)
        return

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
