from __future__ import annotations

import argparse
import logging
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
        signals = self._fetch_signals()
        if not signals:
            logger.info("No prediction signals fetched")
            return

        dedup: Dict[str, PredictionSignal] = {}
        for signal in signals:
            if self.settings.polymarket_only_mode and signal.source != "polymarket":
                continue
            key = f"{signal.source}:{signal.market_id}"
            existing = dedup.get(key)
            if existing is None or signal.updated_at > existing.updated_at:
                dedup[key] = signal

        ranked = sorted(dedup.values(), key=lambda s: (s.liquidity, s.volume_24h), reverse=True)
        finance = [s for s in ranked if _is_finance_signal(s)]
        if limit > 0:
            finance = finance[:limit]

        print(
            "rank | liq_usd | vol24h_usd | prob_yes | gate | source | market_id | question"
        )
        print("-" * 180)
        for idx, s in enumerate(finance, start=1):
            gate = self._gate_reason(s)
            print(
                f"{idx:>4} | "
                f"{s.liquidity:>10.0f} | "
                f"{s.volume_24h:>10.0f} | "
                f"{s.prob_yes*100:>7.2f}% | "
                f"{gate:<11} | "
                f"{s.source:<10} | "
                f"{s.market_id:<12} | "
                f"{(s.question or '').strip()[:160]}"
            )

        logger.info(
            "Finance table complete | total_input=%s source_filtered=%s finance_ranked=%s shown=%s",
            len(signals),
            len(dedup),
            len([s for s in ranked if _is_finance_signal(s)]),
            len(finance),
        )

    def show_cross_venue_table(self, limit: int = 0, min_similarity: float | None = None) -> None:
        source_signals = self._fetch_cross_venue_signals()
        pm_all = source_signals.get("polymarket", [])
        ks_all = source_signals.get("kalshi", [])
        polymarket_signals = [s for s in pm_all if _is_finance_signal(s)]
        kalshi_signals = [s for s in ks_all if _is_finance_signal(s)]

        if ks_all and not kalshi_signals:
            logger.warning(
                "No Kalshi signals classified as finance; using all Kalshi markets for cross-venue matching fallback | total_kalshi=%s",
                len(ks_all),
            )
            kalshi_signals = ks_all

        if not polymarket_signals or not kalshi_signals:
            logger.info(
                "Cross-venue table unavailable | polymarket_total=%s polymarket_finance=%s kalshi_total=%s kalshi_finance=%s",
                len(pm_all),
                len(polymarket_signals),
                len(ks_all),
                len(kalshi_signals),
            )
            return

        threshold = self.settings.cross_venue_min_similarity if min_similarity is None else min_similarity
        matches = match_cross_venue_markets(
            polymarket_signals=polymarket_signals,
            kalshi_signals=kalshi_signals,
            min_similarity=max(0.0, min(1.0, float(threshold))),
        )
        total_matches = len(matches)
        if limit > 0:
            matches = matches[:limit]

        print(
            "rank | liq_sum_usd | prob_pm | prob_kalshi | prob_diff_pp | sim | polymarket_id | kalshi_id | polymarket_question | kalshi_question"
        )
        print("-" * 220)
        for idx, m in enumerate(matches, start=1):
            print(
                f"{idx:>4} | "
                f"{m.liquidity_sum:>11.0f} | "
                f"{m.polymarket.prob_yes*100:>7.2f}% | "
                f"{m.kalshi.prob_yes*100:>10.2f}% | "
                f"{m.probability_diff*100:>12.2f} | "
                f"{m.text_similarity:>4.2f} | "
                f"{m.polymarket.market_id:<12} | "
                f"{m.kalshi.market_id:<14} | "
                f"{(m.polymarket.question or '').strip()[:80]} | "
                f"{(m.kalshi.question or '').strip()[:80]}"
            )

        logger.info(
            "Cross-venue table complete | polymarket_total=%s polymarket_finance=%s kalshi_total=%s kalshi_used=%s matches=%s shown=%s min_similarity=%.2f",
            len(pm_all),
            len(polymarket_signals),
            len(ks_all),
            len(kalshi_signals),
            total_matches,
            len(matches),
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
            key=lambda s: (s.liquidity, s.volume_24h),
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


def _is_finance_signal(signal: PredictionSignal) -> bool:
    raw = signal.raw or {}
    raw_text = " ".join(
        [
            str(raw.get("category") or ""),
            str(raw.get("eventCategory") or ""),
            str(raw.get("subCategory") or ""),
            str(raw.get("eventTitle") or ""),
            str(raw.get("eventSlug") or ""),
            str(raw.get("slug") or ""),
            str(raw.get("event_ticker") or ""),
            str(raw.get("series_ticker") or ""),
            str(raw.get("ticker") or ""),
            str(raw.get("subtitle") or ""),
            str(raw.get("yes_sub_title") or ""),
            str(raw.get("no_sub_title") or ""),
            " ".join(str(x) for x in raw.get("tags", []) if isinstance(x, str)),
        ]
    ).lower()
    question = (signal.question or "").lower()

    finance_terms = (
        "fed",
        "fomc",
        "interest rate",
        "inflation",
        "cpi",
        "pce",
        "gdp",
        "recession",
        "treasury",
        "yield",
        "stock",
        "s&p",
        "nasdaq",
        "dow",
        "bitcoin",
        "ethereum",
        "crypto",
        "oil",
        "brent",
        "wti",
        "earnings",
        "tariff",
        "unemployment",
        "jobless",
        "job growth",
        "rate cut",
        "rate hike",
        "federal funds",
        "treasury",
        "10y",
        "2y",
        "bond",
        "sp500",
        "spx",
        "nas100",
        "russell",
        "vix",
        "dxy",
        "dollar",
        "commodity",
        "gold",
        "silver",
        "gasoline",
        "crude",
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "solana",
        "sol",
    )
    sports_terms = (
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

    # Hard reject sports-like contracts even if metadata is noisy.
    if any(x in raw_text for x in sports_terms) or any(x in question for x in sports_terms):
        return False

    # If category metadata exists, trust explicit finance tags.
    if raw_text.strip() and any(
        x in raw_text for x in ("finance", "markets", "economy", "business", "crypto", "macro", "rates", "inflation")
    ):
        return True

    if any(x in question for x in finance_terms):
        return True

    if _looks_like_competition_market(question):
        return False

    return False


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
        help="Print finance markets ranked by liquidity with gate status, then exit",
    )
    parser.add_argument(
        "--show-cross-venue-table",
        action="store_true",
        help="Print matched finance markets across Polymarket and Kalshi, ranked by combined liquidity",
    )
    parser.add_argument(
        "--table-limit",
        type=int,
        default=0,
        help="Optional max rows for --show-finance-table (0 = all)",
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

    if args.show_finance_table:
        agent.show_finance_table(limit=max(0, args.table_limit))
        return

    if args.show_cross_venue_table:
        agent.show_cross_venue_table(limit=max(0, args.table_limit), min_similarity=args.cross_min_similarity)
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
