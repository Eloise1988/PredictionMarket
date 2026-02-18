from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from prediction_agent.engine.cross_venue import match_cross_venue_markets


def _signal(source: str, market_id: str, question: str, prob_yes: float, liq: float, raw: dict | None = None):
    return SimpleNamespace(
        source=source,
        market_id=market_id,
        question=question,
        prob_yes=prob_yes,
        liquidity=liq,
        volume_24h=liq / 10.0,
        updated_at=datetime(2026, 2, 17, 0, 0, tzinfo=timezone.utc),
        url="",
        raw=raw or {},
    )


class CrossVenueMatcherTests(unittest.TestCase):
    def test_matches_similar_questions(self) -> None:
        pm = [
            _signal("polymarket", "pm1", "Will CPI YoY be above 3.0% in June?", 0.62, 500_000),
        ]
        ks = [
            _signal("kalshi", "ks1", "Will June CPI inflation exceed 3.0%?", 0.55, 250_000),
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].polymarket.market_id, "pm1")
        self.assertEqual(matches[0].kalshi.market_id, "ks1")
        self.assertGreater(matches[0].text_similarity, 0.10)

    def test_respects_similarity_threshold(self) -> None:
        pm = [_signal("polymarket", "pm1", "Will CPI YoY be above 3.0% in June?", 0.62, 500_000)]
        ks = [_signal("kalshi", "ks1", "Will unemployment rate rise above 5%?", 0.40, 250_000)]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.40)
        self.assertEqual(matches, [])

    def test_enforces_one_to_one_matching(self) -> None:
        pm = [
            _signal("polymarket", "pm1", "Will Bitcoin close above 120k this month?", 0.51, 700_000),
            _signal("polymarket", "pm2", "Will Ethereum close above 7k this month?", 0.47, 650_000),
        ]
        ks = [
            _signal("kalshi", "ks1", "Will BTC close above 120k this month?", 0.49, 600_000),
            _signal("kalshi", "ks2", "Will ETH close above 7000 this month?", 0.45, 550_000),
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10)
        self.assertEqual(len(matches), 2)
        kalshi_ids = {m.kalshi.market_id for m in matches}
        self.assertEqual(kalshi_ids, {"ks1", "ks2"})

    def test_fed_directional_markets_do_not_cross_match_hike_and_cut(self) -> None:
        pm = [
            _signal(
                "polymarket",
                "pm-cut",
                "Will the Fed decrease interest rates by 25 bps after the March 2026 meeting?",
                0.62,
                900_000,
            ),
            _signal(
                "polymarket",
                "pm-hike",
                "Will the Fed increase interest rates by 25+ bps after the March 2026 meeting?",
                0.08,
                850_000,
            ),
        ]
        ks = [
            _signal("kalshi", "ks-hike", "Fed decision in March? - Hike 25bps", 0.06, 820_000),
            _signal("kalshi", "ks-cut", "Fed decision in March? - Cut 25bps", 0.59, 810_000),
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10)
        self.assertEqual(len(matches), 2)
        pair_map = {m.polymarket.market_id: m.kalshi.market_id for m in matches}
        self.assertEqual(pair_map.get("pm-cut"), "ks-cut")
        self.assertEqual(pair_map.get("pm-hike"), "ks-hike")

    def test_treats_near_equivalent_deadline_windows_as_match(self) -> None:
        pm = [
            _signal(
                "polymarket",
                "pm-khamenei",
                "Will Ali Khamenei be out as Supreme Leader of Iran by Feb 28, 2026?",
                0.21,
                500_000,
                raw={"startDate": "2026-02-01T00:00:00Z", "endDate": "2026-02-28T23:59:59Z"},
            )
        ]
        ks = [
            _signal(
                "kalshi",
                "ks-khamenei",
                "Will Ali Khamenei be out as Supreme Leader of Iran before March 1, 2026?",
                0.23,
                400_000,
                raw={"open_ts": "2026-02-01T00:00:00Z", "close_time": "2026-03-01T00:00:00Z"},
            )
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10, use_llm_verifier=False)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].polymarket.market_id, "pm-khamenei")
        self.assertEqual(matches[0].kalshi.market_id, "ks-khamenei")

    def test_rejects_related_but_not_same_resolution_window(self) -> None:
        pm = [
            _signal(
                "polymarket",
                "pm-btc-feb",
                "Will Bitcoin hit $150,000 in February 2026?",
                0.17,
                450_000,
                raw={"startDate": "2026-02-01T00:00:00Z", "endDate": "2026-02-28T23:59:59Z"},
            )
        ]
        ks = [
            _signal(
                "kalshi",
                "ks-btc-broad",
                "When will Bitcoin hit $150k? - Before March 2026",
                0.25,
                430_000,
                raw={"open_ts": "2025-12-01T00:00:00Z", "close_time": "2026-03-01T00:00:00Z"},
            )
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10, use_llm_verifier=False)
        self.assertEqual(matches, [])

    def test_rejects_clemency_vs_leadership_false_match(self) -> None:
        pm = [
            _signal(
                "polymarket",
                "pm-pardon",
                "Will Trump pardon Ghislaine Maxwell by end of 2026?",
                0.08,
                300_000,
                raw={"endDate": "2026-12-31T00:00:00Z"},
            )
        ]
        ks = [
            _signal(
                "kalshi",
                "ks-venezuela",
                "Who will lead Venezuela at the end of 2026? - Donald Trump",
                0.04,
                290_000,
                raw={"close_time": "2026-12-31T00:00:00Z"},
            )
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10, use_llm_verifier=False)
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
