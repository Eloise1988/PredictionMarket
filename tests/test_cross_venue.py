from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from prediction_agent.engine.cross_venue import match_cross_venue_markets


def _signal(
    source: str,
    market_id: str,
    question: str,
    prob_yes: float,
    liq: float,
    raw: dict | None = None,
):
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

    def test_does_not_require_category_compatibility(self) -> None:
        pm = [
            _signal(
                "polymarket",
                "pm1",
                "Will Bitcoin reach $100,000 before March 2026?",
                0.62,
                500_000,
                raw={"category": "politics", "endDate": "2026-03-01T00:00:00Z"},
            ),
        ]
        ks = [
            _signal(
                "kalshi",
                "ks1",
                "When will BTC cross $100k? - Before March 2026",
                0.55,
                250_000,
                raw={"category": "finance", "close_time": "2026-03-01T00:00:00Z"},
            ),
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10)
        self.assertEqual(len(matches), 1)

    def test_treats_end_of_february_as_equivalent_to_before_march_first(self) -> None:
        pm = [
            _signal(
                "polymarket",
                "pm1",
                "Ali Khamenei out as Supreme Leader of Iran by February 28, 2026?",
                0.25,
                500_000,
                raw={"endDate": "2026-02-28T00:00:00Z"},
            ),
        ]
        ks = [
            _signal(
                "kalshi",
                "ks1",
                "Ali Khamenei out as Supreme Leader? - Before March 1, 2026",
                0.28,
                250_000,
                raw={"close_time": "2026-03-01T00:00:00Z"},
            ),
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10)
        self.assertEqual(len(matches), 1)

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

    def test_rejects_politics_vs_earnings_false_positive(self) -> None:
        pm = [
            _signal(
                "polymarket",
                "pm-trump-words",
                'Will Trump say "Jesse" or "Jackson" during the Black History Month reception?',
                0.96,
                900_000,
                raw={"category": "politics"},
            ),
        ]
        ks = [
            _signal(
                "kalshi",
                "KXEARNINGSMENTIONEA-25OCT28-MOBL",
                "What will EA say during their next earnings call? - Mobile",
                0.03,
                850_000,
                raw={"category": "finance", "close_time": "2025-10-28T00:00:00Z"},
            ),
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10)
        self.assertEqual(matches, [])

    def test_rejects_fed_cut_vs_hold_and_strike_mismatch(self) -> None:
        pm = [
            _signal(
                "polymarket",
                "pm-cut-25",
                "Will the Fed decrease interest rates by 25 bps after the March 2026 meeting?",
                0.62,
                900_000,
                raw={"category": "finance", "endDate": "2026-03-31T00:00:00Z"},
            ),
        ]
        ks = [
            _signal(
                "kalshi",
                "KXFEDDECISION-26MAR-H0",
                "Fed decision in March? - Fed maintains rate",
                0.55,
                850_000,
                raw={"category": "finance", "close_time": "2026-03-31T00:00:00Z"},
            ),
            _signal(
                "kalshi",
                "KXFEDDECISION-26MAR-C26",
                "Fed decision in March? - Cut >25bps",
                0.10,
                820_000,
                raw={"category": "finance", "close_time": "2026-03-31T00:00:00Z"},
            ),
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10)
        self.assertEqual(matches, [])

    def test_rejects_crypto_threshold_or_expiry_mismatch(self) -> None:
        pm = [
            _signal(
                "polymarket",
                "pm-btc-80k-feb",
                "Will Bitcoin reach $80,000 in February 2026?",
                0.10,
                700_000,
                raw={"category": "finance", "endDate": "2026-02-28T00:00:00Z"},
            ),
        ]
        ks = [
            _signal(
                "kalshi",
                "KXBTCMAX100-26-JUNE",
                "When will Bitcoin cross $100k again? - Before July 2026",
                0.22,
                650_000,
                raw={"category": "finance", "close_time": "2026-07-01T00:00:00Z"},
            ),
        ]

        matches = match_cross_venue_markets(pm, ks, min_similarity=0.10)
        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
