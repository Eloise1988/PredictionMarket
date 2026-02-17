from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from prediction_agent.engine.cross_venue import match_cross_venue_markets


def _signal(source: str, market_id: str, question: str, prob_yes: float, liq: float):
    return SimpleNamespace(
        source=source,
        market_id=market_id,
        question=question,
        prob_yes=prob_yes,
        liquidity=liq,
        volume_24h=liq / 10.0,
        updated_at=datetime(2026, 2, 17, 0, 0, tzinfo=timezone.utc),
        url="",
        raw={},
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


if __name__ == "__main__":
    unittest.main()
