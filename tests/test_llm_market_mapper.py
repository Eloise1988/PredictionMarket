from __future__ import annotations

import unittest

try:
    from prediction_agent.clients.llm_market_mapper import (
        _parse_cross_venue_matches,
        _parse_cross_venue_matches_from_lists,
    )
except ModuleNotFoundError:  # pragma: no cover - allows running tests outside project venv
    _parse_cross_venue_matches = None
    _parse_cross_venue_matches_from_lists = None


@unittest.skipIf(_parse_cross_venue_matches is None, "openai dependency not available in this interpreter")
class LLMCrossVenueStrongMatchParserTests(unittest.TestCase):
    def test_filters_to_allowed_pairs(self) -> None:
        payload = {
            "matches": [
                {"polymarket_id": "pm-1", "kalshi_id": "ka-1"},
                {"polymarket_id": "pm-2", "kalshi_id": "ka-2"},
                {"polymarket_id": "pm-x", "kalshi_id": "ka-y"},
            ]
        }
        allowed_pairs = {("pm-1", "ka-1"), ("pm-2", "ka-2")}

        out = _parse_cross_venue_matches(payload, allowed_pairs=allowed_pairs, max_matches=10)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].polymarket_id, "pm-1")
        self.assertEqual(out[0].kalshi_id, "ka-1")

    def test_enforces_one_to_one(self) -> None:
        payload = {
            "matches": [
                {"polymarket_id": "pm-1", "kalshi_id": "ka-1"},
                {"polymarket_id": "pm-1", "kalshi_id": "ka-2"},
                {"polymarket_id": "pm-2", "kalshi_id": "ka-1"},
            ]
        }
        allowed_pairs = {("pm-1", "ka-1"), ("pm-1", "ka-2"), ("pm-2", "ka-1")}

        out = _parse_cross_venue_matches(payload, allowed_pairs=allowed_pairs, max_matches=10)
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0].polymarket_id, out[0].kalshi_id), ("pm-1", "ka-1"))

    def test_list_mode_filters_to_allowed_ids(self) -> None:
        payload = {
            "matches": [
                {"polymarket_id": "pm-1", "kalshi_id": "ka-1"},
                {"polymarket_id": "pm-x", "kalshi_id": "ka-1"},
                {"polymarket_id": "pm-1", "kalshi_id": "ka-x"},
                {"polymarket_id": "pm-2", "kalshi_id": "ka-2"},
            ]
        }
        out = _parse_cross_venue_matches_from_lists(
            payload,
            allowed_polymarket_ids={"pm-1", "pm-2"},
            allowed_kalshi_ids={"ka-1", "ka-2"},
            max_matches=10,
        )
        self.assertEqual(len(out), 2)
        self.assertEqual((out[0].polymarket_id, out[0].kalshi_id), ("pm-1", "ka-1"))


if __name__ == "__main__":
    unittest.main()
