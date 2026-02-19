from __future__ import annotations

import unittest

from prediction_agent.utils.cross_venue_telegram import format_cross_venue_telegram_card, html_link


class CrossVenueTelegramFormatTests(unittest.TestCase):
    def test_heuristic_card_contains_links_and_similarity(self) -> None:
        row = {
            "pm_yes": 0.12,
            "pm_no": 0.88,
            "ka_yes": 0.19,
            "ka_no": 0.82,
            "pm_question": "Will CPI be above 3%?",
            "ks_question": "Will CPI exceed 3%?",
            "pm_link": "https://polymarket.com/event/cpi",
            "ks_link": "https://kalshi.com/markets/cpi",
            "liquidity_sum": 7_550_182.0,
            "prob_diff_pp": 6.0,
            "sim": "0.42",
            "edge_hint": "yes_pm/no_kalshi",
            "arb_flag": "yes",
            "arb_pnl": 33.47,
        }

        card = format_cross_venue_telegram_card(rank=1, row=row)

        self.assertIn('href="https://polymarket.com/event/cpi"', card)
        self.assertIn('href="https://kalshi.com/markets/cpi"', card)
        self.assertIn("<b>Sim:</b> 0.42", card)

    def test_llm_card_uses_na_similarity(self) -> None:
        row = {
            "pm_yes": 0.45,
            "pm_no": 0.55,
            "ka_yes": 0.48,
            "ka_no": 0.52,
            "pm_question": "Will X happen?",
            "ks_question": "Will X happen on Kalshi?",
            "pm_link": "https://polymarket.com/event/x",
            "ks_link": "https://kalshi.com/markets/x",
            "edge_hint": "aligned",
            "arb_flag": "no",
            "arb_pnl": -1.0,
            "liquidity_sum": 1000.0,
            "prob_diff_pp": 3.0,
            "sim": "n/a",
        }

        card = format_cross_venue_telegram_card(rank=2, row=row)
        self.assertIn("<b>Sim:</b> n/a", card)

    def test_html_link_rejects_non_http_scheme(self) -> None:
        linked = html_link("Click", "javascript:alert(1)")
        self.assertEqual(linked, "Click")


if __name__ == "__main__":
    unittest.main()
