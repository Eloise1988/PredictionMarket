from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from prediction_agent.engine.theme_matcher import ThemeMatcher


class ThemeMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matcher = ThemeMatcher(Path("prediction_agent/knowledge/theme_map.json"))

    def _signal(self, question: str):
        return SimpleNamespace(question=question, prob_yes=0.5)

    def test_sports_market_is_filtered(self) -> None:
        s = self._signal("Will Iran win the 2026 FIFA World Cup?")
        themes = [theme for theme, _ in self.matcher.match(s, strict=True)]
        self.assertNotIn("us_iran_conflict", themes)

    def test_us_iran_conflict_requires_context(self) -> None:
        s = self._signal("Will the U.S. go to war with Iran before 2027?")
        themes = [theme for theme, _ in self.matcher.match(s, strict=True)]
        self.assertIn("us_iran_conflict", themes)

    def test_fed_not_matched_from_federal_substring(self) -> None:
        s = self._signal("Will federal spending fall below $50B this year?")
        themes = [theme for theme, _ in self.matcher.match(s, strict=True)]
        self.assertNotIn("fed_rate_cut_cycle", themes)

    def test_actual_fed_market_matches(self) -> None:
        s = self._signal("Will the Fed cut interest rates by 25bps at the next FOMC?")
        themes = [theme for theme, _ in self.matcher.match(s, strict=True)]
        self.assertIn("fed_rate_cut_cycle", themes)


if __name__ == "__main__":
    unittest.main()
