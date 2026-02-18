from __future__ import annotations

import sys
import types
import unittest

# Keep connector tests independent from optional third-party installs.
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace(Session=lambda: None, HTTPError=Exception)
if "tenacity" not in sys.modules:
    sys.modules["tenacity"] = types.SimpleNamespace(
        retry=lambda *args, **kwargs: (lambda f: f),
        stop_after_attempt=lambda *args, **kwargs: None,
        wait_exponential=lambda *args, **kwargs: None,
    )
if "pydantic" not in sys.modules:
    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _field(default=None, **kwargs):
        return default

    sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=_BaseModel, Field=_field)

from prediction_agent.connectors.kalshi import KalshiConnector


class FakeHttp:
    def __init__(self, pages) -> None:
        self.pages = pages
        self.calls = []

    def get_json(self, url: str, params=None, headers=None):
        params = dict(params or {})
        self.calls.append({"url": url, **params})
        cursor = params.get("cursor")
        return self.pages.get(cursor, {"current_page": []})


def _series_row(event_ticker: str, markets: list[dict]) -> dict:
    return {
        "series_ticker": "KXFEDCHAIRNOM",
        "series_title": "Fed Chair nominee",
        "event_ticker": event_ticker,
        "event_subtitle": "During Trump's term",
        "event_title": "Who will Trump nominate as Fed Chair?",
        "category": "Politics",
        "fee_type": "quadratic",
        "fee_multiplier": 1,
        "product_metadata": {
            "categories": ["Politics"],
            "subcategories": {"Politics": ["Trump"]},
        },
        "total_series_volume": 174363035,
        "total_volume": 174363035,
        "markets": markets,
    }


class KalshiConnectorTests(unittest.TestCase):
    def test_fetch_signals_uses_search_series_endpoint_and_cursor(self) -> None:
        connector = KalshiConnector(
            base_url="https://api.elections.kalshi.com/trade-api/v2",
            limit=3,
            timeout=1,
        )
        fake_http = FakeHttp(
            {
                None: {
                    "current_page": [
                        _series_row(
                            "KXFEDCHAIRNOM-29",
                            [
                                {"ticker": "KXFEDCHAIRNOM-29-KW", "yes_bid": 94, "volume": 1000, "yes_subtitle": "Kevin Warsh"},
                                {"ticker": "KXFEDCHAIRNOM-29-JS", "yes_bid": 3, "volume": 800, "yes_subtitle": "Judy Shelton"},
                            ],
                        )
                    ],
                    "next_cursor": "cursor-2",
                },
                "cursor-2": {
                    "current_page": [
                        _series_row(
                            "KXFEDCHAIRNOM-30",
                            [{"ticker": "KXFEDCHAIRNOM-30-AB", "yes_bid": 55, "volume": 700, "yes_subtitle": "Analyst B"}],
                        )
                    ],
                    "next_cursor": None,
                },
            }
        )
        connector.http = fake_http  # type: ignore[assignment]

        signals = connector.fetch_signals()

        self.assertEqual(len(signals), 3)
        self.assertEqual(
            [c["url"] for c in fake_http.calls],
            [
                "https://api.elections.kalshi.com/v1/search/series",
                "https://api.elections.kalshi.com/v1/search/series",
            ],
        )
        self.assertEqual(fake_http.calls[0]["order_by"], "event-volume")
        self.assertEqual(fake_http.calls[0]["status"], "open,unopened")
        self.assertEqual(fake_http.calls[0]["reverse"], "false")
        self.assertEqual(fake_http.calls[0]["with_milestones"], "true")
        self.assertEqual(fake_http.calls[0]["hydrate"], "milestones,structured_targets")
        self.assertEqual(fake_http.calls[1]["cursor"], "cursor-2")
        self.assertEqual([s.market_id for s in signals], ["KXFEDCHAIRNOM-29-KW", "KXFEDCHAIRNOM-29-JS", "KXFEDCHAIRNOM-30-AB"])

    def test_normalizes_question_probability_and_raw_metadata(self) -> None:
        connector = KalshiConnector(
            base_url="https://api.elections.kalshi.com/v1",
            limit=1,
            timeout=1,
        )
        fake_http = FakeHttp(
            {
                None: {
                    "current_page": [
                        _series_row(
                            "KXFEDCHAIRNOM-29",
                            [
                                {
                                    "ticker": "KXFEDCHAIRNOM-29-KW",
                                    "last_price_dollars": "0.9500",
                                    "volume": 61307746,
                                    "yes_subtitle": "Kevin Warsh",
                                    "no_subtitle": "",
                                }
                            ],
                        )
                    ]
                }
            }
        )
        connector.http = fake_http  # type: ignore[assignment]

        signals = connector.fetch_signals()

        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.question, "Who will Trump nominate as Fed Chair? - Kevin Warsh")
        self.assertAlmostEqual(signal.prob_yes, 0.95, places=6)
        self.assertEqual(
            signal.url,
            "https://kalshi.com/markets/kxfedchairnom/who-will-trump-nominate-as-fed-chair/kxfedchairnom-29-kw",
        )
        self.assertEqual(signal.raw.get("event_ticker"), "KXFEDCHAIRNOM-29")
        self.assertEqual(signal.raw.get("series_ticker"), "KXFEDCHAIRNOM")
        self.assertEqual(signal.raw.get("category"), "Politics")
        self.assertEqual(signal.raw.get("yes_sub_title"), "Kevin Warsh")
        self.assertEqual(signal.raw.get("fee_type"), "quadratic")
        self.assertEqual(signal.raw.get("fee_multiplier"), 1)
        self.assertAlmostEqual(float(signal.raw.get("yes_price")), 0.95, places=6)
        self.assertAlmostEqual(float(signal.raw.get("no_price")), 0.05, places=6)
        tags = signal.raw.get("tags", [])
        self.assertIn("Politics", tags)
        self.assertIn("Trump", tags)

    def test_skips_markets_without_probability(self) -> None:
        connector = KalshiConnector(
            base_url="https://api.elections.kalshi.com/v1",
            limit=10,
            timeout=1,
        )
        fake_http = FakeHttp(
            {
                None: {
                    "current_page": [
                        _series_row(
                            "KXFOO-26",
                            [
                                {"ticker": "KXFOO-26-A", "volume": 1000, "yes_subtitle": "A"},
                                {"ticker": "KXFOO-26-B", "yes_bid": 61, "volume": 1000, "yes_subtitle": "B"},
                            ],
                        )
                    ]
                }
            }
        )
        connector.http = fake_http  # type: ignore[assignment]

        signals = connector.fetch_signals()

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].market_id, "KXFOO-26-B")


if __name__ == "__main__":
    unittest.main()
