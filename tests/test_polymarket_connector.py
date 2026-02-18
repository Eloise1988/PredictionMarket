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

from prediction_agent.connectors.polymarket import PolymarketConnector


class FakeHttp:
    def __init__(self, rows) -> None:
        self.calls = []
        self.rows = list(rows)

    def get_json(self, url: str, params=None, headers=None):
        params = params or {}
        call = {"url": url}
        call.update(dict(params))
        self.calls.append(call)

        if url.endswith("/price"):
            return {"price": "0.50"}

        limit = int(params.get("limit", len(self.rows)))
        offset = int(params.get("offset", 0))
        return self.rows[offset : offset + limit]


def _make_market_row(idx: int) -> dict:
    return {
        "id": str(idx),
        "question": f"Will CPI print above 3% on release {idx}?",
        "outcomes": ["Yes", "No"],
        "outcomePrices": [0.75, 0.25],
        "liquidityNum": 150000,
        "volume24hr": 8000,
        "updatedAt": "2026-02-17T00:00:00Z",
        "slug": f"market-{idx}",
        "events": [
            {
                "id": f"event-{idx}",
                "slug": f"event-{idx}",
                "title": f"CPI event {idx}",
                "category": "Economy",
            }
        ],
    }


class PolymarketConnectorTests(unittest.TestCase):
    def test_fetch_signals_pages_until_limit(self) -> None:
        connector = PolymarketConnector(
            gamma_base_url="https://gamma-api.polymarket.com",
            clob_base_url="https://clob.polymarket.com",
            limit=1200,
            timeout=1,
        )
        fake_http = FakeHttp([_make_market_row(i) for i in range(1, 1201)])
        connector.http = fake_http  # type: ignore[assignment]

        signals = connector.fetch_signals()

        self.assertEqual(len(signals), 1200)
        self.assertEqual([c.get("offset", 0) for c in fake_http.calls], [0, 1000])
        self.assertEqual([c["limit"] for c in fake_http.calls], [1000, 200])
        self.assertEqual([c["active"] for c in fake_http.calls], [True, True])
        self.assertTrue(all("closed" not in c for c in fake_http.calls))

    def test_normalizes_event_metadata_from_gamma_payload(self) -> None:
        connector = PolymarketConnector(
            gamma_base_url="https://gamma-api.polymarket.com",
            clob_base_url="https://clob.polymarket.com",
            limit=1,
            timeout=1,
        )
        fake_http = FakeHttp(
            [
                {
                    "id": "12",
                    "conditionId": "0xe3b423dfad8c22ff75c9899c4e8176f628cf4ad4caa00481764d320e7415f7a9",
                    "question": "Will Joe Biden get Coronavirus before the election?",
                    "slug": "will-joe-biden-get-coronavirus-before-the-election-market",
                    "category": "US-current-affairs",
                    "outcomes": "[\"Yes\", \"No\"]",
                    "outcomePrices": "[\"0.61\", \"0.39\"]",
                    "liquidity": "12345.67",
                    "volume24hr": "555.2",
                    "updatedAt": "2024-04-23T00:49:51.620233Z",
                    "events": [
                        {
                            "id": "4690",
                            "ticker": "will-joe-biden-get-coronavirus-before-the-election",
                            "slug": "will-joe-biden-get-coronavirus-before-the-election",
                            "title": "Will Joe Biden get Coronavirus before the election?",
                            "category": "Politics",
                        }
                    ],
                }
            ]
        )
        connector.http = fake_http  # type: ignore[assignment]

        signals = connector.fetch_signals()

        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.market_id, "12")
        self.assertEqual(signal.url, "https://polymarket.com/event/will-joe-biden-get-coronavirus-before-the-election")
        self.assertAlmostEqual(signal.prob_yes, 0.61, places=6)
        self.assertEqual(signal.raw.get("eventSlug"), "will-joe-biden-get-coronavirus-before-the-election")
        self.assertEqual(signal.raw.get("eventTitle"), "Will Joe Biden get Coronavirus before the election?")
        self.assertEqual(signal.raw.get("eventCategory"), "Politics")
        self.assertEqual(signal.raw.get("eventTicker"), "will-joe-biden-get-coronavirus-before-the-election")
        self.assertEqual(signal.raw.get("category"), "US-current-affairs")


if __name__ == "__main__":
    unittest.main()
