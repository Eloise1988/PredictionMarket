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
    def __init__(self, events, wrapper_key: str = "data") -> None:
        self.calls = []
        self.events = list(events)
        self.wrapper_key = wrapper_key

    def get_json(self, url: str, params=None, headers=None):
        params = params or {}
        call = {"url": url}
        if isinstance(params, dict):
            items = list(params.items())
        else:
            items = list(params)
        for key, value in items:
            if key in call:
                existing = call[key]
                if isinstance(existing, list):
                    existing.append(value)
                else:
                    call[key] = [existing, value]
            else:
                call[key] = value
        self.calls.append(call)

        if url.endswith("/price"):
            return {"price": "0.50"}

        limit = int(_last_param(call.get("limit"), len(self.events)))
        offset = int(_last_param(call.get("offset"), 0))
        if url.endswith("/events/pagination"):
            return {self.wrapper_key: self.events[offset : offset + limit]}
        return []


def _last_param(value, default):
    if isinstance(value, list):
        return value[-1] if value else default
    if value is None:
        return default
    return value


def _make_event_row(idx: int, updated_at: str = "2026-02-17T00:00:00Z", liquidity: float = 150000.0) -> dict:
    return {
        "id": f"event-{idx}",
        "slug": f"event-{idx}",
        "title": f"CPI event {idx}",
        "category": "Economy",
        "active": True,
        "closed": False,
        "archived": False,
        "updatedAt": updated_at,
        "markets": [
            {
                "id": str(idx),
                "question": f"Will CPI print above 3% on release {idx}?",
                "outcomes": ["Yes", "No"],
                "outcomePrices": [0.75, 0.25],
                "liquidityNum": liquidity,
                "volume24hr": 8000,
                "updatedAt": updated_at,
                "slug": f"market-{idx}",
                "active": True,
                "closed": False,
                "archived": False,
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
        fake_http = FakeHttp([_make_event_row(i) for i in range(1, 1201)])
        connector.http = fake_http  # type: ignore[assignment]

        signals = connector.fetch_signals()

        self.assertEqual(len(signals), 1200)
        self.assertEqual([c.get("offset", 0) for c in fake_http.calls], [0, 1000])
        self.assertEqual([c["limit"] for c in fake_http.calls], [1000, 200])
        self.assertEqual([c["active"] for c in fake_http.calls], ["true", "true"])
        self.assertEqual([c["closed"] for c in fake_http.calls], ["false", "false"])
        self.assertEqual([c["archived"] for c in fake_http.calls], ["false", "false"])
        self.assertEqual([c["ascending"] for c in fake_http.calls], ["false", "false"])
        self.assertEqual([c["order"] for c in fake_http.calls], ["volume", "volume"])
        self.assertTrue(all(c["url"].endswith("/events/pagination") for c in fake_http.calls))

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
                    "id": "4690",
                    "slug": "will-joe-biden-get-coronavirus-before-the-election",
                    "ticker": "will-joe-biden-get-coronavirus-before-the-election",
                    "title": "Will Joe Biden get Coronavirus before the election?",
                    "category": "Politics",
                    "active": True,
                    "closed": False,
                    "archived": False,
                    "updatedAt": "2024-04-23T00:49:51.620233Z",
                    "markets": [
                        {
                            "id": "12",
                            "conditionId": "0xe3b423dfad8c22ff75c9899c4e8176f628cf4ad4caa00481764d320e7415f7a9",
                            "question": "Will Joe Biden get Coronavirus before the election?",
                            "ticker": "will-joe-biden-get-coronavirus-before-the-election",
                            "slug": "will-joe-biden-get-coronavirus-before-the-election-market",
                            "category": "US-current-affairs",
                            "outcomes": "[\"Yes\", \"No\"]",
                            "outcomePrices": "[\"0.61\", \"0.39\"]",
                            "liquidity": "12345.67",
                            "volume24hr": "555.2",
                            "updatedAt": "2024-04-23T00:49:51.620233Z",
                            "active": True,
                            "closed": False,
                            "archived": False,
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
        self.assertEqual(
            signal.url,
            "https://polymarket.com/event/will-joe-biden-get-coronavirus-before-the-election/will-joe-biden-get-coronavirus-before-the-election-market",
        )
        self.assertAlmostEqual(signal.prob_yes, 0.61, places=6)
        self.assertEqual(signal.raw.get("eventSlug"), "will-joe-biden-get-coronavirus-before-the-election")
        self.assertEqual(signal.raw.get("eventTitle"), "Will Joe Biden get Coronavirus before the election?")
        self.assertEqual(signal.raw.get("eventCategory"), "Politics")
        self.assertEqual(signal.raw.get("eventTicker"), "will-joe-biden-get-coronavirus-before-the-election")
        self.assertEqual(signal.raw.get("category"), "US-current-affairs")
        self.assertAlmostEqual(float(signal.raw.get("yes_price")), 0.61, places=6)
        self.assertAlmostEqual(float(signal.raw.get("no_price")), 0.39, places=6)

    def test_skips_closed_or_archived_rows_even_if_api_returns_them(self) -> None:
        connector = PolymarketConnector(
            gamma_base_url="https://gamma-api.polymarket.com",
            clob_base_url="https://clob.polymarket.com",
            limit=10,
            timeout=1,
        )
        fake_http = FakeHttp(
            [
                {
                    "id": "event-open",
                    "slug": "event-open",
                    "title": "Open market",
                    "active": True,
                    "closed": False,
                    "archived": False,
                    "markets": [
                        {
                            "id": "open-1",
                            "question": "Open market",
                            "active": True,
                            "closed": False,
                            "archived": False,
                            "outcomes": ["Yes", "No"],
                            "outcomePrices": [0.52, 0.48],
                            "slug": "open-1",
                        }
                    ],
                },
                {
                    "id": "event-closed",
                    "slug": "event-closed",
                    "title": "Closed event market",
                    "active": True,
                    "closed": True,
                    "archived": False,
                    "markets": [
                        {
                            "id": "closed-1",
                            "question": "Closed market",
                            "active": True,
                            "closed": False,
                            "archived": False,
                            "outcomes": ["Yes", "No"],
                            "outcomePrices": [0.40, 0.60],
                            "slug": "closed-1",
                        }
                    ],
                },
                {
                    "id": "event-open-closed-market",
                    "slug": "event-open-closed-market",
                    "title": "Open event closed market",
                    "active": True,
                    "closed": False,
                    "archived": False,
                    "markets": [
                        {
                            "id": "closed-2",
                            "question": "Closed market 2",
                            "active": True,
                            "closed": True,
                            "archived": False,
                            "outcomes": ["Yes", "No"],
                            "outcomePrices": [0.40, 0.60],
                            "slug": "closed-2",
                        }
                    ],
                },
            ]
        )
        connector.http = fake_http  # type: ignore[assignment]

        signals = connector.fetch_signals()

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].market_id, "open-1")

    def test_sorts_by_date_desc_then_liquidity_desc(self) -> None:
        connector = PolymarketConnector(
            gamma_base_url="https://gamma-api.polymarket.com",
            clob_base_url="https://clob.polymarket.com",
            limit=10,
            timeout=1,
        )
        fake_http = FakeHttp(
            [
                _make_event_row(idx=1, updated_at="2026-02-10T00:00:00Z", liquidity=100.0),
                _make_event_row(idx=2, updated_at="2026-02-12T00:00:00Z", liquidity=50.0),
                _make_event_row(idx=3, updated_at="2026-02-12T00:00:00Z", liquidity=200.0),
            ]
        )
        connector.http = fake_http  # type: ignore[assignment]

        signals = connector.fetch_signals()

        self.assertEqual([s.market_id for s in signals], ["3", "2", "1"])

    def test_supports_data_wrapper_shape_from_events_pagination(self) -> None:
        connector = PolymarketConnector(
            gamma_base_url="https://gamma-api.polymarket.com",
            clob_base_url="https://clob.polymarket.com",
            limit=2,
            timeout=1,
        )
        fake_http = FakeHttp(
            [
                _make_event_row(idx=11, updated_at="2026-02-18T17:28:16.838601Z", liquidity=2000.0),
                _make_event_row(idx=12, updated_at="2026-02-17T10:00:00Z", liquidity=1000.0),
            ],
            wrapper_key="data",
        )
        connector.http = fake_http  # type: ignore[assignment]

        signals = connector.fetch_signals()

        self.assertEqual(len(signals), 2)
        self.assertEqual([s.market_id for s in signals], ["11", "12"])


if __name__ == "__main__":
    unittest.main()
