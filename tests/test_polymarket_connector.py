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
    def __init__(self) -> None:
        self.calls = []

    def get_json(self, url: str, params=None, headers=None):
        params = params or {}
        self.calls.append(dict(params))
        limit = int(params.get("limit", 0))
        offset = int(params.get("offset", 0))

        rows = []
        start = offset + 1
        stop = offset + limit + 1
        for i in range(start, stop):
            rows.append(
                {
                    "id": str(i),
                    "question": f"Will CPI print above 3% on release {i}?",
                    "outcomes": ["Yes", "No"],
                    "outcomePrices": [0.75, 0.25],
                    "liquidityNum": 150000,
                    "volume24hr": 8000,
                    "updatedAt": "2026-02-17T00:00:00Z",
                    "eventSlug": f"cpi-{i}",
                }
            )
        return rows


class PolymarketConnectorPaginationTests(unittest.TestCase):
    def test_fetch_signals_pages_until_limit(self) -> None:
        connector = PolymarketConnector(
            gamma_base_url="https://gamma-api.polymarket.com",
            clob_base_url="https://clob.polymarket.com",
            limit=450,
            timeout=1,
        )
        fake_http = FakeHttp()
        connector.http = fake_http  # type: ignore[assignment]

        signals = connector.fetch_signals()

        self.assertEqual(len(signals), 450)
        self.assertEqual([c["offset"] for c in fake_http.calls], [0, 200, 400])
        self.assertEqual([c["limit"] for c in fake_http.calls], [200, 200, 50])


if __name__ == "__main__":
    unittest.main()
