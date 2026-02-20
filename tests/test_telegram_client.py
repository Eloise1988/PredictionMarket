from __future__ import annotations

import sys
import types
import unittest

# Keep Telegram client tests independent from optional third-party installs.
if "requests" not in sys.modules:
    sys.modules["requests"] = types.SimpleNamespace(Session=lambda: None, HTTPError=Exception)
if "tenacity" not in sys.modules:
    sys.modules["tenacity"] = types.SimpleNamespace(
        retry=lambda *args, **kwargs: (lambda f: f),
        stop_after_attempt=lambda *args, **kwargs: None,
        wait_exponential=lambda *args, **kwargs: None,
    )

from prediction_agent.clients.telegram import TelegramClient
from prediction_agent.utils.text import split_for_telegram


class TelegramSplitTests(unittest.TestCase):
    def test_short_message_kept_as_single_chunk(self) -> None:
        msg = "hello"
        chunks = split_for_telegram(msg)
        self.assertEqual(chunks, [msg])

    def test_long_message_is_split(self) -> None:
        msg = "a" * 9000
        chunks = split_for_telegram(msg)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), msg)
        self.assertTrue(all(len(c) <= 4096 for c in chunks))


class _FakeHttp:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {"ok": True}
        self.calls: list[dict] = []

    def post_json(self, url: str, json_body: dict, headers=None):  # noqa: ANN001
        self.calls.append({"url": url, "json_body": json_body, "headers": headers})
        return self.payload

    def get_json(self, url: str, params=None, headers=None):  # noqa: ANN001
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.payload


class TelegramClientTests(unittest.TestCase):
    def test_send_message_passes_parse_mode_and_preview_flag(self) -> None:
        client = TelegramClient(bot_token="abc123", chat_id="42")
        fake = _FakeHttp()
        client.http = fake  # type: ignore[assignment]

        sent = client.send_message("hello", parse_mode="HTML", disable_web_page_preview=False)

        self.assertTrue(sent)
        self.assertEqual(len(fake.calls), 1)
        body = fake.calls[0]["json_body"]
        self.assertEqual(body.get("parse_mode"), "HTML")
        self.assertEqual(body.get("disable_web_page_preview"), False)

    def test_fetch_updates_returns_rows_and_sets_offset(self) -> None:
        client = TelegramClient(bot_token="abc123", chat_id="42")
        fake = _FakeHttp(payload={"ok": True, "result": [{"update_id": 11}, {"update_id": 12}]})
        client.http = fake  # type: ignore[assignment]

        rows = client.fetch_updates(offset=99, timeout_seconds=15)

        self.assertEqual(len(rows), 2)
        self.assertEqual(fake.calls[0]["params"]["offset"], 99)
        self.assertEqual(fake.calls[0]["params"]["timeout"], 15)


if __name__ == "__main__":
    unittest.main()
