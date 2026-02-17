from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
