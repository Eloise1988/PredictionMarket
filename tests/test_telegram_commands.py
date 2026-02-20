from __future__ import annotations

import unittest

from prediction_agent.utils.telegram_commands import (
    extract_update_chat_id,
    extract_update_id,
    extract_update_text,
    is_arb_command,
    is_help_command,
)


class TelegramCommandParserTests(unittest.TestCase):
    def test_arb_command_variants(self) -> None:
        self.assertTrue(is_arb_command("/arb"))
        self.assertTrue(is_arb_command("/arb toupdate"))
        self.assertTrue(is_arb_command("/arb@my_bot now"))
        self.assertFalse(is_arb_command("/start"))
        self.assertFalse(is_arb_command("arb"))

    def test_help_commands(self) -> None:
        self.assertTrue(is_help_command("/help"))
        self.assertTrue(is_help_command("/start"))
        self.assertTrue(is_help_command("/help@my_bot"))
        self.assertFalse(is_help_command("/arb"))

    def test_extractors(self) -> None:
        update = {
            "update_id": 123,
            "message": {
                "text": "/arb toupdate",
                "chat": {"id": 987654321},
            },
        }
        self.assertEqual(extract_update_id(update), 123)
        self.assertEqual(extract_update_text(update), "/arb toupdate")
        self.assertEqual(extract_update_chat_id(update), "987654321")


if __name__ == "__main__":
    unittest.main()
