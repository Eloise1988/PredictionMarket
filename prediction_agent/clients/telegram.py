from __future__ import annotations

import logging

from prediction_agent.clients.http_client import HttpClient
from prediction_agent.utils.text import split_for_telegram

logger = logging.getLogger(__name__)


class TelegramClient:
    def __init__(self, bot_token: str, chat_id: str, timeout: int = 15):
        self.bot_token = bot_token.strip()
        self.chat_id = str(chat_id).strip()
        self.http = HttpClient(timeout=timeout)

    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def fetch_updates(self, offset: int | None = None, timeout_seconds: int = 20) -> list[dict]:
        if not self.enabled():
            return []

        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params: dict[str, object] = {"timeout": max(0, int(timeout_seconds))}
        if offset is not None:
            params["offset"] = int(offset)

        try:
            payload = self.http.get_json(url, params=params)
        except Exception as exc:
            logger.warning("Telegram getUpdates failed", extra={"error": str(exc)})
            return []

        if not (isinstance(payload, dict) and payload.get("ok") is True):
            logger.warning("Telegram getUpdates returned non-ok response", extra={"payload": str(payload)})
            return []

        rows = payload.get("result")
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict)]

    def send_message(
        self,
        text: str,
        parse_mode: str | None = None,
        disable_web_page_preview: bool = True,
    ) -> bool:
        if not self.enabled():
            logger.info("Telegram disabled. Missing token/chat id.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        chunks = split_for_telegram(text)
        for i, chunk in enumerate(chunks, start=1):
            body = {
                "chat_id": self.chat_id,
                "text": chunk,
                "disable_web_page_preview": bool(disable_web_page_preview),
            }
            if parse_mode:
                body["parse_mode"] = parse_mode
            try:
                payload = self.http.post_json(url, json_body=body)
            except Exception as exc:
                logger.warning(
                    "Telegram send failed",
                    extra={
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                        "chat_id": self.chat_id,
                        "error": str(exc),
                    },
                )
                return False

            if not (isinstance(payload, dict) and payload.get("ok") is True):
                logger.warning(
                    "Telegram API returned non-ok response",
                    extra={
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                        "chat_id": self.chat_id,
                        "payload": str(payload),
                    },
                )
                return False

        return True
