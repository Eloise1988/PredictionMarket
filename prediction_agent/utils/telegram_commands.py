from __future__ import annotations

import re
from typing import Any, Dict


_ARB_CMD_RE = re.compile(r"^/arb(?:@[a-z0-9_]+)?(?:\s+.*)?$", re.IGNORECASE)
_HELP_CMD_RE = re.compile(r"^/(?:help|start)(?:@[a-z0-9_]+)?(?:\s+.*)?$", re.IGNORECASE)


def is_arb_command(text: str) -> bool:
    return bool(_ARB_CMD_RE.match((text or "").strip()))


def is_help_command(text: str) -> bool:
    return bool(_HELP_CMD_RE.match((text or "").strip()))


def extract_update_text(update: Dict[str, Any]) -> str:
    msg = _extract_message_obj(update)
    text = msg.get("text") if isinstance(msg, dict) else ""
    return str(text or "").strip()


def extract_update_chat_id(update: Dict[str, Any]) -> str:
    msg = _extract_message_obj(update)
    if not isinstance(msg, dict):
        return ""
    chat = msg.get("chat")
    if not isinstance(chat, dict):
        return ""
    chat_id = chat.get("id")
    return str(chat_id or "").strip()


def extract_update_id(update: Dict[str, Any]) -> int:
    try:
        return int(update.get("update_id"))
    except (TypeError, ValueError, AttributeError):
        return -1


def _extract_message_obj(update: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(update, dict):
        return {}
    msg = update.get("message")
    if isinstance(msg, dict):
        return msg
    edited = update.get("edited_message")
    if isinstance(edited, dict):
        return edited
    return {}
