from __future__ import annotations

from typing import List


def split_for_telegram(text: str, max_chars: int = 3900, hard_cap: int = 4096) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= hard_cap:
        return [text]

    chunks: List[str] = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = remaining.rfind("\n", 0, max_chars)
        if split_at < 0:
            split_at = max_chars
        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks
