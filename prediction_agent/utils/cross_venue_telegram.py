from __future__ import annotations

import html
from datetime import datetime
from typing import Dict, List


def format_cross_venue_telegram_header(
    row_count: int,
    sent_at: datetime,
    use_llm_strong: bool,
    sim_label: str,
) -> str:
    mode = "llm-strong direct match" if use_llm_strong else "heuristic matcher"
    return (
        "<b>Cross-Venue Arbitrage Digest</b>\n"
        f"<b>Rows:</b> {max(0, int(row_count))} | <b>Mode:</b> {escape_html(mode)}\n"
        f"<b>Min sim:</b> {escape_html(sim_label)} | <b>Generated (UTC):</b> {sent_at.strftime('%Y-%m-%d %H:%M')}"
    )


def format_cross_venue_telegram_cards(rows: List[Dict]) -> List[str]:
    cards: List[str] = []
    for rank, row in enumerate(rows, start=1):
        cards.append(format_cross_venue_telegram_card(rank=rank, row=row))
    return cards


def format_cross_venue_telegram_card(rank: int, row: Dict) -> str:
    pm_yes = float(row.get("pm_yes", 0.0))
    pm_no = float(row.get("pm_no", 0.0))
    ka_yes = float(row.get("ka_yes", 0.0))
    ka_no = float(row.get("ka_no", 0.0))
    arb_flag = str(row.get("arb_flag", "no")).upper()
    arb_pnl = float(row.get("arb_pnl", 0.0))
    liquidity_sum = float(row.get("liquidity_sum", 0.0))
    prob_diff_pp = float(row.get("prob_diff_pp", 0.0))
    sim = str(row.get("sim", "n/a"))
    edge_hint = str(row.get("edge_hint", "")).strip()

    pm_question = truncate_text(str(row.get("pm_question", "")).strip(), 240)
    ks_question = truncate_text(str(row.get("ks_question", "")).strip(), 240)
    pm_link = str(row.get("pm_link", "")).strip()
    ks_link = str(row.get("ks_link", "")).strip()

    lines = [
        f"<b>Rank #{max(1, int(rank))}</b> | <b>Arb:</b> {arb_flag} | <b>PnL ($1k net):</b> ${arb_pnl:,.2f}",
        f"<b>PM:</b> YES {format_price_cents(pm_yes)} | NO {format_price_cents(pm_no)}",
        f"<b>Kalshi:</b> YES {format_price_cents(ka_yes)} | NO {format_price_cents(ka_no)}",
        f"<b>Diff:</b> {prob_diff_pp:.2f}pp | <b>Sim:</b> {escape_html(sim)} | <b>Liq sum:</b> ${liquidity_sum:,.0f}",
    ]
    if edge_hint:
        lines.append(f"<b>Edge:</b> {escape_html(edge_hint)}")
    lines.append(f"<b>Polymarket:</b> {html_link(pm_question, pm_link)}")
    lines.append(f"<b>Kalshi:</b> {html_link(ks_question, ks_link)}")
    return "\n".join(lines)


def format_price_cents(prob: float) -> str:
    cents = max(0.0, min(100.0, float(prob) * 100.0))
    if abs(cents - round(cents)) < 0.05:
        return f"{int(round(cents))}c"
    return f"{cents:.1f}c"


def html_link(label: str, url: str) -> str:
    safe_label = escape_html((label or "").strip() or "open")
    safe_url = str(url or "").strip()
    if safe_url and safe_url.startswith(("http://", "https://")):
        return f"<a href=\"{html.escape(safe_url, quote=True)}\">{safe_label}</a>"
    return safe_label


def escape_html(text: str) -> str:
    return html.escape(str(text or ""), quote=False)


def truncate_text(text: str, max_len: int) -> str:
    cleaned = (text or "").strip()
    if max_len <= 3 or len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."
