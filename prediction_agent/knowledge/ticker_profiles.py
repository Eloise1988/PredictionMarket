from __future__ import annotations


def get_ticker_background(ticker: str, company_name: str = "", sector: str = "") -> str:
    t = (ticker or "").upper().strip()
    name = (company_name or "").strip()
    sec = (sector or "").strip()

    if name and sec:
        return f"{t} ({name}) operates in the {sec} sector."
    if name:
        return f"{t} is {name}."
    if sec:
        return f"{t} is linked to the {sec} sector."
    return f"{t} was selected dynamically from the market event using LLM analysis."
