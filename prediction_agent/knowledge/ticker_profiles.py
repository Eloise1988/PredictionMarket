from __future__ import annotations

from typing import Dict

_TICKER_PROFILES: Dict[str, str] = {
    "LMT": "Lockheed Martin is a major U.S. defense contractor with exposure to missiles, aircraft, and military programs.",
    "RTX": "RTX (Raytheon) is a defense and aerospace firm focused on missiles, air defense systems, and aviation components.",
    "NOC": "Northrop Grumman is a defense prime with heavy exposure to aerospace, missile defense, and strategic systems.",
    "GD": "General Dynamics is a defense contractor with naval, combat systems, and aerospace businesses.",
    "HII": "Huntington Ingalls is a U.S. naval shipbuilder tied to defense procurement cycles.",
    "XOM": "ExxonMobil is a global integrated oil major sensitive to crude supply and geopolitical disruptions.",
    "CVX": "Chevron is a large integrated energy company with earnings linked to oil and gas prices.",
    "MPC": "Marathon Petroleum is a U.S. refiner and downstream energy operator exposed to fuel margins and crude flows.",
    "DAL": "Delta Air Lines is an airline whose costs are sensitive to jet fuel and macro demand.",
    "UAL": "United Airlines is a passenger airline sensitive to fuel prices, travel demand, and economic cycles.",
    "QQQ": "QQQ tracks the Nasdaq-100, concentrated in large-cap growth and technology-heavy companies.",
    "IWM": "IWM tracks U.S. small-cap equities, which are typically more rate- and cycle-sensitive.",
    "TLT": "TLT holds long-duration U.S. Treasuries and is sensitive to interest-rate expectations.",
    "KRE": "KRE tracks U.S. regional banks, with performance tied to credit conditions and rate dynamics.",
    "XLE": "XLE tracks large U.S. energy companies and tends to move with oil and gas price expectations.",
    "SMH": "SMH tracks semiconductor companies with cyclical and geopolitical supply-chain exposure.",
    "NVDA": "NVIDIA is a semiconductor and AI infrastructure leader with high growth and valuation sensitivity.",
    "AAPL": "Apple is a global consumer technology company exposed to supply chains, demand cycles, and policy risks.",
    "XLP": "XLP tracks U.S. consumer staples, often viewed as defensive during slower growth periods.",
    "XLU": "XLU tracks U.S. utilities, a defensive sector sensitive to rates and regulated returns.",
    "XLY": "XLY tracks U.S. consumer discretionary companies, usually more cyclical to growth and spending.",
}


def get_ticker_background(ticker: str, company_name: str = "", sector: str = "") -> str:
    t = (ticker or "").upper()
    if t in _TICKER_PROFILES:
        return _TICKER_PROFILES[t]

    name = (company_name or "").strip()
    sec = (sector or "").strip()
    if name and sec:
        return f"{name} operates in the {sec} sector."
    if name:
        return f"{name} is the underlying company for this idea."
    return ""
