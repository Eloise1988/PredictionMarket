from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from prediction_agent.clients.http_client import HttpClient
from prediction_agent.models import EquitySnapshot

logger = logging.getLogger(__name__)


class AlphaVantageClient:
    def __init__(self, api_key: str, base_url: str, timeout: int = 15):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.http = HttpClient(timeout=timeout)

    def enabled(self) -> bool:
        return bool(self.api_key)

    def fetch_snapshot(self, ticker: str) -> Optional[EquitySnapshot]:
        if not self.enabled():
            return None

        symbol = ticker.upper()
        quote = self._request(function="GLOBAL_QUOTE", symbol=symbol)
        overview = self._request(function="OVERVIEW", symbol=symbol)

        if not quote and not overview:
            return None

        q = quote.get("Global Quote", {}) if isinstance(quote, dict) else {}
        o = overview if isinstance(overview, dict) else {}

        price = _to_float(q.get("05. price"))
        change_percent = _pct_to_float(q.get("10. change percent"))
        pe_ratio = _nullable_float(o.get("PERatio"))
        pb_ratio = _nullable_float(o.get("PriceToBookRatio"))
        ev_to_ebitda = _nullable_float(o.get("EVToEBITDA"))
        beta = _nullable_float(o.get("Beta"))
        market_cap = _nullable_float(o.get("MarketCapitalization"))

        return EquitySnapshot(
            ticker=symbol,
            name=(o.get("Name") or "").strip(),
            sector=(o.get("Sector") or "").strip(),
            price=price if price > 0 else None,
            change_percent=change_percent,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            ev_to_ebitda=ev_to_ebitda,
            beta=beta,
            market_cap=market_cap,
            fetched_at=datetime.now(timezone.utc),
            raw={"quote": q, "overview": o},
        )

    def fetch_daily_adjusted_series(self, ticker: str, outputsize: str = "full") -> Dict[date, float]:
        if not self.enabled():
            return {}

        symbol = ticker.upper()
        payload = self._request(
            function="TIME_SERIES_DAILY_ADJUSTED",
            symbol=symbol,
            outputsize=outputsize,
        )
        series = payload.get("Time Series (Daily)", {}) if isinstance(payload, dict) else {}
        if not isinstance(series, dict):
            return {}

        bars: Dict[date, float] = {}
        for day_str, row in series.items():
            if not isinstance(row, dict):
                continue
            try:
                day = date.fromisoformat(day_str)
            except ValueError:
                continue

            close = _to_float(row.get("5. adjusted close") or row.get("4. close"))
            if close > 0:
                bars[day] = close

        return bars

    def _request(self, function: str, symbol: str, **extra: Any) -> Dict[str, Any]:
        params = {"function": function, "symbol": symbol, "apikey": self.api_key, **extra}
        payload = self.http.get_json(self.base_url, params=params)
        if not isinstance(payload, dict):
            return {}

        if "Note" in payload:
            logger.warning("Alpha Vantage rate limit hit", extra={"message": payload.get("Note")})
            return {}
        if "Error Message" in payload:
            logger.debug("Alpha Vantage error", extra={"message": payload.get("Error Message")})
            return {}

        return payload


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _nullable_float(value: Any) -> Optional[float]:
    if value in (None, "", "None", "-", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace("%", "")) / 100.0
    except (TypeError, ValueError):
        return None
