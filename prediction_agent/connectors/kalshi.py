from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse
import re

from prediction_agent.clients.http_client import HttpClient
from prediction_agent.connectors.base import PredictionConnector
from prediction_agent.models import PredictionSignal

logger = logging.getLogger(__name__)


class KalshiConnector(PredictionConnector):
    source_name = "kalshi"

    def __init__(self, base_url: str, limit: int = 200, timeout: int = 15):
        self.base_url = base_url.rstrip("/")
        self.search_url = _resolve_series_search_url(self.base_url)
        self.limit = limit
        self.http = HttpClient(timeout=timeout)

    def fetch_signals(self) -> List[PredictionSignal]:
        signals: List[PredictionSignal] = []
        cursor: Optional[str] = None
        seen_ids: set[str] = set()
        fetched_at = datetime.now(timezone.utc)

        while len(signals) < self.limit:
            remaining = self.limit - len(signals)
            params: Dict[str, Any] = {
                "order_by": "event-volume",
                "status": "open,unopened",
                "reverse": "false",
                "with_milestones": "true",
                "hydrate": "milestones,structured_targets",
                "page_size": min(100, remaining),
            }
            if cursor:
                params["cursor"] = cursor

            payload = self.http.get_json(self.search_url, params=params)
            series_rows = _extract_series_rows(payload)
            if not series_rows:
                break

            added = 0
            for series in series_rows:
                if not isinstance(series, dict):
                    continue

                for market in _extract_series_markets(series):
                    ticker = str(market.get("ticker") or market.get("market_id") or market.get("id") or "").strip()
                    if not ticker or ticker in seen_ids:
                        continue

                    prob_yes = _extract_yes_probability(market)
                    if prob_yes is None:
                        continue

                    question = _build_market_question(series, market)
                    if not question:
                        continue

                    event_ticker = str(series.get("event_ticker") or market.get("event_ticker") or "").strip()
                    series_ticker = str(series.get("series_ticker") or market.get("series_ticker") or "").strip()
                    slug = str(market.get("slug") or "").strip()
                    url = _build_kalshi_market_url(
                        series_ticker=series_ticker,
                        event_ticker=event_ticker,
                        event_title=str(series.get("event_title") or series.get("series_title") or "").strip(),
                        market_ticker=ticker,
                        slug=slug,
                    )
                    yes_price, no_price = _extract_yes_no_prices(market, prob_yes)

                    volume_24h = _to_float(
                        market.get("volume_24h")
                        or market.get("volume24hr")
                        or market.get("volume")
                        or series.get("total_volume")
                        or series.get("total_series_volume")
                        or 0
                    )
                    liquidity = _to_float(
                        market.get("open_interest")
                        or market.get("liquidity")
                        or market.get("volume")
                        or series.get("total_volume")
                        or series.get("total_series_volume")
                        or volume_24h
                    )
                    updated_at = _parse_dt_or_default(
                        market.get("updated_time")
                        or market.get("updated_at")
                        or series.get("updated_at")
                        or series.get("updatedAt"),
                        fetched_at,
                    )

                    seen_ids.add(ticker)
                    added += 1
                    signals.append(
                        PredictionSignal(
                            source=self.source_name,
                            market_id=ticker,
                            question=question,
                            url=url,
                            prob_yes=prob_yes,
                            volume_24h=volume_24h,
                            liquidity=liquidity,
                            updated_at=updated_at,
                            raw={
                                "ticker": ticker,
                                "event_ticker": event_ticker,
                                "series_ticker": series_ticker,
                                "slug": slug,
                                "title": series.get("event_title") or series.get("series_title"),
                                "subtitle": series.get("event_subtitle"),
                                "yes_sub_title": market.get("yes_subtitle") or market.get("yes_sub_title"),
                                "no_sub_title": market.get("no_subtitle") or market.get("no_sub_title"),
                                "status": market.get("status") or series.get("status"),
                                "close_time": market.get("close_ts") or market.get("expected_expiration_ts"),
                                "category": series.get("category"),
                                "event_title": series.get("event_title"),
                                "event_subtitle": series.get("event_subtitle"),
                                "tags": _extract_series_tags(series),
                                "yes_price": yes_price,
                                "no_price": no_price,
                                "yes_bid": market.get("yes_bid"),
                                "yes_ask": market.get("yes_ask"),
                                "last_price": market.get("last_price"),
                                "yes_bid_dollars": market.get("yes_bid_dollars"),
                                "yes_ask_dollars": market.get("yes_ask_dollars"),
                                "last_price_dollars": market.get("last_price_dollars"),
                                "no_bid": market.get("no_bid"),
                                "no_ask": market.get("no_ask"),
                                "fee_type": series.get("fee_type") or market.get("fee_type"),
                                "fee_multiplier": series.get("fee_multiplier") or market.get("fee_multiplier"),
                            },
                        )
                    )
                    if len(signals) >= self.limit:
                        break

                if len(signals) >= self.limit:
                    break

            if added == 0:
                break

            cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
            if not cursor:
                break

        return signals


def _resolve_series_search_url(base_url: str) -> str:
    raw = (base_url or "").strip().rstrip("/")
    if not raw:
        return "https://api.elections.kalshi.com/v1/search/series"
    if raw.endswith("/search/series"):
        return raw
    if "/trade-api/" in raw:
        return f"{raw.split('/trade-api/')[0]}/v1/search/series"

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return f"{raw}/search/series"

    path = parsed.path.rstrip("/")
    if path == "/v1":
        new_path = "/v1/search/series"
    elif path.startswith("/v1/"):
        new_path = "/v1/search/series"
    elif not path:
        new_path = "/v1/search/series"
    else:
        new_path = f"{path}/search/series"

    return urlunparse((parsed.scheme, parsed.netloc, new_path, "", "", ""))


def _extract_series_rows(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    rows = payload.get("current_page")
    if isinstance(rows, list):
        return [x for x in rows if isinstance(x, dict)]
    return []


def _extract_series_markets(series: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = series.get("markets")
    if isinstance(rows, list):
        return [x for x in rows if isinstance(x, dict)]
    if isinstance(rows, dict):
        return [rows]
    return []


def _build_market_question(series: Dict[str, Any], market: Dict[str, Any]) -> str:
    market_title = str(market.get("title") or "").strip()
    if market_title:
        return market_title

    event_title = str(series.get("event_title") or series.get("series_title") or "").strip()
    event_subtitle = str(series.get("event_subtitle") or "").strip()
    yes_subtitle = str(market.get("yes_subtitle") or market.get("yes_sub_title") or "").strip()

    if event_title and yes_subtitle:
        return f"{event_title} - {yes_subtitle}"
    if event_title and event_subtitle:
        return f"{event_title} ({event_subtitle})"
    if event_title:
        return event_title
    if yes_subtitle:
        return yes_subtitle
    return str(market.get("ticker") or market.get("market_id") or "").strip()


def _build_kalshi_market_url(
    series_ticker: str,
    event_ticker: str,
    event_title: str,
    market_ticker: str,
    slug: str = "",
) -> str:
    if slug:
        return f"https://kalshi.com/markets/{slug}"

    market_ticker_l = market_ticker.lower().strip()
    if not market_ticker_l:
        return "https://kalshi.com/markets"

    series_segment = series_ticker.lower().strip()
    if not series_segment:
        if event_ticker:
            series_segment = event_ticker.split("-")[0].lower().strip()

    event_segment = _slugify(event_title)
    if series_segment and event_segment:
        return f"https://kalshi.com/markets/{series_segment}/{event_segment}/{market_ticker_l}"
    if event_ticker:
        return f"https://kalshi.com/markets/{event_ticker.lower()}"
    return f"https://kalshi.com/markets/{market_ticker_l}"


def _extract_series_tags(series: Dict[str, Any]) -> List[str]:
    tags: List[str] = []

    category = series.get("category")
    if isinstance(category, str) and category.strip():
        tags.append(category.strip())

    metadata = series.get("product_metadata")
    if not isinstance(metadata, dict):
        return tags

    categories = metadata.get("categories")
    if isinstance(categories, list):
        for item in categories:
            if isinstance(item, str) and item.strip():
                tags.append(item.strip())

    subcategories = metadata.get("subcategories")
    if isinstance(subcategories, dict):
        for value in subcategories.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        tags.append(item.strip())
            elif isinstance(value, str) and value.strip():
                tags.append(value.strip())

    # Preserve order while deduping.
    out: List[str] = []
    seen: set[str] = set()
    for tag in tags:
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def _extract_yes_no_prices(market: Dict[str, Any], fallback_yes_prob: float) -> tuple[float, float]:
    # Match Kalshi UI-style "Buy Yes / Buy No" as closely as possible:
    # yes side prefers ask, no side prefers explicit no_ask or complement of yes_bid.
    yes = _to_prob(market.get("yes_ask_dollars"))
    if yes is None:
        yes = _to_prob(market.get("yes_ask"))
    if yes is None:
        yes = _to_prob(market.get("last_price_dollars"))
    if yes is None:
        yes = _to_prob(market.get("last_price"))
    if yes is None:
        yes = _to_prob(market.get("yes_bid_dollars"))
    if yes is None:
        yes = _to_prob(market.get("yes_bid"))
    if yes is None:
        yes = max(0.0, min(1.0, float(fallback_yes_prob)))

    no = _to_prob(market.get("no_ask_dollars"))
    if no is None:
        no = _to_prob(market.get("no_ask"))
    if no is None:
        yes_bid = _to_prob(market.get("yes_bid_dollars"))
        if yes_bid is None:
            yes_bid = _to_prob(market.get("yes_bid"))
        if yes_bid is not None:
            no = max(0.0, min(1.0, 1.0 - yes_bid))
    if no is None:
        no = max(0.0, min(1.0, 1.0 - yes))

    return yes, no


def _extract_yes_probability(market: Dict[str, Any]) -> Optional[float]:
    for key in (
        "yes_price",
        "yes_last_price",
        "yes_bid",
        "yes_ask",
        "last_price",
        "yes_bid_dollars",
        "yes_ask_dollars",
        "last_price_dollars",
    ):
        if key in market and market[key] is not None:
            value = _to_float(market[key])
            if value <= 0:
                continue
            if value <= 1:
                return value
            if value <= 100:
                return value / 100.0

    yes_bid = _to_float(market.get("yes_bid"))
    yes_ask = _to_float(market.get("yes_ask"))
    if yes_bid > 0 and yes_ask > 0:
        midpoint = (yes_bid + yes_ask) / 2
        return midpoint if midpoint <= 1 else midpoint / 100.0

    no_bid = _to_float(market.get("no_bid"))
    if no_bid > 0:
        no_prob = no_bid if no_bid <= 1 else no_bid / 100.0
        return max(0.0, min(1.0, 1.0 - no_prob))

    return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_prob(value: Any) -> Optional[float]:
    v = _to_float(value)
    if v <= 0:
        return None
    if v <= 1:
        return max(0.0, min(1.0, v))
    if v <= 100:
        return max(0.0, min(1.0, v / 100.0))
    return None


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return re.sub(r"-{2,}", "-", cleaned)


def _parse_dt(raw: Any) -> datetime:
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _parse_dt_or_default(raw: Any, default: datetime) -> datetime:
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return default
