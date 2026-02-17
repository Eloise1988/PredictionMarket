from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class HttpClient:
    def __init__(self, timeout: int = 15):
        self._session = requests.Session()
        self._timeout = timeout

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        response = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
        response.raise_for_status()
        return response.json()

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def post_json(self, url: str, json_body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
        response = self._session.post(url, json=json_body, headers=headers, timeout=self._timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text[:600] if response is not None else ""
            raise requests.HTTPError(f"{exc} | response_body={body}") from exc
        return response.json()
