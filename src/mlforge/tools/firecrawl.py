from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request as urlrequest


@dataclass(slots=True)
class FirecrawlClient:
    base_url: str = "http://localhost:3002"

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        payload = {"query": query, "limit": limit}
        raw = self._post("/v1/search", payload)
        data = raw.get("data", raw.get("results", []))
        return data if isinstance(data, list) else []

    def scrape(self, url: str) -> dict[str, Any]:
        raw = self._post("/v1/scrape", {"url": url})
        data = raw.get("data", raw)
        return data if isinstance(data, dict) else {"content": str(data)}

    def health(self) -> dict[str, Any]:
        try:
            self.search("health check", limit=1)
        except RuntimeError as exc:
            return {"tool": "firecrawl-local", "ok": False, "error": str(exc)}
        return {"tool": "firecrawl-local", "ok": True, "base_url": self.base_url}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Firecrawl-local request failed: {exc}") from exc
