from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from mlforge.tools.firecrawl import FirecrawlClient


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    name: str
    ok: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentTool(Protocol):
    name: str
    description: str

    def invoke(self, call: ToolCall) -> ToolResult:
        ...


@dataclass(slots=True)
class FirecrawlLocalTool:
    base_url: str = "http://localhost:3002"
    name: str = "firecrawl_local"
    description: str = (
        "Search and scrape web pages through the self-hosted Firecrawl instance. "
        "Actions: search(query, limit), scrape(url)."
    )

    def invoke(self, call: ToolCall) -> ToolResult:
        client = FirecrawlClient(self.base_url)
        action = call.arguments.get("action")
        try:
            if action == "search":
                query = str(call.arguments["query"])
                limit = int(call.arguments.get("limit", 5))
                return ToolResult(self.name, True, client.search(query, limit=limit))
            if action == "scrape":
                url = str(call.arguments["url"])
                return ToolResult(self.name, True, client.scrape(url))
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return ToolResult(self.name, False, error=str(exc))
        return ToolResult(self.name, False, error=f"Unsupported action: {action}")
