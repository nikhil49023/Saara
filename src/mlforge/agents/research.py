from __future__ import annotations

from dataclasses import dataclass

from mlforge.core.hashing import short_hash
from mlforge.core.schemas import DocumentChunk, ResearchSource

from .tools import FirecrawlLocalTool, ToolCall


@dataclass(slots=True)
class ResearchAgent:
    """Deterministic research agent for topic-to-source collection.

    This is intentionally not a free-form autonomous browser. It performs bounded
    tool calls so generated datasets remain reproducible and auditable.
    """

    tool: FirecrawlLocalTool
    search_limit: int = 5
    scrape_limit: int = 5
    max_chunk_chars: int = 6000

    def collect_topic_chunks(self, topic: str) -> list[DocumentChunk]:
        search = self.tool.invoke(
            ToolCall(
                name=self.tool.name,
                arguments={"action": "search", "query": topic, "limit": self.search_limit},
            )
        )
        if not search.ok or not isinstance(search.data, list):
            return []

        chunks: list[DocumentChunk] = []
        for index, result in enumerate(search.data[: self.scrape_limit], start=1):
            if not isinstance(result, dict):
                continue
            url = result.get("url") or result.get("link")
            if not url:
                continue
            scraped = self.tool.invoke(
                ToolCall(name=self.tool.name, arguments={"action": "scrape", "url": str(url)})
            )
            if not scraped.ok or not isinstance(scraped.data, dict):
                continue
            content = (
                scraped.data.get("markdown")
                or scraped.data.get("content")
                or scraped.data.get("text")
                or ""
            )
            content = str(content).strip()
            if not content:
                continue
            chunk_id = f"web_{index}_chunk_1"
            source = ResearchSource(
                source_id=f"web_{index}",
                title=result.get("title") or scraped.data.get("title"),
                url=str(url),
                chunk_id=chunk_id,
                content_hash=short_hash(content),
                metadata={"tool": self.tool.name, "search_result": result},
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    content=content[: self.max_chunk_chars],
                    source=source,
                    kind="web",
                )
            )
        return chunks
