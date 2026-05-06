from __future__ import annotations

from mlforge.agents.research import ResearchAgent
from mlforge.agents.tools import ToolCall, ToolResult


class FakeFirecrawlTool:
    name = "firecrawl_local"
    description = "fake"

    def invoke(self, call: ToolCall) -> ToolResult:
        if call.arguments["action"] == "search":
            return ToolResult(
                self.name,
                True,
                [{"title": "Result", "url": "https://example.com/research"}],
            )
        if call.arguments["action"] == "scrape":
            return ToolResult(
                self.name,
                True,
                {"markdown": "# Research\nUseful source content."},
            )
        return ToolResult(self.name, False, error="bad action")


def test_research_agent_collects_firecrawl_chunks() -> None:
    agent = ResearchAgent(FakeFirecrawlTool())  # type: ignore[arg-type]

    chunks = agent.collect_topic_chunks("dataset distillation")

    assert len(chunks) == 1
    assert chunks[0].kind == "web"
    assert chunks[0].source.url == "https://example.com/research"
    assert "Useful source content" in chunks[0].content
