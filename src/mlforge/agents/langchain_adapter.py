from __future__ import annotations

from typing import Any

from .tools import AgentTool, ToolCall


def to_langchain_tool(tool: AgentTool) -> Any:
    """Adapt a Saara tool to LangChain when langchain-core is installed."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise RuntimeError("Install optional agent dependencies: pip install 'saara-ai[agents]'") from exc

    def _invoke(action: str, query: str | None = None, limit: int = 5, url: str | None = None) -> Any:
        args: dict[str, Any] = {"action": action}
        if query is not None:
            args["query"] = query
        if url is not None:
            args["url"] = url
        args["limit"] = limit
        result = tool.invoke(ToolCall(name=tool.name, arguments=args))
        if not result.ok:
            raise RuntimeError(result.error or "Tool call failed")
        return result.data

    return StructuredTool.from_function(
        func=_invoke,
        name=tool.name,
        description=tool.description,
    )
