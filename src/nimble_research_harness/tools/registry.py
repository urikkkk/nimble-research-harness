"""Tool registry — maps tool names to handlers and builds Anthropic SDK schemas."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from ..infra.logging import get_logger

logger = get_logger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class ToolDefinition:
    """A tool with its Anthropic-compatible schema and handler."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: ToolHandler,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def to_anthropic_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    """Central registry of all available tools for agent phases."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.debug("tool_registered", name=tool.name)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_schemas(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        if names:
            return [
                self._tools[n].to_anthropic_schema()
                for n in names
                if n in self._tools
            ]
        return [t.to_anthropic_schema() for t in self._tools.values()]

    async def dispatch(self, name: str, input_data: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}
        try:
            return await tool.handler(input_data)
        except Exception as e:
            logger.error("tool_dispatch_error", tool=name, error=str(e))
            return {"error": str(e)}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
