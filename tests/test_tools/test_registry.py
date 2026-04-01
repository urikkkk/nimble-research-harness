"""Tests for tool registry."""

import pytest

from nimble_research_harness.tools.registry import ToolDefinition, ToolRegistry


@pytest.fixture
def registry():
    r = ToolRegistry()

    async def handle_test(params):
        return {"result": params.get("input", "default")}

    r.register(
        ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"input": {"type": "string"}},
            },
            handler=handle_test,
        )
    )
    return r


def test_register_and_get(registry):
    tool = registry.get("test_tool")
    assert tool is not None
    assert tool.name == "test_tool"


def test_get_missing_returns_none(registry):
    assert registry.get("nonexistent") is None


def test_schema_generation(registry):
    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "test_tool"
    assert "input_schema" in schemas[0]


def test_filtered_schemas(registry):
    schemas = registry.get_schemas(["test_tool"])
    assert len(schemas) == 1
    schemas = registry.get_schemas(["nonexistent"])
    assert len(schemas) == 0


@pytest.mark.asyncio
async def test_dispatch(registry):
    result = await registry.dispatch("test_tool", {"input": "hello"})
    assert result == {"result": "hello"}


@pytest.mark.asyncio
async def test_dispatch_unknown_tool(registry):
    result = await registry.dispatch("unknown", {})
    assert "error" in result
