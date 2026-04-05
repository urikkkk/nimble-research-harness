"""Base agent classes and the agentic loop runner."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import anthropic

from ..infra.logging import get_logger
from ..tools.registry import ToolRegistry

logger = get_logger(__name__)

DEFAULT_MODEL = os.environ.get("NRH_MODEL", "claude-sonnet-4-6")
FAST_MODEL = os.environ.get("NRH_FAST_MODEL", "claude-haiku-4-5-20251001")


@dataclass
class AgentResult:
    text: str
    tool_calls_made: int = 0
    turns: int = 0


async def run_agent_loop(
    system_prompt: str,
    user_prompt: str,
    registry: ToolRegistry,
    tool_names: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 20,
    max_tokens: int = 4096,
) -> AgentResult:
    """Run an Anthropic SDK agentic loop with tool_use until the agent stops."""
    client = anthropic.AsyncAnthropic()
    tools = registry.get_schemas(tool_names)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    total_tool_calls = 0
    turn = 0

    for turn in range(max_turns):
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await client.messages.create(**kwargs)

        has_tool_use = any(b.type == "tool_use" for b in response.content)

        if not has_tool_use:
            text_parts = [b.text for b in response.content if b.type == "text"]
            return AgentResult(
                text="\n".join(text_parts),
                tool_calls_made=total_tool_calls,
                turns=turn + 1,
            )

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                total_tool_calls += 1
                logger.info(
                    "tool_call",
                    tool=block.name,
                    turn=turn,
                    input_keys=list(block.input.keys()) if isinstance(block.input, dict) else [],
                )
                result = await registry.dispatch(block.name, block.input)
                raw = json.dumps(result, default=str)
                if len(raw) > 4000:
                    content = raw[:3900] + "\n...[truncated]"
                else:
                    content = raw
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    }
                )

        messages.append({"role": "user", "content": tool_results})

    text_parts = []
    if messages and messages[-1].get("role") == "assistant":
        content = messages[-1].get("content", [])
        if isinstance(content, list):
            text_parts = [b.text for b in content if hasattr(b, "text")]
    return AgentResult(
        text="\n".join(text_parts) if text_parts else "(max turns reached)",
        tool_calls_made=total_tool_calls,
        turns=turn + 1,
    )
