"""Planner agent — converts skill spec into an execution plan."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..models.discovery import AgentFitScore
from ..models.enums import ExecutionMode, ToolName
from ..models.plan import PlanStep, ResearchPlan
from ..models.skill import DynamicSkillSpec
from ..models.session import SessionConfig
from .base import FAST_MODEL, run_agent_loop
from ..tools.registry import ToolDefinition, ToolRegistry

SYSTEM_PROMPT = """You are a research execution planner. Given a research skill spec with its objective,
subquestions, and strategies, you must create a concrete execution plan by calling `submit_plan`.

Your plan should:
1. Map each subquestion to specific tool calls (nimble_search, nimble_extract, nimble_map, nimble_agents_run)
2. Order steps by dependency (search before extract)
3. Use WSA agents when available for known domains
4. Include verification searches for important claims
5. Stay within the time budget constraints

Available tools for research steps (use exact param names):
- nimble_search: params must include "query" (single string), optional "focus" and "max_results"
- nimble_extract: params must include "url" (single string URL)
- nimble_map: params must include "url" (single string URL)
- nimble_crawl_run: params must include "url" (single string URL)
- nimble_agents_run: params must include "agent_name" and tool-specific params

IMPORTANT: Each step's params must use singular "query" (not "queries") and "url" (not "urls").
Each step handles ONE query or ONE url.

Always call `submit_plan` with your complete plan."""

_plan_result: ResearchPlan | None = None


async def create_plan(
    config: SessionConfig,
    skill: DynamicSkillSpec,
    wsa_matches: list[AgentFitScore],
    fast_mode: bool = False,
) -> ResearchPlan:
    """Generate an execution plan from a skill spec."""
    global _plan_result
    _plan_result = None

    registry = ToolRegistry()

    async def handle_submit_plan(params: dict[str, Any]) -> dict[str, Any]:
        global _plan_result
        steps = []
        for i, s in enumerate(params.get("steps", [])):
            steps.append(
                PlanStep(
                    order=i,
                    description=s.get("description", ""),
                    tool=ToolName(s.get("tool", "nimble_search")),
                    params=s.get("params", {}),
                    wsa_agent_name=s.get("wsa_agent_name"),
                    timeout_seconds=s.get("timeout", 60),
                )
            )

        _plan_result = ResearchPlan(
            session_id=config.session_id,
            skill_id=skill.skill_id,
            objective=skill.user_objective,
            subquestions=skill.subquestions,
            target_entities=skill.target_entities,
            execution_mode=skill.execution_mode,
            wsa_agents=[s.agent_name for s in wsa_matches if s.is_strong_match],
            steps=steps,
        )
        return {"status": "ok", "plan_id": str(_plan_result.plan_id), "step_count": len(steps)}

    registry.register(
        ToolDefinition(
            name="submit_plan",
            description="Submit the research execution plan.",
            input_schema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "tool": {
                                    "type": "string",
                                    "enum": [t.value for t in ToolName],
                                },
                                "params": {"type": "object"},
                                "wsa_agent_name": {"type": "string"},
                                "timeout": {"type": "integer", "default": 60},
                            },
                            "required": ["description", "tool", "params"],
                        },
                    },
                },
                "required": ["steps"],
            },
            handler=handle_submit_plan,
        )
    )

    wsa_info = ""
    if wsa_matches:
        wsa_lines = [
            f"- {s.agent_name} (score: {s.composite_score:.2f})"
            for s in wsa_matches[:10]
        ]
        wsa_info = f"\n\nAvailable WSA agents that match this task:\n" + "\n".join(wsa_lines)
        wsa_info += "\nPrefer these for structured extraction from their target domains."

    user_prompt = f"""Create an execution plan for this research skill:

Objective: {skill.user_objective}
Task type: {skill.task_type.value}
Time budget: {skill.time_budget.label}
Execution mode: {skill.execution_mode.value}

Subquestions:
{chr(10).join(f'- {q}' for q in skill.subquestions)}

Search queries from skill spec:
{chr(10).join(f'- {q}' for q in skill.search_strategy.queries)}

Max searches: {config.policy.max_searches}
Max extracts: {config.policy.max_extracts}
Max crawl pages: {config.policy.max_crawl_pages}
{wsa_info}

Call `submit_plan` with ordered steps."""

    kwargs = {}
    if fast_mode:
        kwargs["model"] = FAST_MODEL
    await run_agent_loop(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        registry=registry,
        tool_names=["submit_plan"],
        max_turns=3,
        **kwargs,
    )

    if _plan_result is None:
        steps = []
        for i, q in enumerate(skill.search_strategy.queries or [skill.user_objective]):
            steps.append(
                PlanStep(
                    order=i,
                    description=f"Search: {q}",
                    tool=ToolName.SEARCH,
                    params={"query": q, "max_results": 10},
                )
            )
        _plan_result = ResearchPlan(
            session_id=config.session_id,
            skill_id=skill.skill_id,
            objective=skill.user_objective,
            subquestions=skill.subquestions,
            steps=steps,
        )

    return _plan_result
