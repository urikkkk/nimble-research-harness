"""Skill builder agent — generates DynamicSkillSpec via Claude tool_use."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..models.enums import ExecutionMode, SearchFocus, TaskType, TimeBudget, ToolName
from ..models.session import SessionConfig
from ..models.skill import (
    DynamicSkillSpec,
    ExtractionPolicy,
    ExtractionStrategy,
    PlanningPolicy,
    ReportPolicy,
    SearchStrategy,
    SourcePolicy,
    SynthesisPolicy,
    VerificationPolicy,
)
from .base import AgentResult, FAST_MODEL, run_agent_loop
from ..tools.registry import ToolDefinition, ToolRegistry

SYSTEM_PROMPT = """You are a research skill architect. Given a user's research objective and time budget,
you must design a complete research skill specification by calling the `create_skill_spec` tool.

The skill spec defines HOW the research should be conducted — what to search for, where to look,
how to extract information, how to synthesize findings, and how to verify claims.

Consider:
- The time budget constrains depth and breadth
- Different task types need different strategies (e.g., competitive intel needs comparisons)
- Subquestions should decompose the objective into searchable units
- Source types should match the domain (e.g., SEC filings for finance, arxiv for academic)
- Search strategy should use appropriate focus modes
- Verification should be proportional to the time budget

Always call `create_skill_spec` with your complete specification."""

_skill_result: DynamicSkillSpec | None = None


async def build_skill(config: SessionConfig, fast_mode: bool = False) -> DynamicSkillSpec:
    """Generate a dynamic skill spec for the given session config."""
    global _skill_result
    _skill_result = None

    registry = ToolRegistry()

    async def handle_create_skill(params: dict[str, Any]) -> dict[str, Any]:
        global _skill_result
        _skill_result = DynamicSkillSpec(
            session_id=config.session_id,
            title=params.get("title", "Research Skill"),
            user_objective=config.normalized_objective,
            task_type=TaskType(params.get("task_type", config.task_type.value)),
            time_budget=config.time_budget,
            subquestions=params.get("subquestions", []),
            target_entities=params.get("target_entities", config.target_entities),
            likely_source_types=params.get("likely_source_types", []),
            planning_policy=PlanningPolicy(
                depth=config.policy.planning_depth,
                max_subquestions=min(config.policy.search_breadth, 10),
            ),
            source_policy=SourcePolicy(
                min_sources=config.policy.stop_conditions.min_sources,
                domain_include=params.get("domain_include", config.target_domains),
                domain_exclude=params.get("domain_exclude", []),
                preferred_focus_modes=[
                    SearchFocus(f) for f in params.get("focus_modes", ["general"])
                ],
            ),
            extraction_policy=ExtractionPolicy(
                max_content_length=min(10000, config.policy.max_extracts * 500),
            ),
            synthesis_policy=SynthesisPolicy(
                max_findings=config.policy.search_breadth,
                require_comparisons=config.task_type == TaskType.COMPETITIVE_INTEL,
            ),
            verification_policy=VerificationPolicy(
                strictness=config.policy.verification_strictness,
                require_corroboration=config.policy.verification_strictness >= 2,
                max_verification_searches=min(5, config.policy.max_searches // 4),
            ),
            report_policy=ReportPolicy(format=config.report_format),
            search_strategy=SearchStrategy(
                queries=params.get("search_queries", []),
                focus_modes=[SearchFocus(f) for f in params.get("focus_modes", ["general"])],
                include_domains=params.get("domain_include", config.target_domains),
                max_results_per_query=min(20, config.policy.search_breadth),
            ),
            extraction_strategy=ExtractionStrategy(
                priority_urls=params.get("priority_urls", []),
                crawl_targets=params.get("crawl_targets", []),
            ),
            execution_mode=config.execution_mode,
        )
        return {"status": "ok", "skill_id": str(_skill_result.skill_id)}

    registry.register(
        ToolDefinition(
            name="create_skill_spec",
            description="Create a research skill specification.",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title for this research skill"},
                    "task_type": {
                        "type": "string",
                        "enum": [t.value for t in TaskType],
                    },
                    "subquestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Break the objective into searchable subquestions",
                    },
                    "target_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key entities to research (companies, people, products, etc.)",
                    },
                    "likely_source_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Types of sources to look for (news, docs, filings, etc.)",
                    },
                    "search_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific search queries to execute",
                    },
                    "focus_modes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["general", "news", "coding", "academic", "shopping", "social"],
                        },
                        "default": ["general"],
                    },
                    "domain_include": {"type": "array", "items": {"type": "string"}, "default": []},
                    "domain_exclude": {"type": "array", "items": {"type": "string"}, "default": []},
                    "priority_urls": {"type": "array", "items": {"type": "string"}, "default": []},
                    "crawl_targets": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": ["title", "subquestions", "search_queries"],
            },
            handler=handle_create_skill,
        )
    )

    user_prompt = f"""Research objective: {config.normalized_objective}

Task type: {config.task_type.value}
Time budget: {config.time_budget.label} ({config.policy.max_searches} max searches, {config.policy.max_extracts} max extracts)
Target domains: {config.target_domains or 'none specified'}
Report format: {config.report_format.value}

Design a complete research skill specification for this objective.
Call `create_skill_spec` with all the details."""

    kwargs = {}
    if fast_mode:
        kwargs["model"] = FAST_MODEL
    result = await run_agent_loop(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        registry=registry,
        tool_names=["create_skill_spec"],
        max_turns=3,
        **kwargs,
    )

    if _skill_result is None:
        _skill_result = DynamicSkillSpec(
            session_id=config.session_id,
            title=f"Research: {config.user_query[:50]}",
            user_objective=config.normalized_objective,
            task_type=config.task_type,
            time_budget=config.time_budget,
            subquestions=[config.normalized_objective],
            search_strategy=SearchStrategy(queries=[config.user_query]),
        )

    return _skill_result
