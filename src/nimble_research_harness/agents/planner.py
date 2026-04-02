"""Planner agent — converts skill spec into an execution plan."""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from ..models.discovery import AgentFitScore
from ..models.enums import ExecutionMode, ToolName
from ..models.evidence import EvidenceItem
from ..models.plan import PlanStep, ResearchPlan
from ..models.skill import DynamicSkillSpec
from ..models.session import SessionConfig
from .base import FAST_MODEL, run_agent_loop
from ..tools.registry import ToolDefinition, ToolRegistry

SYSTEM_PROMPT = """You are a research execution planner. Given a research skill spec with its objective,
subquestions, and strategies, you must create a concrete execution plan by calling `submit_plan`.

Your plan should:
1. Map each subquestion to specific tool calls (nimble_search, nimble_extract, nimble_map, nimble_agents_run)
2. PRIORITIZE nimble_search — it yields 10x more evidence than extraction for most queries
3. Use nimble_extract SPARINGLY — only for specific high-value pages, NOT for retailer category pages (they are JS-rendered and return empty content)
4. Use WSA agents when available for known domains
5. Include verification searches for important claims
6. Stay within the time budget constraints

STRATEGY — TWO-WAVE PIPELINE:

Wave 1 (parallel seed layer — all run at once, no dependencies):
- nimble_search steps: broad web queries to find URLs, snippets, and general data
- SERP WSA steps: structured search on specific retailer/directory sites (if available)
- Both run IN PARALLEL to maximize data collection speed

Wave 2 (dependent detail layer — runs after Wave 1 completes):
- PDP WSA steps: use product URLs or IDs found in Wave 1 to extract full product details
- These steps should set depends_on to reference Wave 1 step IDs
- Only include Wave 2 if PDP WSAs are available AND Wave 1 will produce URLs

When NO WSA agents are available, allocate 100% to nimble_search.
Use nimble_extract only for specific article URLs, not generic retailer pages.

Available tools for research steps (use exact param names):
- nimble_search: params must include "query" (single string), optional "focus" and "max_results". Use focus "shopping" for product pricing queries.
- nimble_extract: params must include "url" (single string URL). Only use for specific article/product URLs, not category pages.
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
        wsa_lines = []
        for s in wsa_matches[:10]:
            if s.composite_score >= 0.3:
                line = f"- {s.agent_name} (score: {s.composite_score:.2f}, domain: {s.agent_domain or '?'}, type: {s.agent_entity_type or '?'})"
                if s.agent_description:
                    line += f"\n    {s.agent_description[:120]}"
                if s.input_params_hint:
                    line += f"\n    Input params: {s.input_params_hint}"
                elif s.input_properties:
                    params = ", ".join(f'"{k}"' for k in list(s.input_properties.keys())[:5])
                    line += f"\n    Input params: {params}"
                wsa_lines.append(line)
        if wsa_lines:
            wsa_info = "\n\nAvailable WSA agents (structured web extractors — use for high-quality data):\n" + "\n".join(wsa_lines)
            wsa_info += """

WSA USAGE RULES:
- CRITICAL: Use the EXACT param names listed above for each agent (e.g., "keyword" not "query")
- For SERP agents: use "keyword" param with the search term
- For PDP agents: use "url" param with the product page URL
- Set wsa_agent_name in each WSA step to the agent template name

TWO-WAVE EXECUTION PATTERN:
Wave 1 — Run ALL of these in parallel (no depends_on):
  - nimble_search steps (broad web queries)
  - SERP WSA steps (e.g., walmart_serp with keyword="Cheerios")
  Generate MANY parallel calls: one per query × agent combination
  Example: 4 retailers × 5 products = 20 SERP WSA steps + 10 web searches = 30 parallel steps

Wave 2 — Run AFTER Wave 1 (set depends_on to Wave 1 step IDs):
  - PDP WSA steps using URLs/IDs discovered in Wave 1
  - Only add if PDP WSAs are available for the relevant domains

WSAs have HIGH CONCURRENCY — maximize parallel calls.
WSAs and regular searches complement each other — always run both in Wave 1."""

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


# --- Evidence Sufficiency Assessment ---

SUFFICIENCY_PROMPT = """You are a research quality assessor. Given a research objective and a summary of
evidence collected so far, determine whether the evidence is SUFFICIENT to produce a high-quality answer.

Respond by calling `assess` with your judgment. Consider:
- Does the evidence cover the key dimensions of the question?
- Are there major gaps (e.g., missing geographies, categories, sources)?
- For "find all" or "list" questions: is the sample size large enough?
- For analytical questions: is there enough data to draw conclusions?

If NOT sufficient, suggest 5-10 specific follow-up search queries that would fill the gaps.
The queries should explore NEW dimensions not already covered."""


async def assess_evidence_sufficiency(
    config: SessionConfig,
    skill: DynamicSkillSpec,
    evidence: list[EvidenceItem],
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Ask the LLM whether collected evidence is sufficient to answer the question."""
    registry = ToolRegistry()
    _result: dict[str, Any] = {"sufficient": True, "reason": "default", "suggested_queries": []}

    async def handle_assess(params: dict[str, Any]) -> dict[str, Any]:
        nonlocal _result
        _result = {
            "sufficient": params.get("sufficient", True),
            "reason": params.get("reason", ""),
            "suggested_queries": params.get("suggested_queries", []),
        }
        return {"status": "ok"}

    registry.register(
        ToolDefinition(
            name="assess",
            description="Submit your evidence sufficiency assessment.",
            input_schema={
                "type": "object",
                "properties": {
                    "sufficient": {"type": "boolean", "description": "True if evidence is enough to answer well"},
                    "reason": {"type": "string", "description": "Why sufficient or not"},
                    "suggested_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "If not sufficient: 5-10 follow-up search queries to fill gaps",
                    },
                },
                "required": ["sufficient", "reason"],
            },
            handler=handle_assess,
        )
    )

    # Compact evidence summary — domains + sample titles (keep token count low)
    domains = {}
    for e in evidence:
        d = e.source_domain or "unknown"
        domains[d] = domains.get(d, 0) + 1
    domain_summary = ", ".join(f"{d} ({c})" for d, c in sorted(domains.items(), key=lambda x: -x[1])[:15])
    sample_titles = "\n".join(f"- {e.title or e.source_url}" for e in evidence[:15])

    user_prompt = f"""Research objective: {skill.user_objective}

Evidence: {len(evidence)} items from {len(domains)} unique domains
Top domains: {domain_summary}

Sample (first 15):
{sample_titles}

Is this sufficient? Call `assess`."""

    kwargs = {}
    if fast_mode:
        kwargs["model"] = FAST_MODEL
    await run_agent_loop(
        system_prompt=SUFFICIENCY_PROMPT,
        user_prompt=user_prompt,
        registry=registry,
        tool_names=["assess"],
        max_turns=2,
        max_tokens=2048,
        **kwargs,
    )
    return _result


# --- Follow-up Plan Generator ---

FOLLOWUP_PROMPT = """You are a research deepening planner. Given a research objective and a summary of what
has already been collected, generate NEW search queries that explore UNCOVERED dimensions
of the search space.

Rules:
- Never repeat queries that already produced results
- Decompose the search space along whatever dimensions are natural for this domain
- Use varied query formulations (synonyms, related terms, alternative phrasings)
- Generate 20-30 search steps per round for maximum coverage
- Each step should target a unique sub-area not yet covered
- Always set max_results to 20 or higher for maximum coverage per query
- For queries about places, locations, or geographic entities, use focus: "location" or "geo"
- Use ONLY nimble_search steps for follow-up rounds (nimble_extract is too slow for bulk collection)
- Exception: include 2-3 nimble_map steps ONLY for directory/listing sites found in evidence (e.g., healthgrades.com, yelp.com, zocdoc.com) to discover paginated listing URLs

Available tools (use exact param names):
- nimble_search: params must include "query" (single string), "max_results" (integer, use 20+), optional "focus" ("general", "location", "geo", "news", "shopping")
- nimble_map: params must include "url" (single string URL) — use sparingly, only for directory sites

IMPORTANT: Each step's params must use singular "query" (not "queries") and "url" (not "urls").

Call `submit_plan` with your follow-up steps."""


async def create_followup_plan(
    config: SessionConfig,
    skill: DynamicSkillSpec,
    evidence: list[EvidenceItem],
    suggested_queries: Optional[list[str]] = None,
    fast_mode: bool = False,
    iteration: int = 0,
) -> Optional[ResearchPlan]:
    """Generate a follow-up research plan to deepen evidence collection."""
    global _plan_result
    _plan_result = None

    registry = ToolRegistry()

    async def handle_submit(params: dict[str, Any]) -> dict[str, Any]:
        global _plan_result
        steps = []
        for i, s in enumerate(params.get("steps", [])):
            tool_name = s.get("tool", "nimble_search")
            try:
                tool = ToolName(tool_name)
            except ValueError:
                tool = ToolName.SEARCH
            steps.append(
                PlanStep(
                    order=i,
                    description=s.get("description", f"Step {i}"),
                    tool=tool,
                    params=s.get("params", {}),
                    timeout_seconds=s.get("timeout_seconds", 30),
                )
            )
        _plan_result = ResearchPlan(
            session_id=config.session_id,
            skill_id=skill.skill_id,
            objective=f"Deepen: {skill.user_objective}",
            subquestions=[],
            steps=steps,
        )
        return {"status": "ok", "steps": len(steps)}

    registry.register(
        ToolDefinition(
            name="submit_plan",
            description="Submit follow-up research plan with new steps.",
            input_schema={
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "tool": {"type": "string", "enum": [t.value for t in ToolName]},
                                "params": {"type": "object"},
                                "timeout_seconds": {"type": "integer", "default": 30},
                            },
                            "required": ["description", "tool", "params"],
                        },
                    },
                },
                "required": ["steps"],
            },
            handler=handle_submit,
        )
    )

    # Build evidence summary
    domains = {}
    for e in evidence:
        d = e.source_domain or "unknown"
        domains[d] = domains.get(d, 0) + 1
    domain_summary = ", ".join(f"{d} ({c})" for d, c in sorted(domains.items(), key=lambda x: -x[1])[:20])

    suggestions = ""
    if suggested_queries:
        suggestions = "\nSuggested follow-up queries from assessment:\n" + "\n".join(f"- {q}" for q in suggested_queries)

    rec_max = 20 if iteration < 2 else 30
    user_prompt = f"""Research objective: {skill.user_objective}

This is deepening round {iteration + 1}. Use max_results: {rec_max} for all search steps.

Already collected: {len(evidence)} evidence items from these domains:
{domain_summary}
{suggestions}

Generate 20-30 NEW search steps that explore dimensions NOT yet covered.
Call `submit_plan` with your steps."""

    kwargs = {}
    if fast_mode:
        kwargs["model"] = FAST_MODEL
    await run_agent_loop(
        system_prompt=FOLLOWUP_PROMPT,
        user_prompt=user_prompt,
        registry=registry,
        tool_names=["submit_plan"],
        max_turns=2,
        max_tokens=4096,
        **kwargs,
    )
    return _plan_result
