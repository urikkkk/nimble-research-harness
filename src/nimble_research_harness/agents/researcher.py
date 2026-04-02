"""Researcher agent — parallel execution of research plan steps."""

from __future__ import annotations

import asyncio
from typing import Any

from ..infra.logging import get_logger
from ..models.plan import ResearchPlan
from ..models.session import SessionConfig
from ..tools.registry import ToolRegistry

logger = get_logger(__name__)

_VALID_FOCUS = {"general", "news", "coding", "academic", "shopping", "social", "geo", "location"}


def _normalize_step_params(tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fix common parameter mistakes from the planner LLM."""
    if "queries" in params and "query" not in params:
        val = params.pop("queries")
        params["query"] = val[0] if isinstance(val, list) else val
    if "urls" in params and "url" not in params:
        val = params.pop("urls")
        params["url"] = val[0] if isinstance(val, list) else val
    if isinstance(params.get("query"), list):
        params["query"] = params["query"][0] if params["query"] else ""
    if isinstance(params.get("url"), list):
        params["url"] = params["url"][0] if params["url"] else ""
    if tool_name == "nimble_search" and params.get("focus") not in _VALID_FOCUS:
        params["focus"] = "general"
    return params


async def execute_research(
    config: SessionConfig,
    plan: ResearchPlan,
    registry: ToolRegistry,
) -> dict[str, Any]:
    """Execute research plan steps with bounded concurrency."""
    semaphore = asyncio.Semaphore(config.policy.concurrency_limit)
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    async def run_step(step):
        async with semaphore:
            logger.info(
                "executing_step",
                order=step.order,
                tool=step.tool.value,
                description=step.description[:80],
            )
            try:
                params = dict(step.params)
                if step.wsa_agent_name:
                    params["agent_name"] = step.wsa_agent_name
                params = _normalize_step_params(step.tool.value, params)

                result = await asyncio.wait_for(
                    registry.dispatch(step.tool.value, params),
                    timeout=step.timeout_seconds,
                )
                results.append({
                    "step_order": step.order,
                    "tool": step.tool.value,
                    "result": result,
                })
            except asyncio.TimeoutError:
                errors.append(f"Step {step.order} ({step.tool.value}) timed out")
                logger.warning("step_timeout", order=step.order, tool=step.tool.value)
            except Exception as e:
                errors.append(f"Step {step.order} ({step.tool.value}): {str(e)}")
                logger.error("step_error", order=step.order, error=str(e))

    # Group steps by dependency — steps with no deps run in parallel
    independent = [s for s in plan.steps if not s.depends_on]
    dependent = [s for s in plan.steps if s.depends_on]

    # Run independent steps concurrently
    if independent:
        async with asyncio.TaskGroup() as tg:
            for step in independent:
                tg.create_task(run_step(step))

    # Run dependent steps sequentially
    for step in dependent:
        await run_step(step)

    return {
        "completed": len(results),
        "errors": len(errors),
        "error_details": errors,
        "total_steps": plan.total_steps,
    }
