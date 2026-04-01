"""Intake agent — classifies query and normalizes the task."""

from __future__ import annotations

from ..budget.presets import suggest_budget
from ..models.enums import ExecutionMode, TaskType, TimeBudget
from ..models.session import SessionConfig, TimeBudgetPolicy, UserResearchRequest


def classify_task_type(query: str) -> TaskType:
    """Simple heuristic classification of the research task."""
    q = query.lower()
    if any(w in q for w in ["price", "cost", "pricing", "buy", "product"]):
        return TaskType.DATA_COLLECTION
    if any(w in q for w in ["competitor", "versus", "vs", "compare", "alternative"]):
        return TaskType.COMPETITIVE_INTEL
    if any(w in q for w in ["market", "industry", "sector", "landscape"]):
        return TaskType.MARKET_RESEARCH
    if any(w in q for w in ["company", "startup", "who is", "about"]):
        return TaskType.COMPANY_DEEP_DIVE
    if any(w in q for w in ["trend", "forecast", "growth", "emerging"]):
        return TaskType.TREND_ANALYSIS
    if any(w in q for w in ["true", "false", "claim", "verify", "fact"]):
        return TaskType.VERIFICATION
    if any(w in q for w in ["what is", "define", "explain", "who"]):
        return TaskType.FACTUAL_LOOKUP
    return TaskType.OPEN_EXPLORATION


def extract_target_domains(query: str) -> list[str]:
    """Extract domain hints from query."""
    import re
    urls = re.findall(r'(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})', query)
    return urls


def normalize_request(request: UserResearchRequest) -> SessionConfig:
    """Convert raw user request into normalized session config."""
    policy = TimeBudgetPolicy.from_budget(request.time_budget)
    task_type = classify_task_type(request.user_query)
    domains = extract_target_domains(request.user_query) + [
        h for h in request.context_hints if "." in h
    ]

    objective = request.user_query.strip()
    if not objective.endswith((".", "?", "!")):
        objective += "."

    return SessionConfig(
        request_id=request.request_id,
        user_query=request.user_query,
        normalized_objective=objective,
        time_budget=request.time_budget,
        policy=policy,
        execution_mode=request.preferred_mode or ExecutionMode.HYBRID,
        report_format=request.preferred_format,
        task_type=task_type,
        target_domains=domains,
    )
