"""Session and time-budget models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enums import ExecutionMode, ReportFormat, TaskType, TimeBudget


class StopConditions(BaseModel):
    model_config = ConfigDict(frozen=True)

    wall_clock_seconds: int = Field(default=600, ge=10, le=7200)
    min_evidence_items: int = Field(default=5, ge=0)
    min_sources: int = Field(default=3, ge=0)
    min_claims_verified: int = Field(default=0, ge=0)


class TimeBudgetPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    budget: TimeBudget
    planning_depth: int = Field(ge=1, le=5)
    search_breadth: int = Field(ge=1, le=50)
    max_searches: int = Field(ge=1, le=200)
    max_extracts: int = Field(ge=0, le=500)
    max_crawl_pages: int = Field(ge=0, le=5000)
    concurrency_limit: int = Field(ge=1, le=30)
    verification_strictness: int = Field(ge=0, le=3)
    max_refinement_loops: int = Field(ge=0, le=5)
    stop_conditions: StopConditions = Field(default_factory=StopConditions)

    @classmethod
    def from_budget(cls, budget: TimeBudget) -> TimeBudgetPolicy:
        presets: dict[TimeBudget, dict] = {
            TimeBudget.QUICK_30S: dict(
                planning_depth=1,
                search_breadth=2,
                max_searches=3,
                max_extracts=1,
                max_crawl_pages=0,
                concurrency_limit=3,
                verification_strictness=0,
                max_refinement_loops=0,
                stop_conditions=StopConditions(
                    wall_clock_seconds=30, min_evidence_items=1, min_sources=1
                ),
            ),
            TimeBudget.SHORT_2M: dict(
                planning_depth=2,
                search_breadth=5,
                max_searches=8,
                max_extracts=5,
                max_crawl_pages=0,
                concurrency_limit=5,
                verification_strictness=1,
                max_refinement_loops=1,
                stop_conditions=StopConditions(
                    wall_clock_seconds=120, min_evidence_items=3, min_sources=2
                ),
            ),
            TimeBudget.MEDIUM_5M: dict(
                planning_depth=3,
                search_breadth=10,
                max_searches=20,
                max_extracts=15,
                max_crawl_pages=50,
                concurrency_limit=8,
                verification_strictness=2,
                max_refinement_loops=2,
                stop_conditions=StopConditions(
                    wall_clock_seconds=300,
                    min_evidence_items=5,
                    min_sources=3,
                    min_claims_verified=2,
                ),
            ),
            TimeBudget.STANDARD_10M: dict(
                planning_depth=4,
                search_breadth=20,
                max_searches=40,
                max_extracts=30,
                max_crawl_pages=200,
                concurrency_limit=20,
                verification_strictness=2,
                max_refinement_loops=3,
                stop_conditions=StopConditions(
                    wall_clock_seconds=600,
                    min_evidence_items=10,
                    min_sources=5,
                    min_claims_verified=5,
                ),
            ),
            TimeBudget.DEEP_30M: dict(
                planning_depth=5,
                search_breadth=35,
                max_searches=100,
                max_extracts=80,
                max_crawl_pages=1000,
                concurrency_limit=25,
                verification_strictness=3,
                max_refinement_loops=4,
                stop_conditions=StopConditions(
                    wall_clock_seconds=1800,
                    min_evidence_items=20,
                    min_sources=10,
                    min_claims_verified=10,
                ),
            ),
            TimeBudget.EXHAUSTIVE_1H: dict(
                planning_depth=5,
                search_breadth=50,
                max_searches=200,
                max_extracts=500,
                max_crawl_pages=5000,
                concurrency_limit=30,
                verification_strictness=3,
                max_refinement_loops=5,
                stop_conditions=StopConditions(
                    wall_clock_seconds=3600,
                    min_evidence_items=50,
                    min_sources=20,
                    min_claims_verified=20,
                ),
            ),
        }
        return cls(budget=budget, **presets[budget])


class UserResearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    request_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_query: str = Field(min_length=3, max_length=5000)
    time_budget: TimeBudget = Field(default=TimeBudget.MEDIUM_5M)
    preferred_format: ReportFormat = Field(default=ReportFormat.FULL_REPORT)
    preferred_mode: Optional[ExecutionMode] = None
    context_hints: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # New fields inspired by Parallel AI
    output_schema: Optional[dict] = Field(
        default=None, description="Optional JSON schema for structured output"
    )
    source_policy: Optional[dict] = Field(
        default=None,
        description="Source policy with preferred_domains/disallowed_domains",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="User-provided tracking metadata"
    )
    previous_session_id: Optional[str] = Field(
        default=None, description="For follow-up research (context chaining)"
    )
    fast_mode: bool = Field(
        default=False, description="Trade freshness for speed (cached results OK)"
    )

    @field_validator("user_query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_query must contain non-whitespace characters")
        return v


class SessionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    request_id: uuid.UUID
    user_query: str
    normalized_objective: str
    time_budget: TimeBudget
    policy: TimeBudgetPolicy
    execution_mode: ExecutionMode = Field(default=ExecutionMode.HYBRID)
    report_format: ReportFormat = Field(default=ReportFormat.FULL_REPORT)
    task_type: TaskType = Field(default=TaskType.OPEN_EXPLORATION)
    target_domains: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
