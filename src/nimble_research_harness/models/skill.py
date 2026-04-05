"""Dynamic skill specification models."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    DeploymentStatus,
    ExecutionMode,
    ReportFormat,
    SearchFocus,
    TaskType,
    TimeBudget,
    ToolName,
)


class PlanningPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    depth: int = Field(default=3, ge=1, le=5)
    max_subquestions: int = Field(default=5, ge=1, le=20)
    require_entity_extraction: bool = True
    allow_replanning: bool = True


class SourcePolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_sources: int = Field(default=3, ge=1)
    require_diversity: bool = True
    domain_include: list[str] = Field(default_factory=list)
    domain_exclude: list[str] = Field(default_factory=list)
    freshness_days: Optional[int] = None
    preferred_focus_modes: list[SearchFocus] = Field(
        default_factory=lambda: [SearchFocus.GENERAL]
    )
    # Enhanced: Parallel AI-inspired domain controls
    preferred_domains: list[str] = Field(
        default_factory=list, description="Domains weighted higher in ranking"
    )
    disallowed_domains: list[str] = Field(
        default_factory=list, description="Domains hard-blocked from all tool calls"
    )


class ExtractionPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    extract_mode: str = Field(default="markdown")
    max_content_length: int = Field(default=10000, ge=500)
    structured_fields: list[str] = Field(default_factory=list)
    render_js: bool = False


class SynthesisPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    style: str = Field(default="analytical")
    max_findings: int = Field(default=20, ge=1)
    require_comparisons: bool = False
    group_by_theme: bool = True


class VerificationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    strictness: int = Field(default=2, ge=0, le=3)
    require_corroboration: bool = False
    flag_contradictions: bool = True
    max_verification_searches: int = Field(default=5, ge=0)
    confidence_threshold: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Findings below this threshold get flagged, not included",
    )


class ReportPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    format: ReportFormat = Field(default=ReportFormat.FULL_REPORT)
    sections: list[str] = Field(
        default_factory=lambda: [
            "executive_summary",
            "key_findings",
            "detailed_analysis",
            "evidence",
            "methodology",
        ]
    )
    include_evidence_table: bool = True
    include_source_list: bool = True
    include_confidence_ratings: bool = True


class SearchStrategy(BaseModel):
    model_config = ConfigDict(frozen=True)

    queries: list[str] = Field(default_factory=list)
    focus_modes: list[SearchFocus] = Field(
        default_factory=lambda: [SearchFocus.GENERAL]
    )
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    max_results_per_query: int = Field(default=10, ge=1, le=100)


class ExtractionStrategy(BaseModel):
    model_config = ConfigDict(frozen=True)

    priority_urls: list[str] = Field(default_factory=list)
    extract_format: str = Field(default="markdown")
    use_map_first: bool = False
    crawl_targets: list[str] = Field(default_factory=list)


class DynamicSkillSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    title: str
    slug: str = Field(default="", description="URL-safe slug derived from title")

    @model_validator(mode="before")
    @classmethod
    def _derive_slug(cls, data: dict) -> dict:
        if isinstance(data, dict) and not data.get("slug") and data.get("title"):
            raw = re.sub(r"[^a-z0-9]+", "-", data["title"].lower()).strip("-")[:60]
            data["slug"] = raw
        return data
    user_objective: str
    task_type: TaskType
    time_budget: TimeBudget

    subquestions: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    likely_source_types: list[str] = Field(default_factory=list)

    planning_policy: PlanningPolicy = Field(default_factory=PlanningPolicy)
    source_policy: SourcePolicy = Field(default_factory=SourcePolicy)
    extraction_policy: ExtractionPolicy = Field(default_factory=ExtractionPolicy)
    synthesis_policy: SynthesisPolicy = Field(default_factory=SynthesisPolicy)
    verification_policy: VerificationPolicy = Field(default_factory=VerificationPolicy)
    report_policy: ReportPolicy = Field(default_factory=ReportPolicy)

    search_strategy: SearchStrategy = Field(default_factory=SearchStrategy)
    extraction_strategy: ExtractionStrategy = Field(default_factory=ExtractionStrategy)

    tool_permissions: list[ToolName] = Field(
        default_factory=lambda: [
            ToolName.SEARCH,
            ToolName.EXTRACT,
            ToolName.MAP,
            ToolName.AGENTS_RUN,
        ]
    )
    execution_mode: ExecutionMode = Field(default=ExecutionMode.HYBRID)
    deliverables: list[str] = Field(default_factory=lambda: ["research_report"])
    output_schema: Optional[dict] = Field(
        default=None, description="User-provided JSON schema for structured output"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeploymentRecord(BaseModel):
    deployment_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    skill_id: uuid.UUID
    session_id: uuid.UUID
    status: DeploymentStatus = Field(default=DeploymentStatus.PENDING)
    registered_tools: list[str] = Field(default_factory=list)
    deployed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
