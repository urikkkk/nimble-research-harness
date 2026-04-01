"""Report and session summary output models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .enums import ExecutionMode, ExecutionStage, ReportFormat, TimeBudget
from .evidence import Claim, EvidenceItem, VerificationResult


class ResearchReport(BaseModel):
    report_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    title: str
    format: ReportFormat = Field(default=ReportFormat.FULL_REPORT)

    executive_summary: str = ""
    key_findings: list[str] = Field(default_factory=list)
    detailed_analysis: str = ""
    methodology: str = ""

    claims: list[Claim] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    verifications: list[VerificationResult] = Field(default_factory=list)

    sources: list[dict[str, Any]] = Field(default_factory=list)
    known_unknowns: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def confidence_rating(self) -> str:
        if not self.claims:
            return "low"
        verified = sum(1 for c in self.claims if c.confidence.value == "verified")
        ratio = verified / len(self.claims)
        if ratio >= 0.7:
            return "high"
        elif ratio >= 0.4:
            return "medium"
        return "low"

    @computed_field
    @property
    def total_sources(self) -> int:
        return len(self.sources)


class SessionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: uuid.UUID
    user_query: str
    time_budget: TimeBudget
    execution_mode: ExecutionMode
    final_stage: ExecutionStage
    total_tool_calls: int = Field(default=0, ge=0)
    total_evidence: int = Field(default=0, ge=0)
    total_sources: int = Field(default=0, ge=0)
    total_claims: int = Field(default=0, ge=0)
    verified_claims: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    report_confidence: Optional[str] = None
    skill_title: Optional[str] = None
    wsa_agents_used: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def budget_utilization_pct(self) -> float:
        budget_seconds = self.time_budget.seconds
        if budget_seconds == 0:
            return 0.0
        return round(min(self.elapsed_seconds / budget_seconds * 100, 100.0), 1)
