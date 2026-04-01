"""Research plan models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .enums import ExecutionMode, ToolName


class PlanStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    order: int = Field(ge=0)
    description: str
    tool: ToolName
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[uuid.UUID] = Field(default_factory=list)
    expected_output: Optional[str] = None
    wsa_agent_name: Optional[str] = None
    timeout_seconds: int = Field(default=60, ge=5)


class ResearchPlan(BaseModel):
    plan_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    skill_id: uuid.UUID
    objective: str
    subquestions: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    execution_mode: ExecutionMode = Field(default=ExecutionMode.HYBRID)
    wsa_agents: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    revision: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_steps(self) -> int:
        return len(self.steps)
