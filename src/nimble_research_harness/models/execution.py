"""Execution tracking models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from .enums import ExecutionStage, ToolCallStatus, ToolName


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    tool: ToolName
    params: dict[str, Any] = Field(default_factory=dict)
    status: ToolCallStatus = Field(default=ToolCallStatus.SUCCESS)
    response_summary: Optional[str] = None
    result_count: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RunCheckpoint(BaseModel):
    checkpoint_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    stage: ExecutionStage
    stage_index: int = Field(ge=0)
    completed_steps: int = Field(default=0, ge=0)
    total_steps: int = Field(default=0, ge=0)
    tool_calls_made: int = Field(default=0, ge=0)
    evidence_count: int = Field(default=0, ge=0)
    source_count: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def progress_pct(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return round(self.completed_steps / self.total_steps * 100, 1)
