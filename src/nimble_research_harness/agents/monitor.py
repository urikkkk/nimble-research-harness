"""Monitor agent — budget watchdog and progress tracking."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..infra.context import get_context
from ..infra.logging import get_logger
from ..models.enums import ExecutionStage
from ..models.execution import RunCheckpoint
from ..models.session import SessionConfig

logger = get_logger(__name__)


class BudgetMonitor:
    """Tracks execution progress and enforces time budget."""

    def __init__(self, config: SessionConfig):
        self.config = config
        self.start_time = time.time()
        self.wall_clock_limit = config.policy.stop_conditions.wall_clock_seconds
        self._cancelled = False
        self._current_stage = ExecutionStage.INTAKE

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def remaining_seconds(self) -> float:
        return max(0, self.wall_clock_limit - self.elapsed_seconds)

    @property
    def is_over_budget(self) -> bool:
        return self.elapsed_seconds >= self.wall_clock_limit

    @property
    def budget_utilization_pct(self) -> float:
        return min(100.0, self.elapsed_seconds / self.wall_clock_limit * 100)

    def set_stage(self, stage: ExecutionStage) -> None:
        self._current_stage = stage
        logger.info(
            "stage_transition",
            stage=stage.value,
            elapsed=f"{self.elapsed_seconds:.1f}s",
            remaining=f"{self.remaining_seconds:.1f}s",
            utilization=f"{self.budget_utilization_pct:.0f}%",
        )

    def cancel(self) -> None:
        self._cancelled = True

    async def create_checkpoint(
        self,
        stage: ExecutionStage,
        stage_index: int,
        completed_steps: int = 0,
        total_steps: int = 0,
        tool_calls: int = 0,
        evidence_count: int = 0,
        source_count: int = 0,
        **metadata: Any,
    ) -> RunCheckpoint:
        ctx = get_context()
        checkpoint = RunCheckpoint(
            session_id=ctx.session_id,
            stage=stage,
            stage_index=stage_index,
            completed_steps=completed_steps,
            total_steps=total_steps,
            tool_calls_made=tool_calls,
            evidence_count=evidence_count,
            source_count=source_count,
            elapsed_seconds=self.elapsed_seconds,
            metadata=metadata,
        )
        await ctx.storage.save_checkpoint(checkpoint)
        return checkpoint

    def should_skip_stage(self, stage: ExecutionStage) -> bool:
        """Check if we should skip a stage due to budget constraints."""
        pct = self.budget_utilization_pct
        if stage == ExecutionStage.VERIFICATION and pct > 90:
            return True
        if stage == ExecutionStage.EXTRACTION and pct > 80:
            return True
        return self._cancelled
