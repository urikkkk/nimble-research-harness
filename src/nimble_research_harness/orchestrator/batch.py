"""Batch research — fan-out multiple queries with aggregated progress."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..infra.events import EventStream
from ..infra.logging import get_logger
from ..models.enums import TimeBudget
from ..models.output import SessionSummary
from ..models.session import UserResearchRequest
from ..nimble.provider import NimbleProvider
from ..storage.backend import StorageBackend
from .engine import run_research

logger = get_logger(__name__)


@dataclass
class BatchResult:
    """Aggregated result of a batch research run."""

    batch_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    total_queries: int = 0
    completed: int = 0
    failed: int = 0
    summaries: list[SessionSummary] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.completed / self.total_queries * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "total_queries": self.total_queries,
            "completed": self.completed,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 1),
            "summaries": [s.model_dump(mode="json") for s in self.summaries],
            "errors": self.errors,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


async def batch_research(
    queries: list[str],
    time_budget: TimeBudget,
    provider: NimbleProvider,
    storage: StorageBackend,
    concurrency: int = 5,
    event_stream: EventStream | None = None,
    metadata: dict[str, str] | None = None,
) -> BatchResult:
    """Execute multiple research queries concurrently with bounded concurrency.

    Args:
        queries: List of research queries to execute
        time_budget: Time budget for each individual query
        provider: Nimble API provider
        storage: Storage backend
        concurrency: Maximum concurrent research sessions
        event_stream: Optional event stream for progress updates
        metadata: Optional metadata applied to all queries
    """
    result = BatchResult(total_queries=len(queries))
    semaphore = asyncio.Semaphore(concurrency)

    if event_stream:
        await event_stream.emit("batch.started", {
            "batch_id": result.batch_id,
            "total_queries": len(queries),
            "concurrency": concurrency,
        })

    async def _run_one(idx: int, query: str) -> None:
        async with semaphore:
            try:
                if event_stream:
                    await event_stream.emit("batch.query_started", {
                        "index": idx,
                        "query": query[:200],
                    })

                request = UserResearchRequest(
                    user_query=query,
                    time_budget=time_budget,
                    metadata=metadata or {},
                )
                summary = await run_research(request, provider, storage)
                result.summaries.append(summary)
                result.completed += 1

                if event_stream:
                    await event_stream.emit("batch.query_completed", {
                        "index": idx,
                        "session_id": str(summary.session_id),
                        "confidence": summary.report_confidence,
                    })

            except Exception as e:
                result.failed += 1
                result.errors.append({"query": query, "error": str(e)})
                logger.error("batch_query_failed", index=idx, query=query[:100], error=str(e))

                if event_stream:
                    await event_stream.emit("batch.query_failed", {
                        "index": idx,
                        "error": str(e),
                    })

    tasks = [_run_one(i, q) for i, q in enumerate(queries)]
    await asyncio.gather(*tasks)

    result.completed_at = datetime.now(timezone.utc)

    if event_stream:
        await event_stream.emit("batch.completed", {
            "batch_id": result.batch_id,
            "completed": result.completed,
            "failed": result.failed,
            "success_rate": result.success_rate,
        })

    logger.info(
        "batch_complete",
        batch_id=result.batch_id,
        total=result.total_queries,
        completed=result.completed,
        failed=result.failed,
    )

    return result
