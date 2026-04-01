"""SSE event streaming for real-time progress updates."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResearchEvent:
    """A single event emitted during research execution."""

    event_type: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        payload = {
            "type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }
        return f"data: {json.dumps(payload, default=str)}\n\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class EventStream:
    """Async event stream for research progress.

    Supports both direct consumption (async iteration) and SSE formatting.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._queue: asyncio.Queue[ResearchEvent | None] = asyncio.Queue()
        self._history: list[ResearchEvent] = []
        self._closed = False

    async def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Emit an event to all listeners."""
        if self._closed:
            return
        event = ResearchEvent(event_type=event_type, data=data or {})
        self._history.append(event)
        await self._queue.put(event)
        logger.debug("event_emitted", type=event_type, session=self.session_id)

    async def close(self) -> None:
        """Signal end of stream."""
        self._closed = True
        await self._queue.put(None)

    async def listen(self) -> AsyncIterator[ResearchEvent]:
        """Async iterator that yields events as they arrive."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    async def listen_sse(self) -> AsyncIterator[str]:
        """Async iterator that yields SSE-formatted strings."""
        async for event in self.listen():
            yield event.to_sse()

    @property
    def history(self) -> list[ResearchEvent]:
        return list(self._history)

    # --- Convenience emitters for common events ---

    async def session_started(self, query: str, budget: str) -> None:
        await self.emit("session.started", {"query": query, "budget": budget})

    async def stage_entered(self, stage: str, elapsed: float = 0.0) -> None:
        await self.emit("stage.entered", {"stage": stage, "elapsed_seconds": round(elapsed, 1)})

    async def tool_called(self, tool: str, params: dict[str, Any] | None = None) -> None:
        safe_params = {k: str(v)[:200] for k, v in (params or {}).items()}
        await self.emit("tool.called", {"tool": tool, "params": safe_params})

    async def tool_completed(
        self, tool: str, summary: str, latency_ms: int = 0
    ) -> None:
        await self.emit(
            "tool.completed",
            {"tool": tool, "summary": summary, "latency_ms": latency_ms},
        )

    async def finding_added(self, summary: str) -> None:
        await self.emit("finding.added", {"summary": summary[:500]})

    async def claim_verified(self, claim: str, confidence: str) -> None:
        await self.emit(
            "claim.verified", {"claim": claim[:300], "confidence": confidence}
        )

    async def budget_warning(self, remaining_pct: float) -> None:
        await self.emit("budget.warning", {"remaining_pct": round(remaining_pct, 1)})

    async def session_completed(self, summary: dict[str, Any]) -> None:
        await self.emit("session.completed", summary)
        await self.close()

    async def session_failed(self, error: str) -> None:
        await self.emit("session.failed", {"error": error})
        await self.close()
