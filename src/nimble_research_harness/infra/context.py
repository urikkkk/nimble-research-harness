"""Run context using contextvars for async safety."""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..storage.backend import StorageBackend

_current_context: ContextVar[Optional["RunContext"]] = ContextVar(
    "run_context", default=None
)


@dataclass
class RunContext:
    session_id: uuid.UUID
    storage: StorageBackend
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def elapsed_seconds(self) -> float:
        delta = datetime.now(timezone.utc) - self.started_at
        return delta.total_seconds()


def set_context(ctx: RunContext) -> None:
    _current_context.set(ctx)


def get_context() -> RunContext:
    ctx = _current_context.get()
    if ctx is None:
        raise RuntimeError("No RunContext set — call set_context() first")
    return ctx
