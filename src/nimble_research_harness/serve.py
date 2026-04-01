"""FastAPI wrapper — enhanced with SSE streaming, batch, and follow-up endpoints."""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .infra.events import EventStream
from .infra.logging import setup_logging
from .models.enums import ReportFormat, TimeBudget
from .models.session import UserResearchRequest
from .nimble.client import NimbleClient
from .nimble.mock import MockNimbleProvider
from .orchestrator.batch import batch_research
from .orchestrator.engine import run_research
from .orchestrator.followup import follow_up_research
from .reports.formatter import format_report, format_summary
from .storage.json_backend import JsonStorageBackend

load_dotenv()
setup_logging()

app = FastAPI(title="Nimble Research Harness", version="0.2.0")


# --- Request/Response Models ---


class ResearchRequest(BaseModel):
    query: str
    time_budget: str = "5m"
    report_format: str = "full_report"
    mock: bool = False
    fast_mode: bool = False
    output_schema: Optional[dict] = None
    source_policy: Optional[dict] = None
    metadata: dict[str, str] = Field(default_factory=dict)


class FollowUpRequest(BaseModel):
    query: str
    time_budget: str = "5m"


class BatchRequest(BaseModel):
    queries: list[str]
    time_budget: str = "5m"
    concurrency: int = 5
    mock: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)


# --- Helpers ---


def _get_provider(mock: bool = False):
    if mock:
        return MockNimbleProvider()
    api_key = os.environ.get("NIMBLE_API_KEY", "")
    if not api_key:
        return MockNimbleProvider()
    return NimbleClient(api_key=api_key)


# --- Research Endpoints ---


@app.post("/v1/research")
async def start_research(req: ResearchRequest):
    """Start a research session. Returns session summary and report."""
    request = UserResearchRequest(
        user_query=req.query,
        time_budget=TimeBudget(req.time_budget),
        preferred_format=ReportFormat(req.report_format),
        fast_mode=req.fast_mode,
        output_schema=req.output_schema,
        source_policy=req.source_policy,
        metadata=req.metadata,
    )
    provider = _get_provider(req.mock)
    storage = JsonStorageBackend()

    summary = await run_research(request, provider, storage)
    report = await storage.load_report(str(summary.session_id))

    return {
        "session_id": str(summary.session_id),
        "summary": summary.model_dump(mode="json"),
        "report": format_report(report, ReportFormat(req.report_format)) if report else None,
    }


@app.get("/v1/research/{session_id}")
async def get_research_status(session_id: str):
    """Poll status of a research session."""
    storage = JsonStorageBackend()
    config = await storage.load_session(session_id)
    if not config:
        raise HTTPException(404, "Session not found")

    checkpoint = await storage.load_latest_checkpoint(session_id)
    summary_path = storage._session_dir(session_id) / "summary.json"

    result = {
        "session_id": session_id,
        "query": config.user_query,
        "budget": config.time_budget.value,
        "mode": config.execution_mode.value,
    }

    if checkpoint:
        result["stage"] = checkpoint.stage.value
        result["progress_pct"] = checkpoint.progress_pct
        result["elapsed_seconds"] = checkpoint.elapsed_seconds

    if summary_path.exists():
        result["status"] = "completed"
    else:
        result["status"] = "running" if checkpoint else "queued"

    return result


@app.get("/v1/research/{session_id}/events")
async def stream_research_events(session_id: str):
    """SSE stream of research events for a session."""
    storage = JsonStorageBackend()
    events_path = storage._session_dir(session_id) / "events.json"

    async def _generate():
        data = storage._read_json(events_path)
        if data:
            for event in data:
                import json
                yield f"data: {json.dumps(event, default=str)}\n\n"
        yield "data: {\"type\": \"stream.end\"}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/v1/research/{session_id}/result")
async def get_research_result(session_id: str, format: str = "full_report"):
    """Get completed research result."""
    storage = JsonStorageBackend()
    report = await storage.load_report(session_id)
    if not report:
        raise HTTPException(404, "No report found — research may still be running")

    return {
        "session_id": session_id,
        "report": format_report(report, ReportFormat(format)),
        "report_json": report.model_dump(mode="json"),
        "confidence": report.confidence_rating,
        "total_sources": report.total_sources,
    }


@app.post("/v1/research/{session_id}/follow-up")
async def research_follow_up(session_id: str, req: FollowUpRequest):
    """Run follow-up research building on a prior session."""
    provider = _get_provider()
    storage = JsonStorageBackend()

    try:
        summary = await follow_up_research(
            previous_session_id=session_id,
            new_query=req.query,
            time_budget=TimeBudget(req.time_budget),
            provider=provider,
            storage=storage,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))

    report = await storage.load_report(str(summary.session_id))
    return {
        "session_id": str(summary.session_id),
        "previous_session_id": session_id,
        "summary": summary.model_dump(mode="json"),
        "report": format_report(report, ReportFormat.FULL_REPORT) if report else None,
    }


@app.delete("/v1/research/{session_id}")
async def cancel_research(session_id: str):
    """Cancel a running research session (best-effort)."""
    storage = JsonStorageBackend()
    config = await storage.load_session(session_id)
    if not config:
        raise HTTPException(404, "Session not found")
    return {"session_id": session_id, "status": "cancellation_requested"}


# --- Batch Endpoints ---


@app.post("/v1/research/batch")
async def start_batch_research(req: BatchRequest):
    """Start batch research on multiple queries."""
    provider = _get_provider(req.mock)
    storage = JsonStorageBackend()

    result = await batch_research(
        queries=req.queries,
        time_budget=TimeBudget(req.time_budget),
        provider=provider,
        storage=storage,
        concurrency=req.concurrency,
        metadata=req.metadata,
    )

    return result.to_dict()


# --- Session Endpoints ---


@app.get("/v1/sessions")
async def list_sessions():
    """List all research sessions."""
    storage = JsonStorageBackend()
    return await storage.list_sessions()


@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    storage = JsonStorageBackend()
    config = await storage.load_session(session_id)
    if not config:
        raise HTTPException(404, "Session not found")
    return config.model_dump(mode="json")


@app.get("/v1/sessions/{session_id}/skill")
async def get_skill(session_id: str):
    """Get the generated skill spec for a session."""
    storage = JsonStorageBackend()
    skill = await storage.load_skill(session_id)
    if not skill:
        raise HTTPException(404, "No skill found")
    return skill.model_dump(mode="json")


@app.get("/v1/sessions/{session_id}/plan")
async def get_plan(session_id: str):
    """Get the research plan for a session."""
    storage = JsonStorageBackend()
    plan = await storage.load_plan(session_id)
    if not plan:
        raise HTTPException(404, "No plan found")
    return plan.model_dump(mode="json")


@app.get("/v1/sessions/{session_id}/evidence")
async def get_evidence(session_id: str):
    """Get evidence items for a session."""
    storage = JsonStorageBackend()
    evidence = await storage.get_evidence(session_id)
    return {
        "session_id": session_id,
        "count": len(evidence),
        "evidence": [e.model_dump(mode="json") for e in evidence],
    }


@app.get("/v1/sessions/{session_id}/report")
async def get_report(session_id: str, format: str = "full_report"):
    """Get the research report for a session."""
    storage = JsonStorageBackend()
    report = await storage.load_report(session_id)
    if not report:
        raise HTTPException(404, "No report found")
    return {"report": format_report(report, ReportFormat(format))}
