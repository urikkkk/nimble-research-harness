"""Optional FastAPI wrapper for the research harness."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .infra.logging import setup_logging
from .models.enums import ReportFormat, TimeBudget
from .models.session import UserResearchRequest
from .nimble.client import NimbleClient
from .nimble.mock import MockNimbleProvider
from .orchestrator.engine import run_research
from .reports.formatter import format_report, format_summary
from .storage.json_backend import JsonStorageBackend

load_dotenv()
setup_logging()

app = FastAPI(title="Nimble Research Harness", version="0.1.0")


class ResearchRequest(BaseModel):
    query: str
    time_budget: str = "5m"
    report_format: str = "full_report"
    mock: bool = False


def _get_provider(mock: bool = False):
    if mock:
        return MockNimbleProvider()
    api_key = os.environ.get("NIMBLE_API_KEY", "")
    if not api_key:
        return MockNimbleProvider()
    return NimbleClient(api_key=api_key)


@app.post("/research/start")
async def start_research(req: ResearchRequest):
    request = UserResearchRequest(
        user_query=req.query,
        time_budget=TimeBudget(req.time_budget),
        preferred_format=ReportFormat(req.report_format),
    )
    provider = _get_provider(req.mock)
    storage = JsonStorageBackend()

    summary = await run_research(request, provider, storage)
    report = await storage.load_report(str(summary.session_id))

    return {
        "summary": summary.model_dump(mode="json"),
        "report": format_report(report, ReportFormat(req.report_format)) if report else None,
    }


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    storage = JsonStorageBackend()
    config = await storage.load_session(session_id)
    if not config:
        raise HTTPException(404, "Session not found")
    return config.model_dump(mode="json")


@app.get("/session/{session_id}/report")
async def get_report(session_id: str, format: str = "full_report"):
    storage = JsonStorageBackend()
    report = await storage.load_report(session_id)
    if not report:
        raise HTTPException(404, "No report found")
    return {"report": format_report(report, ReportFormat(format))}


@app.get("/session/{session_id}/skill")
async def get_skill(session_id: str):
    storage = JsonStorageBackend()
    skill = await storage.load_skill(session_id)
    if not skill:
        raise HTTPException(404, "No skill found")
    return skill.model_dump(mode="json")


@app.get("/sessions")
async def list_sessions():
    storage = JsonStorageBackend()
    return await storage.list_sessions()
