"""Tests for JSON storage backend."""

import uuid

import pytest

from nimble_research_harness.models.enums import (
    ClaimConfidence,
    ExecutionMode,
    ExecutionStage,
    ReportFormat,
    TaskType,
    TimeBudget,
    ToolCallStatus,
    ToolName,
)
from nimble_research_harness.models.evidence import Claim, EvidenceItem
from nimble_research_harness.models.execution import RunCheckpoint, ToolCallRecord
from nimble_research_harness.models.output import ResearchReport
from nimble_research_harness.models.session import (
    SessionConfig,
    TimeBudgetPolicy,
)
from nimble_research_harness.storage.json_backend import JsonStorageBackend


@pytest.fixture
def storage(tmp_path):
    return JsonStorageBackend(base_dir=str(tmp_path / "sessions"))


@pytest.fixture
def sample_config():
    return SessionConfig(
        request_id=uuid.uuid4(),
        user_query="Test query",
        normalized_objective="Test query.",
        time_budget=TimeBudget.MEDIUM_5M,
        policy=TimeBudgetPolicy.from_budget(TimeBudget.MEDIUM_5M),
    )


@pytest.mark.asyncio
async def test_create_and_load_session(storage, sample_config):
    sid = await storage.create_session(sample_config)
    loaded = await storage.load_session(sid)
    assert loaded is not None
    assert loaded.user_query == "Test query"


@pytest.mark.asyncio
async def test_evidence_roundtrip(storage, sample_config):
    sid = await storage.create_session(sample_config)
    evidence = EvidenceItem(
        session_id=sample_config.session_id,
        source_url="https://example.com",
        content="Test evidence",
    )
    await storage.insert_evidence(evidence)
    items = await storage.get_evidence(sid)
    assert len(items) == 1
    assert items[0].content == "Test evidence"


@pytest.mark.asyncio
async def test_checkpoint_roundtrip(storage, sample_config):
    sid = await storage.create_session(sample_config)
    cp = RunCheckpoint(
        session_id=sample_config.session_id,
        stage=ExecutionStage.RESEARCH,
        stage_index=5,
        completed_steps=3,
        total_steps=10,
    )
    await storage.save_checkpoint(cp)
    loaded = await storage.load_latest_checkpoint(sid)
    assert loaded is not None
    assert loaded.stage == ExecutionStage.RESEARCH
    assert loaded.progress_pct == 30.0


@pytest.mark.asyncio
async def test_list_sessions(storage, sample_config):
    await storage.create_session(sample_config)
    sessions = await storage.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["user_query"] == "Test query"
