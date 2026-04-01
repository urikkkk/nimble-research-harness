"""Shared test fixtures."""

import os
import tempfile
import uuid

import pytest

from nimble_research_harness.infra.context import RunContext, set_context
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend


@pytest.fixture
def mock_provider():
    return MockNimbleProvider()


@pytest.fixture
def temp_storage(tmp_path):
    return JsonStorageBackend(base_dir=str(tmp_path / "sessions"))


@pytest.fixture
def session_id():
    return uuid.uuid4()


@pytest.fixture
def run_context(temp_storage, session_id):
    ctx = RunContext(session_id=session_id, storage=temp_storage)
    set_context(ctx)
    return ctx
