"""Tests for SSE event streaming."""

import asyncio
import json

import pytest

from nimble_research_harness.infra.events import EventStream, ResearchEvent


class TestResearchEvent:
    def test_to_sse_format(self):
        event = ResearchEvent(event_type="test.event", data={"key": "value"})
        sse = event.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        payload = json.loads(sse[6:].strip())
        assert payload["type"] == "test.event"
        assert payload["data"]["key"] == "value"

    def test_to_dict(self):
        event = ResearchEvent(event_type="test", data={"x": 1})
        d = event.to_dict()
        assert d["type"] == "test"
        assert d["data"]["x"] == 1
        assert "timestamp" in d


class TestEventStream:
    @pytest.mark.asyncio
    async def test_emit_and_listen(self):
        stream = EventStream(session_id="test")
        events = []

        async def _collect():
            async for event in stream.listen():
                events.append(event)

        collector = asyncio.create_task(_collect())
        await stream.emit("stage.entered", {"stage": "intake"})
        await stream.emit("tool.called", {"tool": "search"})
        await stream.close()
        await collector

        assert len(events) == 2
        assert events[0].event_type == "stage.entered"
        assert events[1].event_type == "tool.called"

    @pytest.mark.asyncio
    async def test_history(self):
        stream = EventStream(session_id="test")
        await stream.emit("a", {})
        await stream.emit("b", {})
        assert len(stream.history) == 2
        assert stream.history[0].event_type == "a"

    @pytest.mark.asyncio
    async def test_convenience_emitters(self):
        stream = EventStream(session_id="test")
        await stream.session_started("query", "5m")
        await stream.stage_entered("intake", 1.5)
        await stream.finding_added("Found something")
        await stream.budget_warning(30.0)

        assert len(stream.history) == 4
        assert stream.history[0].event_type == "session.started"
        assert stream.history[1].data["stage"] == "intake"

    @pytest.mark.asyncio
    async def test_closed_stream_ignores_emit(self):
        stream = EventStream(session_id="test")
        await stream.close()
        await stream.emit("should_be_ignored", {})
        assert len(stream.history) == 0
