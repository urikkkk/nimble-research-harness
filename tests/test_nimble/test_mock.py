"""Tests for mock Nimble provider."""

import pytest

from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.nimble.types import ExtractParams, MapParams, SearchParams


@pytest.fixture
def provider():
    return MockNimbleProvider()


@pytest.mark.asyncio
async def test_search(provider):
    resp = await provider.search(SearchParams(query="test query"))
    assert len(resp.results) == 2
    assert resp.results[0].title.startswith("Result for:")
    assert resp.request_id == "mock-req-001"


@pytest.mark.asyncio
async def test_search_with_answer(provider):
    resp = await provider.search(SearchParams(query="test", include_answer=True))
    assert resp.answer is not None


@pytest.mark.asyncio
async def test_extract(provider):
    resp = await provider.extract(ExtractParams(url="https://example.com"))
    assert resp.markdown is not None
    assert "example.com" in resp.markdown


@pytest.mark.asyncio
async def test_map(provider):
    resp = await provider.map_urls(MapParams(url="https://example.com"))
    assert len(resp.links) == 3
    assert resp.success is True


@pytest.mark.asyncio
async def test_list_agents(provider):
    agents = await provider.list_agents()
    assert len(agents) == 4
    assert agents[0].name == "amazon_search"


@pytest.mark.asyncio
async def test_list_agents_with_filter(provider):
    agents = await provider.list_agents(query="amazon")
    assert len(agents) == 2


@pytest.mark.asyncio
async def test_run_agent(provider):
    resp = await provider.run_agent("amazon_search", {"query": "laptop"})
    assert resp.status == "success"
    assert resp.data is not None


@pytest.mark.asyncio
async def test_call_log(provider):
    await provider.search(SearchParams(query="q1"))
    await provider.extract(ExtractParams(url="https://x.com"))
    assert len(provider.call_log) == 2
    assert provider.call_log[0]["method"] == "search"
    assert provider.call_log[1]["method"] == "extract"
