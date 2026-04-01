"""NimbleProvider protocol and implementations."""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .types import (
    AgentDetails,
    AgentRunResponse,
    AgentSummary,
    CrawlParams,
    CrawlRunResponse,
    CrawlStatusResponse,
    ExtractParams,
    ExtractResponse,
    MapParams,
    MapResponse,
    SearchParams,
    SearchResponse,
    TaskResults,
)


@runtime_checkable
class NimbleProvider(Protocol):
    async def search(self, params: SearchParams) -> SearchResponse: ...
    async def extract(self, params: ExtractParams) -> ExtractResponse: ...
    async def map_urls(self, params: MapParams) -> MapResponse: ...
    async def crawl_run(self, params: CrawlParams) -> CrawlRunResponse: ...
    async def crawl_status(self, crawl_id: str) -> CrawlStatusResponse: ...
    async def list_agents(self, query: Optional[str] = None, limit: int = 100) -> list[AgentSummary]: ...
    async def get_agent(self, name: str) -> AgentDetails: ...
    async def run_agent(self, agent_name: str, params: dict) -> AgentRunResponse: ...
    async def task_results(self, task_id: str) -> TaskResults: ...
