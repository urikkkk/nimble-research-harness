"""Live Nimble API client using httpx."""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from ..infra.errors import NimbleApiError
from ..infra.logging import get_logger
from ..infra.retry import nimble_retry
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

logger = get_logger(__name__)

TIMEOUTS = {
    "search": 60.0,
    "extract": 60.0,
    "map": 60.0,
    "crawl": 300.0,
    "agents": 120.0,
    "tasks": 30.0,
}


class NimbleClient:
    """Live implementation of NimbleProvider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("NIMBLE_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("NIMBLE_BASE_URL", "https://sdk.nimbleway.com/v1")
        ).rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0),
            )
        return self._client

    async def _request(
        self, method: str, path: str, timeout_key: str = "search", **kwargs: Any
    ) -> dict[str, Any]:
        client = await self._get_client()
        timeout = TIMEOUTS.get(timeout_key, 60.0)
        try:
            resp = await client.request(
                method, path, timeout=httpx.Timeout(timeout), **kwargs
            )
        except httpx.TimeoutException as e:
            raise NimbleApiError(408, str(e), f"Timeout on {method} {path}")
        if resp.status_code >= 400:
            raise NimbleApiError(resp.status_code, resp.text)
        return resp.json()

    @nimble_retry
    async def search(self, params: SearchParams) -> SearchResponse:
        body = params.model_dump(exclude_none=True)
        data = await self._request("POST", "/search", timeout_key="search", json=body)
        results = []
        for i, r in enumerate(data.get("results", data.get("organic_results", []))):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", r.get("link", "")),
                    "snippet": r.get("snippet", r.get("description", "")),
                    "content": r.get("content"),
                    "position": r.get("position", i + 1),
                }
            )
        return SearchResponse(
            results=results,
            answer=data.get("answer"),
            request_id=data.get("request_id"),
        )

    @nimble_retry
    async def extract(self, params: ExtractParams) -> ExtractResponse:
        body = params.model_dump(exclude_none=True)
        data = await self._request("POST", "/extract", timeout_key="extract", json=body)
        return ExtractResponse(
            url=data.get("url", params.url),
            task_id=data.get("task_id"),
            status=data.get("status", ""),
            markdown=data.get("data", {}).get("markdown") if isinstance(data.get("data"), dict) else None,
            html=data.get("data", {}).get("html") if isinstance(data.get("data"), dict) else None,
            status_code=data.get("status_code"),
        )

    @nimble_retry
    async def map_urls(self, params: MapParams) -> MapResponse:
        body = params.model_dump(exclude_none=True)
        data = await self._request("POST", "/map", timeout_key="map", json=body)
        links = [
            {"url": l.get("url", ""), "title": l.get("title"), "description": l.get("description")}
            for l in data.get("links", [])
        ]
        return MapResponse(
            task_id=data.get("task_id"),
            success=data.get("success", True),
            links=links,
        )

    @nimble_retry
    async def crawl_run(self, params: CrawlParams) -> CrawlRunResponse:
        body = params.model_dump(exclude_none=True)
        data = await self._request("POST", "/crawl", timeout_key="crawl", json=body)
        return CrawlRunResponse(
            crawl_id=data.get("crawl_id", ""),
            status=data.get("status", ""),
        )

    @nimble_retry
    async def crawl_status(self, crawl_id: str) -> CrawlStatusResponse:
        data = await self._request("GET", f"/crawl/{crawl_id}", timeout_key="crawl")
        return CrawlStatusResponse(**data)

    @nimble_retry
    async def list_agents(
        self, query: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> list[AgentSummary]:
        params: dict[str, Any] = {"limit": limit, "privacy": "all", "offset": offset}
        if query:
            params["search"] = query
        data = await self._request("GET", "/agents", timeout_key="agents", params=params)
        agents_list = data if isinstance(data, list) else data.get("agents", data.get("data", []))
        return [AgentSummary(**a) for a in agents_list]

    @nimble_retry
    async def get_agent(self, name: str) -> AgentDetails:
        data = await self._request("GET", f"/agents/{name}", timeout_key="agents")
        return AgentDetails(**data)

    @nimble_retry
    async def run_agent(self, agent_name: str, params: dict) -> AgentRunResponse:
        body = {"agent": agent_name, "params": params}
        data = await self._request("POST", "/agents/run", timeout_key="agents", json=body)
        return AgentRunResponse(
            task_id=data.get("task_id"),
            status=data.get("status", ""),
            data=data.get("data"),
        )

    @nimble_retry
    async def task_results(self, task_id: str) -> TaskResults:
        data = await self._request("GET", f"/tasks/{task_id}/results", timeout_key="tasks")
        return TaskResults(
            task_id=task_id,
            status=data.get("status", ""),
            data=data.get("data"),
        )

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
