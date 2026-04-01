"""Mock Nimble provider for testing."""

from __future__ import annotations

from typing import Optional

from .types import (
    AgentDetails,
    AgentRunResponse,
    AgentSummary,
    CrawlParams,
    CrawlRunResponse,
    CrawlStatusResponse,
    ExtractParams,
    ExtractResponse,
    MapLink,
    MapParams,
    MapResponse,
    SearchParams,
    SearchResponse,
    SearchResult,
    TaskResults,
)


class MockNimbleProvider:
    """Deterministic mock provider for testing — no network calls."""

    def __init__(self) -> None:
        self.call_log: list[dict] = []

    def _log(self, method: str, **kwargs) -> None:
        self.call_log.append({"method": method, **kwargs})

    async def search(self, params: SearchParams) -> SearchResponse:
        self._log("search", query=params.query)
        return SearchResponse(
            results=[
                SearchResult(
                    title=f"Result for: {params.query}",
                    url=f"https://example.com/search?q={params.query.replace(' ', '+')}",
                    snippet=f"Mock search result snippet for '{params.query}'.",
                    position=1,
                ),
                SearchResult(
                    title=f"Second result: {params.query}",
                    url=f"https://example.org/article/{params.query.replace(' ', '-')}",
                    snippet=f"Another mock result for '{params.query}'.",
                    position=2,
                ),
            ],
            answer=f"Mock answer: {params.query}" if params.include_answer else None,
            request_id="mock-req-001",
        )

    async def extract(self, params: ExtractParams) -> ExtractResponse:
        self._log("extract", url=params.url)
        return ExtractResponse(
            url=params.url,
            task_id="mock-task-extract-001",
            status="success",
            markdown=f"# Mock Extraction\n\nExtracted content from {params.url}.\n\nThis is mock content for testing the research harness pipeline.",
        )

    async def map_urls(self, params: MapParams) -> MapResponse:
        self._log("map", url=params.url)
        base = params.url.rstrip("/")
        return MapResponse(
            task_id="mock-task-map-001",
            success=True,
            links=[
                MapLink(url=f"{base}/about", title="About"),
                MapLink(url=f"{base}/products", title="Products"),
                MapLink(url=f"{base}/blog", title="Blog"),
            ],
        )

    async def crawl_run(self, params: CrawlParams) -> CrawlRunResponse:
        self._log("crawl_run", url=params.url)
        return CrawlRunResponse(crawl_id="mock-crawl-001", status="running")

    async def crawl_status(self, crawl_id: str) -> CrawlStatusResponse:
        self._log("crawl_status", crawl_id=crawl_id)
        return CrawlStatusResponse(
            crawl_id=crawl_id,
            status="succeeded",
            total=3,
            completed=3,
            failed=0,
            pending=0,
        )

    async def list_agents(
        self, query: Optional[str] = None, limit: int = 100
    ) -> list[AgentSummary]:
        self._log("list_agents", query=query)
        agents = [
            AgentSummary(
                name="amazon_search",
                display_name="Amazon Search",
                description="Search Amazon product listings",
                vertical="Ecommerce",
                entity_type="Search Engine Results Page (SERP)",
                domain="amazon.com",
            ),
            AgentSummary(
                name="amazon_pdp",
                display_name="Amazon Product Page",
                description="Extract Amazon product details",
                vertical="Ecommerce",
                entity_type="Product Detail Page (PDP)",
                domain="amazon.com",
            ),
            AgentSummary(
                name="linkedin_job_posting",
                display_name="LinkedIn Job Posting",
                description="Extract LinkedIn job details",
                vertical="Jobs & Careers",
                entity_type="Custom",
                domain="linkedin.com",
            ),
            AgentSummary(
                name="yahoo_finance_stock",
                display_name="Yahoo Finance Stock",
                description="Get stock data from Yahoo Finance",
                vertical="Finance",
                entity_type="Custom",
                domain="finance.yahoo.com",
            ),
        ]
        if query:
            agents = [a for a in agents if query.lower() in (a.name + (a.description or "")).lower()]
        return agents[:limit]

    async def get_agent(self, name: str) -> AgentDetails:
        self._log("get_agent", name=name)
        return AgentDetails(
            name=name,
            display_name=name.replace("_", " ").title(),
            description=f"Mock agent: {name}",
            input_properties={"url": {"type": "string", "required": True}},
            output_schema={"title": "string", "data": "object"},
        )

    async def run_agent(self, agent_name: str, params: dict) -> AgentRunResponse:
        self._log("run_agent", agent=agent_name, params=params)
        return AgentRunResponse(
            task_id="mock-task-agent-001",
            status="success",
            data={"mock": True, "agent": agent_name, "results": [{"item": "mock result"}]},
        )

    async def task_results(self, task_id: str) -> TaskResults:
        self._log("task_results", task_id=task_id)
        return TaskResults(
            task_id=task_id,
            status="success",
            data={"content": "Mock task results"},
        )

    async def close(self) -> None:
        pass
