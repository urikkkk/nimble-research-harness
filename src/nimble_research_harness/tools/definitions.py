"""Build and register all Nimble tool definitions."""

from __future__ import annotations

import time
import uuid
from typing import Any

from ..infra.context import get_context
from ..models.enums import ToolCallStatus, ToolName
from ..models.evidence import EvidenceItem
from ..models.execution import ToolCallRecord
from ..nimble.provider import NimbleProvider
from ..nimble.types import (
    CrawlParams,
    ExtractParams,
    MapParams,
    SearchParams,
)
from .registry import ToolDefinition, ToolRegistry

_VALID_FOCUS = {"general", "news", "coding", "academic", "shopping", "social", "geo", "location"}

_MIN_CONTENT_LENGTH = 50  # Below this, extraction likely returned a JS shell


def _ensure_list(val: Any) -> list[str]:
    """Coerce a string to a list; split on newlines if present."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        items = [line.strip().lstrip("- ").lstrip("* ") for line in val.split("\n") if line.strip()]
        return items if items else [val]
    return []


def build_registry(provider: NimbleProvider) -> ToolRegistry:
    """Create a ToolRegistry with all Nimble tool definitions wired to the provider."""
    registry = ToolRegistry()

    # --- nimble_search ---
    async def handle_search(params: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        ctx = get_context()
        if params.get("focus", "general") not in _VALID_FOCUS:
            params["focus"] = "general"
        search_params = SearchParams(**params)
        resp = await provider.search(search_params)
        latency = int((time.time() - start) * 1000)

        await ctx.storage.insert_tool_call(
            ToolCallRecord(
                session_id=ctx.session_id,
                tool=ToolName.SEARCH,
                params=params,
                status=ToolCallStatus.SUCCESS,
                response_summary=f"{len(resp.results)} results",
                result_count=len(resp.results),
                latency_ms=latency,
            )
        )

        for r in resp.results:
            if r.url:
                await ctx.storage.insert_evidence(
                    EvidenceItem(
                        session_id=ctx.session_id,
                        source_url=r.url,
                        title=r.title,
                        content=r.snippet or r.content or "",
                        content_type="search_result",
                        relevance_score=max(0.3, 1.0 - r.position * 0.1),
                    )
                )

        return {
            "results": [r.model_dump() for r in resp.results],
            "answer": resp.answer,
            "count": len(resp.results),
        }

    registry.register(
        ToolDefinition(
            name="nimble_search",
            description="Search the web using Nimble Search API. Returns ranked results with titles, URLs, and snippets.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 10},
                    "focus": {
                        "type": "string",
                        "enum": ["general", "news", "coding", "academic", "shopping", "social", "geo", "location"],
                        "default": "general",
                    },
                    "search_depth": {"type": "string", "enum": ["lite", "fast", "deep"], "default": "lite"},
                    "include_answer": {"type": "boolean", "default": False},
                    "include_domains": {"type": "array", "items": {"type": "string"}, "default": []},
                    "exclude_domains": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": ["query"],
            },
            handler=handle_search,
        )
    )

    # --- nimble_extract ---
    async def handle_extract(params: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        ctx = get_context()

        extract_params = ExtractParams(**params)
        resp = await provider.extract(extract_params)
        content = resp.markdown or resp.html or ""

        # If content is too short, retry with JS rendering enabled
        if len(content.strip()) < _MIN_CONTENT_LENGTH and not extract_params.render:
            params["render"] = True
            retry_params = ExtractParams(**params)
            resp = await provider.extract(retry_params)
            content = resp.markdown or resp.html or ""

        latency = int((time.time() - start) * 1000)
        content_is_substantive = len(content.strip()) >= _MIN_CONTENT_LENGTH

        await ctx.storage.insert_tool_call(
            ToolCallRecord(
                session_id=ctx.session_id,
                tool=ToolName.EXTRACT,
                params=params,
                status=ToolCallStatus.SUCCESS if content_is_substantive else ToolCallStatus.PARTIAL,
                response_summary=f"Extracted {len(content)} chars" if content_is_substantive else "Empty/minimal content",
                result_count=1 if content_is_substantive else 0,
                latency_ms=latency,
            )
        )

        if content_is_substantive:
            await ctx.storage.insert_evidence(
                EvidenceItem(
                    session_id=ctx.session_id,
                    source_url=params["url"],
                    content=content[:10000],
                    content_type="extracted_page",
                    relevance_score=0.7,
                )
            )

        return {"url": resp.url, "content": content[:5000], "status": resp.status}

    registry.register(
        ToolDefinition(
            name="nimble_extract",
            description="Extract structured content from a URL using Nimble Extract API. Returns markdown or HTML.",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to extract content from"},
                    "formats": {"type": "array", "items": {"type": "string"}, "default": ["markdown"]},
                    "render": {"type": "boolean", "default": False},
                },
                "required": ["url"],
            },
            handler=handle_extract,
        )
    )

    # --- nimble_map ---
    async def handle_map(params: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        ctx = get_context()
        map_params = MapParams(**params)
        resp = await provider.map_urls(map_params)
        latency = int((time.time() - start) * 1000)

        await ctx.storage.insert_tool_call(
            ToolCallRecord(
                session_id=ctx.session_id,
                tool=ToolName.MAP,
                params=params,
                status=ToolCallStatus.SUCCESS,
                response_summary=f"{len(resp.links)} links discovered",
                result_count=len(resp.links),
                latency_ms=latency,
            )
        )

        return {
            "links": [l.model_dump() for l in resp.links[:50]],
            "total": len(resp.links),
        }

    registry.register(
        ToolDefinition(
            name="nimble_map",
            description="Discover URLs and site structure using Nimble Map API. Returns list of discovered links.",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Website URL to map"},
                    "limit": {"type": "integer", "default": 100},
                    "sitemap": {"type": "string", "enum": ["include", "only", "skip"], "default": "include"},
                },
                "required": ["url"],
            },
            handler=handle_map,
        )
    )

    # --- nimble_crawl_run ---
    async def handle_crawl_run(params: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        ctx = get_context()
        crawl_params = CrawlParams(**params)
        resp = await provider.crawl_run(crawl_params)
        latency = int((time.time() - start) * 1000)

        await ctx.storage.insert_tool_call(
            ToolCallRecord(
                session_id=ctx.session_id,
                tool=ToolName.CRAWL_RUN,
                params=params,
                status=ToolCallStatus.SUCCESS,
                response_summary=f"Crawl started: {resp.crawl_id}",
                latency_ms=latency,
            )
        )

        return {"crawl_id": resp.crawl_id, "status": resp.status}

    registry.register(
        ToolDefinition(
            name="nimble_crawl_run",
            description="Start a crawl job using Nimble Crawl API. Returns crawl_id for status tracking.",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Starting URL for crawl"},
                    "limit": {"type": "integer", "default": 50},
                    "max_discovery_depth": {"type": "integer", "default": 3},
                },
                "required": ["url"],
            },
            handler=handle_crawl_run,
        )
    )

    # --- nimble_crawl_status ---
    async def handle_crawl_status(params: dict[str, Any]) -> dict[str, Any]:
        ctx = get_context()
        resp = await provider.crawl_status(params["crawl_id"])
        return {
            "crawl_id": resp.crawl_id,
            "status": resp.status,
            "total": resp.total,
            "completed": resp.completed,
            "pending": resp.pending,
        }

    registry.register(
        ToolDefinition(
            name="nimble_crawl_status",
            description="Check the status of a running crawl job.",
            input_schema={
                "type": "object",
                "properties": {
                    "crawl_id": {"type": "string", "description": "Crawl job ID"},
                },
                "required": ["crawl_id"],
            },
            handler=handle_crawl_status,
        )
    )

    # --- nimble_agents_run ---
    async def handle_agents_run(params: dict[str, Any]) -> dict[str, Any]:
        start = time.time()
        ctx = get_context()
        agent_name = params.pop("agent_name")
        resp = await provider.run_agent(agent_name, params)
        latency = int((time.time() - start) * 1000)

        await ctx.storage.insert_tool_call(
            ToolCallRecord(
                session_id=ctx.session_id,
                tool=ToolName.AGENTS_RUN,
                params={"agent_name": agent_name, **params},
                status=ToolCallStatus.SUCCESS,
                response_summary=f"WSA {agent_name}: {resp.status}",
                latency_ms=latency,
            )
        )

        return {"task_id": resp.task_id, "status": resp.status, "data": resp.data}

    registry.register(
        ToolDefinition(
            name="nimble_agents_run",
            description="Run a pre-built Nimble WSA (Web Search Agent) for structured data extraction from specific websites.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "WSA agent template name (e.g. 'amazon_search')"},
                    "url": {"type": "string", "description": "Target URL for the agent"},
                    "query": {"type": "string", "description": "Search query (for SERP agents)"},
                },
                "required": ["agent_name"],
            },
            handler=handle_agents_run,
        )
    )

    # --- write_finding ---
    async def handle_write_finding(params: dict[str, Any]) -> dict[str, Any]:
        ctx = get_context()
        await ctx.storage.insert_evidence(
            EvidenceItem(
                session_id=ctx.session_id,
                source_url=params.get("source_url", ""),
                title=params.get("title"),
                content=params["content"],
                content_type="finding",
                relevance_score=params.get("relevance", 0.5),
            )
        )
        return {"status": "ok"}

    registry.register(
        ToolDefinition(
            name="write_finding",
            description="Record a research finding with its source.",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The finding text"},
                    "source_url": {"type": "string", "description": "Source URL"},
                    "title": {"type": "string", "description": "Finding title"},
                    "relevance": {"type": "number", "default": 0.5},
                },
                "required": ["content"],
            },
            handler=handle_write_finding,
        )
    )

    # --- read_evidence ---
    async def handle_read_evidence(params: dict[str, Any]) -> dict[str, Any]:
        ctx = get_context()
        evidence = await ctx.storage.get_evidence(str(ctx.session_id))
        items = [
            {
                "id": str(e.evidence_id),
                "url": e.source_url,
                "title": e.title,
                "content": e.content[:500],
                "type": e.content_type,
                "relevance": e.relevance_score,
            }
            for e in evidence
        ]
        return {"evidence": items, "count": len(items)}

    registry.register(
        ToolDefinition(
            name="read_evidence",
            description="Read all collected evidence for this research session.",
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=handle_read_evidence,
        )
    )

    # --- write_claim ---
    async def handle_write_claim(params: dict[str, Any]) -> dict[str, Any]:
        from ..models.evidence import Claim
        from ..models.enums import ClaimConfidence

        ctx = get_context()
        claim = Claim(
            session_id=ctx.session_id,
            statement=params["statement"],
            evidence_ids=[uuid.UUID(eid) for eid in params.get("evidence_ids", [])],
            confidence=ClaimConfidence(params.get("confidence", "unresolved")),
            category=params.get("category"),
            importance=params.get("importance", 1),
        )
        await ctx.storage.insert_claim(claim)
        return {"claim_id": str(claim.claim_id), "status": "ok"}

    registry.register(
        ToolDefinition(
            name="write_claim",
            description="Record a research claim linked to supporting evidence.",
            input_schema={
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "description": "The claim statement"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}, "default": []},
                    "confidence": {
                        "type": "string",
                        "enum": ["verified", "partially_verified", "weak_support", "unresolved"],
                        "default": "unresolved",
                    },
                    "category": {"type": "string"},
                    "importance": {"type": "integer", "default": 1},
                },
                "required": ["statement"],
            },
            handler=handle_write_claim,
        )
    )

    # --- write_report ---
    async def handle_write_report(params: dict[str, Any]) -> dict[str, Any]:
        from ..models.output import ResearchReport

        ctx = get_context()
        report = ResearchReport(
            session_id=ctx.session_id,
            title=params.get("title", "Research Report"),
            executive_summary=params.get("executive_summary", ""),
            key_findings=_ensure_list(params.get("key_findings", [])),
            detailed_analysis=params.get("detailed_analysis", ""),
            methodology=params.get("methodology", ""),
            known_unknowns=_ensure_list(params.get("known_unknowns", [])),
            limitations=_ensure_list(params.get("limitations", [])),
        )
        await ctx.storage.save_report(report)
        return {"report_id": str(report.report_id), "status": "ok"}

    registry.register(
        ToolDefinition(
            name="write_report",
            description="Write the final research report with executive summary, findings, and analysis.",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "executive_summary": {"type": "string"},
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "detailed_analysis": {"type": "string"},
                    "methodology": {"type": "string"},
                    "known_unknowns": {"type": "array", "items": {"type": "string"}},
                    "limitations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "executive_summary", "key_findings"],
            },
            handler=handle_write_report,
        )
    )

    return registry
