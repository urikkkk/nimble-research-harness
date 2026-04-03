"""Parallel sub-agent search — executes Nimble searches and extractions directly."""

from __future__ import annotations

import asyncio
from typing import Any

from ..infra.logging import get_logger
from ..nimble.provider import NimbleProvider
from ..nimble.types import ExtractParams, SearchParams
from .state import SearchFinding

logger = get_logger(__name__)


async def _search_one(
    query: str,
    provider: NimbleProvider,
    focus: str = "general",
    max_results: int = 10,
) -> list[SearchFinding]:
    """Execute a single search query and return findings."""
    try:
        params = SearchParams(
            query=query,
            max_results=max_results,
            focus=focus,
        )
        resp = await provider.search(params)
        findings = []
        for r in resp.results:
            if r.url:
                findings.append(SearchFinding(
                    query=query,
                    url=r.url,
                    title=r.title or "",
                    snippet=r.snippet or r.content or "",
                    relevance=max(0.3, 1.0 - r.position * 0.1),
                ))
        return findings
    except Exception as e:
        logger.warning("search_failed", query=query[:60], error=str(e))
        return []


async def _extract_one(
    url: str,
    provider: NimbleProvider,
    timeout: float = 20.0,
) -> str:
    """Extract full content from a URL."""
    try:
        params = ExtractParams(url=url)
        resp = await asyncio.wait_for(
            provider.extract(params),
            timeout=timeout,
        )
        return resp.markdown or resp.html or ""
    except Exception as e:
        logger.warning("extract_failed", url=url[:60], error=str(e))
        return ""


async def search_hop(
    queries: list[str],
    provider: NimbleProvider,
    max_parallel: int = 4,
    extract_top_n: int = 3,
    focus: str = "general",
) -> tuple[list[SearchFinding], int, int]:
    """Execute multiple search queries in parallel, extract top results.

    Returns:
        (findings, search_count, extract_count)
    """
    semaphore = asyncio.Semaphore(max_parallel)
    all_findings: list[SearchFinding] = []
    search_count = 0
    extract_count = 0

    async def _search_with_limit(query: str) -> list[SearchFinding]:
        async with semaphore:
            return await _search_one(query, provider, focus=focus)

    # Phase 1: Execute all searches in parallel
    search_tasks = [_search_with_limit(q) for q in queries]
    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, list):
            all_findings.extend(result)
            search_count += 1

    # Phase 2: Extract top-N most relevant unique URLs
    seen_urls: set[str] = set()
    urls_to_extract: list[str] = []
    for f in sorted(all_findings, key=lambda x: x.relevance, reverse=True):
        if f.url not in seen_urls and len(urls_to_extract) < extract_top_n:
            seen_urls.add(f.url)
            urls_to_extract.append(f.url)

    if urls_to_extract:
        async def _extract_with_limit(url: str) -> tuple[str, str]:
            async with semaphore:
                content = await _extract_one(url, provider)
                return url, content

        extract_tasks = [_extract_with_limit(u) for u in urls_to_extract]
        extract_results = await asyncio.gather(*extract_tasks, return_exceptions=True)

        for result in extract_results:
            if isinstance(result, tuple):
                url, content = result
                if content:
                    extract_count += 1
                    # Enrich the finding with full content
                    for f in all_findings:
                        if f.url == url:
                            f.full_content = content[:5000]
                            break

    logger.info(
        "hop_search_complete",
        queries=len(queries),
        findings=len(all_findings),
        searches=search_count,
        extracts=extract_count,
    )

    return all_findings, search_count, extract_count
