"""Agentic deep research — single Opus loop with direct tool access.

Unlike the pipeline approach (decompose → queries → search → extract),
this gives Claude Opus direct access to search and extract tools.
Opus sees each result, reasons about it, and decides the next search
in real-time — exactly how a human researcher would work.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import anthropic

from ..infra.logging import get_logger
from ..nimble.provider import NimbleProvider
from ..nimble.types import ExtractParams, SearchParams
from ..tools.registry import ToolDefinition, ToolRegistry
from .state import Candidate, DeepResearchSession

logger = get_logger(__name__)

OPUS_MODEL = "claude-opus-4-6"

SYSTEM_PROMPT = """You are a world-class research agent solving extremely difficult trivia questions.
These questions describe something (a person, place, event, movie, etc.) using multiple overlapping constraints.
Your job is to find the SPECIFIC SHORT ANSWER — typically a name, date, time, number, or title (1-5 words).

PROVEN STRATEGY (from analyzing successful solves):

Phase 1 — Identify the Entity:
1. Read the question carefully. Identify the 2-3 MOST DISTINCTIVE constraints.
2. Search with SHORT queries (3-6 words) combining the most unusual constraints.
3. After EACH search, carefully read the titles and snippets. Look for proper nouns that could be the entity.
4. When you spot a promising entity name, IMMEDIATELY search "[entity name] Wikipedia" and extract the page.

Phase 2 — Extract the Answer:
5. Once you know the entity, search "[entity name] [specific detail the question asks for]"
6. Extract relevant pages to find the exact detail (date, time, place, etc.)
7. VERIFY: re-read the question and check if your candidate satisfies ALL constraints.

Phase 3 — Verify & Submit:
8. If confident, call submit_answer. If not, keep searching from different angles.

CRITICAL RULES:
- Each search query MUST be 3-8 words. NEVER paste the full question as a query.
- After finding an entity name, ALWAYS extract its Wikipedia/bio page for details.
- If stuck after 8-10 searches, PIVOT completely: try different constraints, different databases.
- Use specialized sources: site:wikipedia.org, site:imdb.com, site:transfermarkt.com, site:rsssf.com
- Search for LISTS when direct search fails: "list of [category] [constraint]"
- LinkedIn is excellent for finding people with specific employment history.
- You have up to 50 tool calls. Be persistent — these questions ARE solvable.
- A wrong guess is better than no answer. ALWAYS call submit_answer before finishing."""


async def agentic_research(
    question: str,
    provider: NimbleProvider,
    timeout_seconds: float = 1740.0,
    max_turns: int = 50,
) -> DeepResearchSession:
    """Run a single Opus agentic loop for deep multi-hop research.

    Claude Opus gets direct access to nimble_search, nimble_extract, and submit_answer.
    It controls the entire research process — no pre-planned queries or pipelines.
    """
    start = time.time()
    session = DeepResearchSession(question=question)
    _answer_submitted = {"answer": "", "confidence": 0.0, "reasoning": ""}

    # --- Build tool registry ---
    registry = ToolRegistry()

    async def handle_search(params: dict[str, Any]) -> dict[str, Any]:
        query = params.get("query", "")
        focus = params.get("focus", "general")
        max_results = params.get("max_results", 10)

        if focus not in ("general", "news", "coding", "academic", "shopping", "social", "geo", "location"):
            focus = "general"

        try:
            resp = await provider.search(SearchParams(query=query, max_results=max_results, focus=focus))
            session.total_searches += 1
            results = []
            for r in resp.results:
                results.append({
                    "title": r.title or "",
                    "url": r.url or "",
                    "snippet": (r.snippet or r.content or "")[:300],
                    "position": r.position,
                })
            return {"results": results[:max_results], "count": len(results)}
        except Exception as e:
            return {"error": str(e)[:200]}

    async def handle_extract(params: dict[str, Any]) -> dict[str, Any]:
        url = params.get("url", "")
        try:
            resp = await asyncio.wait_for(
                provider.extract(ExtractParams(url=url)),
                timeout=30,
            )
            session.total_extracts += 1
            content = resp.markdown or resp.html or ""
            return {"url": url, "content": content[:8000], "length": len(content)}
        except Exception as e:
            return {"error": str(e)[:200]}

    async def handle_submit(params: dict[str, Any]) -> dict[str, Any]:
        _answer_submitted["answer"] = params.get("answer", "")
        _answer_submitted["confidence"] = float(params.get("confidence", 0.5))
        _answer_submitted["reasoning"] = params.get("reasoning", "")
        return {"status": "submitted", "answer": _answer_submitted["answer"]}

    registry.register(ToolDefinition(
        name="nimble_search",
        description="Search the web. Returns titles, URLs, and snippets. Use short queries (3-8 words).",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (3-8 words, be specific)"},
                "focus": {
                    "type": "string",
                    "enum": ["general", "news", "academic", "coding", "shopping", "social"],
                    "default": "general",
                },
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
        handler=handle_search,
    ))

    registry.register(ToolDefinition(
        name="nimble_extract",
        description="Extract full page content from a URL. Use this to read Wikipedia pages, articles, and databases in detail.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to extract content from"},
            },
            "required": ["url"],
        },
        handler=handle_extract,
    ))

    registry.register(ToolDefinition(
        name="submit_answer",
        description="Submit your final answer. ALWAYS call this before finishing, even if you're not sure.",
        input_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Your specific, short answer (1-5 words)"},
                "confidence": {"type": "number", "description": "Confidence 0.0-1.0"},
                "reasoning": {"type": "string", "description": "Brief explanation of how you found this answer"},
            },
            "required": ["answer", "confidence"],
        },
        handler=handle_submit,
    ))

    # --- Run the agentic loop ---
    logger.info("agentic_research_start", question=question[:80], timeout=timeout_seconds, model=OPUS_MODEL)

    client = anthropic.AsyncAnthropic()
    tools = registry.get_schemas()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    total_tool_calls = 0

    try:
        for turn in range(max_turns):
            elapsed = time.time() - start
            if elapsed >= timeout_seconds:
                logger.info("agentic_timeout", turn=turn, elapsed=f"{elapsed:.0f}s")
                break

            response = await asyncio.wait_for(
                client.messages.create(
                    model=OPUS_MODEL,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=tools,
                    max_tokens=16384,
                ),
                timeout=min(120, timeout_seconds - elapsed),
            )
            session.total_llm_calls += 1

            has_tool_use = any(b.type == "tool_use" for b in response.content)

            if not has_tool_use:
                # Agent finished without submitting — try to extract answer from text
                text_parts = [b.text for b in response.content if b.type == "text"]
                final_text = "\n".join(text_parts)

                if not _answer_submitted["answer"]:
                    # Try to parse answer from text
                    match = re.search(r"(?:Exact Answer|Answer|ANSWER)[:\s]+(.+?)(?:\n|$)", final_text)
                    if match:
                        _answer_submitted["answer"] = match.group(1).strip()
                        _answer_submitted["confidence"] = 0.3
                break

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    total_tool_calls += 1
                    logger.info("agentic_tool", tool=block.name, turn=turn, input_preview=str(block.input)[:100])

                    result = await registry.dispatch(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str)[:12000],
                    })

                    # If answer submitted, we can stop
                    if block.name == "submit_answer" and _answer_submitted["answer"]:
                        logger.info(
                            "answer_submitted",
                            answer=_answer_submitted["answer"],
                            confidence=_answer_submitted["confidence"],
                            turn=turn,
                            tool_calls=total_tool_calls,
                        )

            messages.append({"role": "user", "content": tool_results})

            # Stop if answer was submitted
            if _answer_submitted["answer"]:
                break

    except asyncio.TimeoutError:
        logger.warning("agentic_overall_timeout", elapsed=f"{time.time() - start:.0f}s")
    except Exception as e:
        logger.error("agentic_error", error=str(e))

    # --- Build session result ---
    session.elapsed_seconds = time.time() - start
    session.final_answer = _answer_submitted["answer"]
    session.final_confidence = _answer_submitted["confidence"]

    if session.final_answer:
        session.candidates.append(Candidate(
            answer=session.final_answer,
            confidence=session.final_confidence,
            source_snippet=_answer_submitted["reasoning"][:500],
        ))

    logger.info(
        "agentic_complete",
        answer=session.final_answer or "(none)",
        confidence=session.final_confidence,
        searches=session.total_searches,
        extracts=session.total_extracts,
        llm_calls=session.total_llm_calls,
        elapsed=f"{session.elapsed_seconds:.0f}s",
    )

    return session
