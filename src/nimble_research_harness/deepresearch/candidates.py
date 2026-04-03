"""Candidate extraction and ranking from search findings."""

from __future__ import annotations

import json

import anthropic

from ..infra.logging import get_logger
from .prompts import EXTRACT_CANDIDATES_PROMPT, INITIAL_QUERIES_PROMPT, REFINE_QUERIES_PROMPT, GAP_ANALYSIS_PROMPT
from .state import Candidate, Constraint, SearchFinding

logger = get_logger(__name__)

LLM_MODEL = "claude-sonnet-4-6"


def _format_constraints(constraints: list[Constraint]) -> str:
    return "\n".join(f"- [{c.category}] {c.text}" for c in constraints)


def _format_findings(findings: list[SearchFinding], max_items: int = 30) -> str:
    lines = []
    for f in findings[:max_items]:
        content = f.full_content[:800] if f.full_content else f.snippet[:300]
        lines.append(f"[{f.url}] {f.title}\n  {content}")
    return "\n\n".join(lines)


def _format_candidates(candidates: list[Candidate]) -> str:
    if not candidates:
        return "(none yet)"
    return "\n".join(
        f"- \"{c.answer}\" (confidence: {c.confidence:.1f}, met: {', '.join(c.constraints_met[:3])})"
        for c in candidates
    )


def _parse_json_response(text: str) -> list | dict:
    """Parse JSON from LLM response, handling code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    return json.loads(text)


async def generate_initial_queries(
    question: str,
    constraints: list[Constraint],
    num_queries: int = 6,
    answer_type: str = "other",
) -> list[str]:
    """Generate first-hop search queries from constraints."""
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": INITIAL_QUERIES_PROMPT.format(
            question=question,
            constraints=_format_constraints(constraints),
            num_queries=num_queries,
            answer_type=answer_type,
        )}],
        max_tokens=1024,
    )
    try:
        queries = _parse_json_response(response.content[0].text)
        return [q for q in queries if isinstance(q, str)][:num_queries]
    except (json.JSONDecodeError, IndexError):
        logger.warning("initial_queries_parse_failed")
        # Fallback: use first 2 constraints as queries
        return [c.text for c in constraints[:num_queries]]


async def generate_refined_queries(
    question: str,
    constraints: list[Constraint],
    candidates: list[Candidate],
    search_history: list[str],
    gap_analysis: str,
    num_queries: int = 6,
    answer_type: str = "other",
) -> list[str]:
    """Generate refined queries based on what we've learned."""
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": REFINE_QUERIES_PROMPT.format(
            question=question,
            constraints=_format_constraints(constraints),
            search_history="\n".join(f"- {q}" for q in search_history),
            candidates=_format_candidates(candidates),
            gap_analysis=gap_analysis,
            num_queries=num_queries,
            answer_type=answer_type,
        )}],
        max_tokens=1024,
    )
    try:
        queries = _parse_json_response(response.content[0].text)
        return [q for q in queries if isinstance(q, str)][:num_queries]
    except (json.JSONDecodeError, IndexError):
        logger.warning("refine_queries_parse_failed")
        # Fallback: combine unmet constraints
        unmet = [c.text for c in constraints if not c.is_met]
        return [f"{unmet[0]} {unmet[1]}" if len(unmet) >= 2 else unmet[0]] if unmet else []


async def extract_candidates(
    question: str,
    constraints: list[Constraint],
    findings: list[SearchFinding],
    existing_candidates: list[Candidate],
    hop: int = 0,
    answer_type: str = "other",
) -> list[Candidate]:
    """Extract potential answers from search findings."""
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": EXTRACT_CANDIDATES_PROMPT.format(
            question=question,
            constraints=_format_constraints(constraints),
            findings=_format_findings(findings),
            existing_candidates=_format_candidates(existing_candidates),
            answer_type=answer_type,
        )}],
        max_tokens=2048,
    )

    try:
        raw = _parse_json_response(response.content[0].text)
    except (json.JSONDecodeError, IndexError):
        logger.warning("candidates_parse_failed")
        return []

    candidates = []
    existing_answers = {c.answer.lower() for c in existing_candidates}

    for item in raw:
        if not isinstance(item, dict):
            continue
        answer = item.get("answer", "").strip()
        if not answer or answer.lower() in existing_answers or answer.lower() == "none":
            continue

        candidates.append(Candidate(
            answer=answer,
            confidence=float(item.get("confidence", 0.3)),
            source_url=item.get("source_url", ""),
            source_snippet=item.get("source_snippet", "")[:500],
            constraints_met=item.get("constraints_met", []),
            hop_found=hop,
        ))

    logger.info("candidates_extracted", count=len(candidates), hop=hop)
    return candidates


async def analyze_gaps(
    question: str,
    constraints: list[Constraint],
    search_history: list[str],
    candidates: list[Candidate],
    answer_type: str = "other",
) -> str:
    """Analyze what's missing and suggest next search direction."""
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": GAP_ANALYSIS_PROMPT.format(
            question=question,
            constraints=_format_constraints(constraints),
            search_history="\n".join(f"- {q}" for q in search_history[-15:]),
            candidates=_format_candidates(candidates),
            answer_type=answer_type,
        )}],
        max_tokens=512,
    )
    return response.content[0].text.strip()
