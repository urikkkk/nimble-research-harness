"""Candidate extraction and ranking from search findings."""

from __future__ import annotations

import json

import anthropic

from ..infra.logging import get_logger
from .prompts import (
    ANSWER_FROM_ENTITY_PROMPT,
    ENTITY_DISCOVERY_PROMPT,
    EXTRACT_CANDIDATES_PROMPT,
    INITIAL_QUERIES_PROMPT,
    REFINE_QUERIES_PROMPT,
    GAP_ANALYSIS_PROMPT,
    SPECIALIZED_SOURCES,
)
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


# --- Entity Discovery Functions ---


def detect_domain(constraints: list[Constraint], answer_type: str) -> list[str]:
    """Detect domain from constraints and return specialized site: prefixes."""
    text = " ".join(c.text.lower() for c in constraints) + " " + answer_type.lower()

    domains = []
    if any(w in text for w in ("soccer", "football", "referee", "match", "player", "goal", "league")):
        domains.extend(SPECIALIZED_SOURCES.get("sports", []))
    if any(w in text for w in ("actor", "actress", "film", "movie", "directed", "starring")):
        domains.extend(SPECIALIZED_SOURCES.get("movie", []))
    if any(w in text for w in ("tv series", "tv show", "episode", "season", "sitcom")):
        domains.extend(SPECIALIZED_SOURCES.get("tv", []))
    if any(w in text for w in ("manga", "anime", "chapter", "mangaka")):
        domains.extend(SPECIALIZED_SOURCES.get("manga", []))
    if any(w in text for w in ("died", "death", "passed away", "obituary", "funeral")):
        domains.extend(SPECIALIZED_SOURCES.get("death", []))
    if any(w in text for w in ("paper", "published", "journal", "research", "author", "phd", "thesis")):
        domains.extend(SPECIALIZED_SOURCES.get("academic", []))
    if any(w in text for w in ("born", "worked at", "employed", "founded", "career")):
        domains.extend(SPECIALIZED_SOURCES.get("person", []))
    if any(w in text for w in ("album", "song", "band", "musician", "singer")):
        domains.extend(SPECIALIZED_SOURCES.get("music", []))

    # Always include Wikipedia
    if "site:wikipedia.org" not in domains:
        domains.append("site:wikipedia.org")

    return domains[:6]  # Max 6 site: prefixes


async def discover_entities(
    question: str,
    constraints: list[Constraint],
    findings: list[SearchFinding],
) -> list[dict]:
    """Discover named entities from search findings."""
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": ENTITY_DISCOVERY_PROMPT.format(
            question=question,
            constraints=_format_constraints(constraints),
            findings=_format_findings(findings),
        )}],
        max_tokens=1024,
    )
    try:
        entities = _parse_json_response(response.content[0].text)
        return [e for e in entities if isinstance(e, dict) and e.get("entity")]
    except (json.JSONDecodeError, IndexError):
        logger.warning("entity_discovery_parse_failed")
        return []


async def extract_answer_from_entity(
    question: str,
    entity_name: str,
    answer_type: str,
    findings: list[SearchFinding],
) -> Candidate | None:
    """Given an identified entity, extract the specific answer from evidence."""
    # Build evidence focused on this entity
    entity_lower = entity_name.lower()
    evidence = []
    for f in findings:
        text = f.full_content or f.snippet
        if entity_lower in text.lower() or any(w in text.lower() for w in entity_lower.split() if len(w) > 3):
            evidence.append(f"[{f.url}] {f.title}\n{text[:1000]}")

    if not evidence:
        return None

    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": ANSWER_FROM_ENTITY_PROMPT.format(
            question=question,
            entity_name=entity_name,
            answer_type=answer_type,
            evidence="\n\n".join(evidence[:10]),
        )}],
        max_tokens=512,
    )
    try:
        result = _parse_json_response(response.content[0].text)
        if isinstance(result, dict) and result.get("answer"):
            answer = result["answer"].strip()
            if answer.lower() != "none":
                return Candidate(
                    answer=answer,
                    confidence=float(result.get("confidence", 0.5)),
                    source_snippet=result.get("source_snippet", "")[:500],
                    constraints_met=[],
                )
    except (json.JSONDecodeError, IndexError):
        pass
    return None


def generate_entity_pivot_queries(entity_name: str, answer_type: str, constraints: list[Constraint]) -> list[str]:
    """Generate queries to find specific details about a known entity."""
    queries = [
        f"{entity_name} Wikipedia",
        f"{entity_name} biography",
        f'"{entity_name}" {answer_type}',
    ]

    # Add constraint-specific queries
    for c in constraints[:3]:
        queries.append(f"{entity_name} {c.text[:30]}")

    return queries[:6]
