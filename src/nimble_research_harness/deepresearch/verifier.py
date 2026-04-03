"""Constraint verification — check if a candidate satisfies ALL constraints."""

from __future__ import annotations

import json

import anthropic

from ..infra.logging import get_logger
from .prompts import VERIFY_PROMPT
from .state import Candidate, Constraint, SearchFinding

logger = get_logger(__name__)

VERIFY_MODEL = "claude-sonnet-4-6"


async def verify_candidate(
    question: str,
    candidate: Candidate,
    constraints: list[Constraint],
    findings: list[SearchFinding],
) -> tuple[bool, float, list[Constraint]]:
    """Verify a candidate answer against all constraints.

    Returns:
        (all_met, confidence, updated_constraints)
    """
    client = anthropic.AsyncAnthropic()

    # Build evidence from findings that mention the candidate
    answer_lower = candidate.answer.lower()
    relevant_evidence = []
    for f in findings:
        text = f.full_content or f.snippet
        if answer_lower in text.lower() or any(
            word in text.lower() for word in answer_lower.split() if len(word) > 3
        ):
            relevant_evidence.append(f"[{f.url}] {f.title}\n  {text[:500]}")

    # Also include the candidate's own source
    if candidate.source_snippet:
        relevant_evidence.append(f"[Candidate source] {candidate.source_snippet[:500]}")

    evidence_text = "\n\n".join(relevant_evidence[:10]) if relevant_evidence else "(no direct evidence found for this candidate)"

    response = await client.messages.create(
        model=VERIFY_MODEL,
        messages=[{"role": "user", "content": VERIFY_PROMPT.format(
            question=question,
            candidate_answer=candidate.answer,
            constraints="\n".join(f"- [{c.category}] {c.text}" for c in constraints),
            evidence=evidence_text,
        )}],
        max_tokens=2048,
    )

    text = response.content[0].text.strip()

    # Parse JSON response
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("verify_parse_failed", text=text[:200])
        return False, 0.0, constraints

    all_met = result.get("all_met", False)
    confidence = float(result.get("overall_confidence", 0.0))

    # Update constraints with verification results
    updated = list(constraints)
    for item in result.get("constraints", []):
        if isinstance(item, dict):
            for c in updated:
                if c.text == item.get("text", ""):
                    c.is_met = item.get("met", False)
                    c.evidence = item.get("evidence", "")

    met_count = sum(1 for c in updated if c.is_met)
    total = len(updated)

    logger.info(
        "verification_result",
        candidate=candidate.answer[:50],
        all_met=all_met,
        confidence=confidence,
        constraints_met=f"{met_count}/{total}",
    )

    return all_met, confidence, updated
