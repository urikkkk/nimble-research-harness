"""Constraint decomposition — break multi-hop questions into searchable constraints."""

from __future__ import annotations

import json

import anthropic

from ..infra.logging import get_logger
from .prompts import DECOMPOSE_PROMPT
from .state import Constraint

logger = get_logger(__name__)

DECOMPOSE_MODEL = "claude-sonnet-4-6"


async def decompose_question(question: str) -> list[Constraint]:
    """Break a multi-hop question into independent searchable constraints."""
    client = anthropic.AsyncAnthropic()

    response = await client.messages.create(
        model=DECOMPOSE_MODEL,
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}],
        max_tokens=2048,
    )

    text = response.content[0].text.strip()

    # Parse JSON array from response (handle markdown code fences)
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("decompose_parse_failed", text=text[:200])
        # Fallback: treat the whole question as one constraint
        return [Constraint(text=question, category="general")]

    constraints = []
    for item in raw:
        if isinstance(item, dict):
            constraints.append(Constraint(
                text=item.get("text", ""),
                category=item.get("category", ""),
            ))
        elif isinstance(item, str):
            constraints.append(Constraint(text=item))

    logger.info("decomposed", count=len(constraints), categories=[c.category for c in constraints])
    return constraints
