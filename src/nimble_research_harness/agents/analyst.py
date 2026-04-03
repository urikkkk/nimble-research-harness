"""Analyst agent — synthesizes evidence into findings and report."""

from __future__ import annotations

from ..models.session import SessionConfig
from ..models.skill import DynamicSkillSpec
from ..tools.registry import ToolRegistry
from .base import FAST_MODEL, run_agent_loop

SYSTEM_PROMPT = """You are a research analyst. You have access to collected evidence from web research.
Your job is to:

1. Read all collected evidence using `read_evidence`
2. CAREFULLY extract specific data points (prices, numbers, names, dates, URLs, quotes) from evidence
3. Record individual claims with `write_claim`, linking them to supporting evidence IDs
4. Write a comprehensive report using `write_report`

CRITICAL RULES:
- Evidence content contains real data — extract specific prices ($X.XX), dates, dollar amounts, company names, URLs
- Do NOT say "no data found" if evidence contains relevant information — look harder
- ALWAYS cite source URLs in your detailed analysis (e.g., "according to [source.com]...")
- Look for the STRONGEST evidence: official announcements, dollar amounts, direct quotes, named entities, dates
- Include verbatim quotes from evidence when they are strong signals

For each claim:
- Set confidence: "verified" (2+ sources), "partially_verified" (1 strong source), "weak_support", "unresolved"
- Link to evidence IDs that support it
- Rate importance 1-5

For the report:
- Executive summary: 2-3 sentences with the most important quantified findings
- Key findings: Most important discoveries with specific data points (as JSON array of strings)
- Detailed analysis: Full narrative organized by theme, with source citations inline
- Known unknowns: What couldn't be determined (as JSON array of strings)
- Limitations: Methodology constraints (as JSON array of strings)

Every claim must cite evidence. Never assert without support."""


async def analyze_and_report(
    config: SessionConfig,
    skill: DynamicSkillSpec,
    registry: ToolRegistry,
    fast_mode: bool = False,
) -> str:
    """Synthesize evidence into claims and produce a report."""
    user_prompt = f"""Analyze the collected research evidence and produce a report.

Research objective: {skill.user_objective}
Task type: {skill.task_type.value}
Subquestions investigated:
{chr(10).join(f'- {q}' for q in skill.subquestions)}

Steps:
1. Call `read_evidence` to see all collected data
2. For each key finding, call `write_claim` with the statement and supporting evidence IDs
3. Call `write_report` with the complete analysis

Focus on answering the research objective with cited evidence."""

    tool_names = ["read_evidence", "write_claim", "write_report"]
    kwargs = {}
    if fast_mode:
        kwargs["model"] = FAST_MODEL
    result = await run_agent_loop(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        registry=registry,
        tool_names=tool_names,
        max_turns=15,
        max_tokens=16384,
        **kwargs,
    )
    return result.text
