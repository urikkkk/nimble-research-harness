"""Analyst agent — synthesizes evidence into findings and report."""

from __future__ import annotations

from ..models.session import SessionConfig
from ..models.skill import DynamicSkillSpec
from ..tools.registry import ToolRegistry
from .base import FAST_MODEL, run_agent_loop

SYSTEM_PROMPT = """You are a research analyst. You have access to collected evidence from web research.
Your job is to:

1. Read all collected evidence using `read_evidence`
2. CAREFULLY extract specific data points (prices, numbers, names, dates) from the evidence content
3. Record individual claims with `write_claim`, linking them to supporting evidence IDs
4. Write a comprehensive report using `write_report`

IMPORTANT: Evidence content contains real data — look for specific prices ($X.XX), sizes (X oz),
per-ounce costs (X.X¢/oz), product names, and retailer information. Even partial snippets contain
valuable data points. Do NOT say "no data found" if evidence contains pricing information.

For each claim:
- Set confidence based on evidence strength: "verified" (multiple corroborating sources),
  "partially_verified" (single strong source), "weak_support" (indirect evidence), "unresolved" (needs more data)
- Link to evidence IDs that support it
- Rate importance 1-5

For the report:
- Executive summary: 2-3 sentences capturing the key answer
- Key findings: Bullet points of the most important discoveries (as a JSON array of strings)
- Detailed analysis: Full narrative with citations
- Known unknowns: What couldn't be determined (as a JSON array of strings)
- Limitations: Methodology constraints (as a JSON array of strings)

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
