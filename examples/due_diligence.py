"""Example: Investment due diligence research brief.

Inspired by nimble-due-diligence skill — multi-pass search across news,
SEC filings, business intelligence, key personnel, and risk signals.
Uses deep 30m budget for comprehensive coverage.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

DD_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "industry": {"type": "string"},
        "founded": {"type": "string"},
        "headquarters": {"type": "string"},
        "funding_total": {"type": "string"},
        "last_round": {"type": "object", "properties": {
            "type": {"type": "string"},
            "amount": {"type": "string"},
            "date": {"type": "string"},
            "lead_investor": {"type": "string"},
        }},
        "key_personnel": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "role": {"type": "string"},
            "background": {"type": "string"},
        }}},
        "recent_developments": {"type": "array", "items": {"type": "string"}},
        "risk_factors": {"type": "array", "items": {"type": "object", "properties": {
            "risk": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        }}},
        "competitive_landscape": {"type": "array", "items": {"type": "string"}},
        "investment_thesis": {"type": "string"},
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Due diligence on Stripe: company overview, funding history, "
            "key personnel, recent developments, regulatory risks, "
            "competitive landscape, and investment thesis."
        ),
        time_budget=TimeBudget.DEEP_30M,
        preferred_format=ReportFormat.FULL_REPORT,
        output_schema=DD_SCHEMA,
        context_hints=[
            "Multi-pass: news, SEC filings, business intel, leadership bios, risk signals",
            "Prioritize authoritative financial sources (Bloomberg, PitchBook, Crunchbase)",
            "Flag any ongoing litigation or regulatory investigations",
        ],
        metadata={"vertical": "finance", "use_case": "due_diligence"},
    )

    provider = MockNimbleProvider()
    storage = JsonStorageBackend()

    summary = await run_research(request, provider, storage)
    print(format_summary(summary))
    print()

    report = await storage.load_report(str(summary.session_id))
    if report:
        print(format_report(report))

    # Export the skill spec to see what the system generated
    skill = await storage.load_skill(str(summary.session_id))
    if skill:
        from nimble_research_harness.skillgen.exporter import export_skill_markdown
        print("\n--- Generated Skill (Markdown) ---")
        print(export_skill_markdown(skill))


if __name__ == "__main__":
    asyncio.run(main())
