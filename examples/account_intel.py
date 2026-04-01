"""Example: Account intelligence card for sales teams.

Inspired by nimble-account-intel skill — SDR-ready research card with
company overview, recent signals, key people, and talking points.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

ACCOUNT_INTEL_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "industry": {"type": "string"},
        "hq_location": {"type": "string"},
        "employee_count": {"type": "string"},
        "funding_stage": {"type": "string"},
        "recent_signals": {"type": "array", "items": {"type": "object", "properties": {
            "signal": {"type": "string"},
            "date": {"type": "string"},
            "source": {"type": "string"},
        }}},
        "key_people": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "title": {"type": "string"},
            "recent_activity": {"type": "string"},
        }}},
        "competitors": {"type": "array", "items": {"type": "string"}},
        "talking_points": {"type": "array", "items": {"type": "string"}},
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Account intelligence on Datadog: company overview, recent funding or "
            "product launches, key decision-makers in engineering leadership, "
            "and talking points for a sales meeting."
        ),
        time_budget=TimeBudget.MEDIUM_5M,
        preferred_format=ReportFormat.BRIEF,
        output_schema=ACCOUNT_INTEL_SCHEMA,
        context_hints=[
            "Focus on signals from the last 30 days",
            "Search news, general web, and social for key people",
            "Include LinkedIn-style professional context for contacts",
        ],
        fast_mode=True,  # Sales teams need quick results
        metadata={"vertical": "sales", "use_case": "account_intel"},
    )

    provider = MockNimbleProvider()
    storage = JsonStorageBackend()

    summary = await run_research(request, provider, storage)
    print(format_summary(summary))
    print()

    report = await storage.load_report(str(summary.session_id))
    if report:
        print(format_report(report, ReportFormat.BRIEF))


if __name__ == "__main__":
    asyncio.run(main())
