"""Example: 1-hour deep investigation."""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.storage.json_backend import JsonStorageBackend


async def main():
    request = UserResearchRequest(
        user_query="Comprehensive analysis of the AI agent ecosystem: key players, architectures, funding trends, and market projections for 2025-2027",
        time_budget=TimeBudget.EXHAUSTIVE_1H,
        preferred_format=ReportFormat.FULL_REPORT,
        context_hints=["focus on developer tools", "include open source projects"],
    )

    provider = MockNimbleProvider()
    storage = JsonStorageBackend()

    summary = await run_research(request, provider, storage)
    print(format_summary(summary))
    print()

    report = await storage.load_report(str(summary.session_id))
    if report:
        print(format_report(report))

    # Export the generated skill spec
    skill = await storage.load_skill(str(summary.session_id))
    if skill:
        print("\n--- Generated Skill Spec ---")
        import json
        print(json.dumps(skill.model_dump(mode="json"), indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
