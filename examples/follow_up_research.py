"""Example: Follow-up research — chain queries with prior session context.

Demonstrates the Parallel AI-inspired Interactions pattern: run initial
research, then deepen specific findings without re-running the original.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.orchestrator.followup import follow_up_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend


async def main():
    provider = MockNimbleProvider()
    storage = JsonStorageBackend()

    # --- Step 1: Initial broad research ---
    print("=" * 60)
    print("STEP 1: Initial Research")
    print("=" * 60)

    request = UserResearchRequest(
        user_query="Overview of the cloud data warehouse market: key players, pricing models, and growth trends",
        time_budget=TimeBudget.MEDIUM_5M,
        preferred_format=ReportFormat.BRIEF,
    )

    summary = await run_research(request, provider, storage)
    session_id = str(summary.session_id)
    print(format_summary(summary))

    # --- Step 2: Follow-up to deepen specific area ---
    print("\n" + "=" * 60)
    print("STEP 2: Follow-up — Deeper dive on Snowflake vs Databricks")
    print("=" * 60)

    followup_summary = await follow_up_research(
        previous_session_id=session_id,
        new_query=(
            "Deep comparison of Snowflake vs Databricks: "
            "TCO for a 10TB workload, performance benchmarks, "
            "ML/AI capabilities, and enterprise adoption trends"
        ),
        time_budget=TimeBudget.STANDARD_10M,
        provider=provider,
        storage=storage,
    )

    print(format_summary(followup_summary))

    report = await storage.load_report(str(followup_summary.session_id))
    if report:
        print()
        print(format_report(report, ReportFormat.BRIEF))


if __name__ == "__main__":
    asyncio.run(main())
