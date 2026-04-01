"""Example: Location intelligence — discover and enrich local businesses.

Inspired by find-all-locations skill — discovers every place of a given type
in a neighborhood, enriches with ratings, contact, and social presence.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

LOCATION_SCHEMA = {
    "type": "object",
    "properties": {
        "place_type": {"type": "string"},
        "area": {"type": "string"},
        "total_found": {"type": "integer"},
        "places": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "address": {"type": "string"},
            "rating": {"type": "number"},
            "review_count": {"type": "integer"},
            "phone": {"type": "string"},
            "website": {"type": "string"},
            "instagram": {"type": "string"},
            "price_level": {"type": "string"},
            "highlights": {"type": "array", "items": {"type": "string"}},
        }}},
        "area_insights": {"type": "array", "items": {"type": "string"}},
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Find all specialty coffee shops in Williamsburg, Brooklyn. "
            "Include ratings, contact info, Instagram handles, and what "
            "makes each one unique. Sort by rating."
        ),
        time_budget=TimeBudget.MEDIUM_5M,
        preferred_format=ReportFormat.FULL_REPORT,
        output_schema=LOCATION_SCHEMA,
        context_hints=[
            "Use location-focused search to discover all places",
            "Enrich each with social media profiles and review highlights",
            "Include opening hours and neighborhood context",
        ],
        metadata={"vertical": "location_intel", "use_case": "place_discovery"},
    )

    provider = MockNimbleProvider()
    storage = JsonStorageBackend()

    summary = await run_research(request, provider, storage)
    print(format_summary(summary))
    print()

    report = await storage.load_report(str(summary.session_id))
    if report:
        print(format_report(report))


if __name__ == "__main__":
    asyncio.run(main())
