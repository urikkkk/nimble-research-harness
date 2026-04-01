"""Example: Travel research with property discovery and price comparison.

Inspired by nimble-travel-research skill — discovers top-rated properties
via review aggregators, then gets per-property pricing from multiple OTAs.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

TRAVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {"type": "string"},
        "dates": {"type": "object", "properties": {
            "check_in": {"type": "string"},
            "check_out": {"type": "string"},
            "nights": {"type": "integer"},
        }},
        "top_properties": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "stars": {"type": "integer"},
            "rating": {"type": "number"},
            "review_count": {"type": "integer"},
            "price_per_night": {"type": "number"},
            "total_estimated": {"type": "number"},
            "best_source": {"type": "string"},
            "neighborhood": {"type": "string"},
        }}},
        "best_value_pick": {"type": "string"},
        "budget_pick": {"type": "string"},
        "luxury_pick": {"type": "string"},
        "area_insights": {"type": "array", "items": {"type": "string"}},
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Find the best hotels in Barcelona for June 15-20, 2026 (2 guests). "
            "Budget: $150-300/night. Prefer boutique hotels in Gothic Quarter or "
            "Eixample. Compare prices across Booking.com, Expedia, and direct sites."
        ),
        time_budget=TimeBudget.STANDARD_10M,
        preferred_format=ReportFormat.FULL_REPORT,
        output_schema=TRAVEL_SCHEMA,
        source_policy={
            "preferred_domains": [
                "tripadvisor.com", "booking.com", "expedia.com",
                "hotels.com", "usnews.com/travel",
            ],
        },
        context_hints=[
            "Phase 1: Discover top-rated properties via review aggregators",
            "Phase 2: Get per-property pricing from OTAs",
            "Include neighborhood context and walkability notes",
        ],
        metadata={"vertical": "travel", "use_case": "hotel_search"},
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
