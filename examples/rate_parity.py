"""Example: Hotel rate parity check across OTAs.

Inspired by nimble-rate-intel skill — compares hotel rates across Booking.com,
Expedia, Hotels.com, and direct channels to detect parity violations.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

RATE_SCHEMA = {
    "type": "object",
    "properties": {
        "property_name": {"type": "string"},
        "location": {"type": "string"},
        "check_in": {"type": "string"},
        "check_out": {"type": "string"},
        "rates": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"},
            "room_type": {"type": "string"},
            "nightly_rate": {"type": "number"},
            "total_cost": {"type": "number"},
            "cancellation_policy": {"type": "string"},
            "includes_breakfast": {"type": "boolean"},
        }}},
        "parity_status": {
            "type": "string",
            "enum": ["in_parity", "minor_violation", "major_violation"],
        },
        "price_spread_pct": {"type": "number"},
        "best_deal": {"type": "object", "properties": {
            "source": {"type": "string"},
            "total_cost": {"type": "number"},
            "why": {"type": "string"},
        }},
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Rate parity check for Marriott Marquis Times Square NYC, "
            "July 10-13 2026, 2 guests, standard king room. "
            "Compare rates across Booking.com, Expedia, Hotels.com, "
            "Marriott.com direct, and Google Hotels."
        ),
        time_budget=TimeBudget.STANDARD_10M,
        preferred_format=ReportFormat.FULL_REPORT,
        output_schema=RATE_SCHEMA,
        source_policy={
            "preferred_domains": [
                "booking.com", "expedia.com", "hotels.com",
                "marriott.com", "google.com/travel",
            ],
        },
        metadata={"vertical": "travel", "use_case": "rate_parity"},
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
