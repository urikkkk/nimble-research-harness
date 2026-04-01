"""Example: Multi-retailer product search with scoring.

Inspired by product-shopping-skill — searches 13+ retailers simultaneously
via Nimble WSA agents, scores products on customizable dimensions,
and produces a ranked comparison.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

SHOPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "budget": {"type": "string"},
        "results": {"type": "array", "items": {"type": "object", "properties": {
            "rank": {"type": "integer"},
            "product_name": {"type": "string"},
            "retailer": {"type": "string"},
            "price": {"type": "number"},
            "rating": {"type": "number"},
            "review_count": {"type": "integer"},
            "in_stock": {"type": "boolean"},
            "url": {"type": "string"},
            "score_breakdown": {"type": "object", "properties": {
                "features": {"type": "number"},
                "reviews": {"type": "number"},
                "value": {"type": "number"},
                "build_quality": {"type": "number"},
                "total": {"type": "number"},
            }},
        }}},
        "top_pick": {"type": "string"},
        "best_value": {"type": "string"},
        "retailers_searched": {"type": "array", "items": {"type": "string"}},
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Find the best ergonomic split keyboard under $300. "
            "Search Amazon, Walmart, B&H Photo, and specialty retailers. "
            "Score on: ergonomics (40%), reviews (20%), value (20%), build quality (20%). "
            "Must be mechanical with hot-swap switches."
        ),
        time_budget=TimeBudget.STANDARD_10M,
        preferred_format=ReportFormat.FULL_REPORT,
        output_schema=SHOPPING_SCHEMA,
        source_policy={
            "preferred_domains": [
                "amazon.com", "walmart.com", "bhphotovideo.com",
                "mechanicalkeyboards.com",
            ],
        },
        context_hints=[
            "Use WSA retail agents for Amazon, Walmart, B&H pricing",
            "Supplement with search for specialty ergonomic keyboard retailers",
            "Enrich top 5 results with professional review scores from rtings.com",
        ],
        metadata={"vertical": "ecommerce", "use_case": "product_shopping"},
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
