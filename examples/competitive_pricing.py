"""Example: Competitive pricing intelligence across retailers.

Inspired by nimble-competitive-pricing skill — collects product pricing
from Amazon, Walmart, and Target using WSA agents, then compares.
Uses structured output schema to get a clean pricing table.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.nimble.client import NimbleClient
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.storage.json_backend import JsonStorageBackend

# Structured output schema — forces the report into a pricing comparison table
PRICING_SCHEMA = {
    "type": "object",
    "properties": {
        "product_name": {"type": "string"},
        "retailers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "retailer": {"type": "string"},
                    "price": {"type": "number"},
                    "in_stock": {"type": "boolean"},
                    "seller_type": {"type": "string", "enum": ["1P", "3P", "unknown"]},
                    "url": {"type": "string"},
                },
            },
        },
        "lowest_price": {"type": "number"},
        "price_spread_pct": {"type": "number"},
        "recommendation": {"type": "string"},
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Compare prices for Sony WH-1000XM5 headphones across Amazon, Walmart, "
            "and Target. Include stock status, seller type (1P vs 3P), and identify "
            "the best deal."
        ),
        time_budget=TimeBudget.MEDIUM_5M,
        preferred_format=ReportFormat.FULL_REPORT,
        output_schema=PRICING_SCHEMA,
        source_policy={
            "preferred_domains": ["amazon.com", "walmart.com", "target.com"],
        },
        metadata={"vertical": "retail", "use_case": "competitive_pricing"},
    )

    provider = MockNimbleProvider()  # Replace with NimbleClient for live data
    storage = JsonStorageBackend()

    summary = await run_research(request, provider, storage)
    print(format_summary(summary))
    print()

    report = await storage.load_report(str(summary.session_id))
    if report:
        print(format_report(report))


if __name__ == "__main__":
    asyncio.run(main())
