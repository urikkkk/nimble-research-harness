"""Example: MAP (Minimum Advertised Price) compliance monitoring.

Inspired by nimble-catalog-compliance skill — checks retailer pricing
against MAP policy, flags violations by severity tier.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

MAP_SCHEMA = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "map_price": {"type": "number"},
        "retailers": {"type": "array", "items": {"type": "object", "properties": {
            "retailer": {"type": "string"},
            "listed_price": {"type": "number"},
            "seller_name": {"type": "string"},
            "seller_type": {"type": "string", "enum": ["1P", "3P", "unknown"]},
            "deviation_pct": {"type": "number"},
            "violation_tier": {
                "type": "string",
                "enum": ["compliant", "warning", "minor", "major", "egregious"],
            },
        }}},
        "compliance_summary": {"type": "object", "properties": {
            "compliant_count": {"type": "integer"},
            "warning_count": {"type": "integer"},
            "violation_count": {"type": "integer"},
        }},
        "unauthorized_sellers": {"type": "array", "items": {"type": "string"}},
        "recommended_actions": {"type": "array", "items": {"type": "string"}},
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "MAP compliance check for Dyson V15 Detect (MAP: $749.99) "
            "across Amazon, Walmart, Best Buy, Target, and Home Depot. "
            "Flag any 3P sellers undercutting MAP. "
            "Violation tiers: warning (0-5% below), minor (5-15%), "
            "major (15-30%), egregious (>30%)."
        ),
        time_budget=TimeBudget.MEDIUM_5M,
        preferred_format=ReportFormat.EVIDENCE_TABLE,
        output_schema=MAP_SCHEMA,
        source_policy={
            "preferred_domains": [
                "amazon.com", "walmart.com", "bestbuy.com",
                "target.com", "homedepot.com",
            ],
        },
        metadata={"vertical": "ecommerce", "use_case": "map_compliance"},
    )

    provider = MockNimbleProvider()
    storage = JsonStorageBackend()

    summary = await run_research(request, provider, storage)
    print(format_summary(summary))
    print()

    report = await storage.load_report(str(summary.session_id))
    if report:
        print(format_report(report, ReportFormat.EVIDENCE_TABLE))


if __name__ == "__main__":
    asyncio.run(main())
