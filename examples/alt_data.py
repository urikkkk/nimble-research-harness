"""Example: Alternative data collection for investment analysis.

Inspired by nimble-alt-data skill — routes by data type (jobs, reviews,
app rankings, web traffic, SEC filings) to optimal collection methods.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

ALT_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "signals": {"type": "array", "items": {"type": "object", "properties": {
            "data_type": {
                "type": "string",
                "enum": ["jobs", "reviews", "app_rankings", "web_traffic", "sec_filings", "social"],
            },
            "signal": {"type": "string"},
            "trend": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "detail": {"type": "string"},
        }}},
        "hiring_summary": {"type": "object", "properties": {
            "total_open_roles": {"type": "integer"},
            "top_departments": {"type": "array", "items": {"type": "string"}},
            "yoy_change": {"type": "string"},
        }},
        "employee_sentiment": {"type": "object", "properties": {
            "glassdoor_rating": {"type": "number"},
            "recommend_to_friend_pct": {"type": "number"},
            "top_pros": {"type": "array", "items": {"type": "string"}},
            "top_cons": {"type": "array", "items": {"type": "string"}},
        }},
        "overall_assessment": {
            "type": "string",
            "enum": ["strongly_bullish", "bullish", "neutral", "bearish", "strongly_bearish"],
        },
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Alternative data analysis for Snowflake vs Databricks: "
            "job postings (hiring velocity by department), employee reviews "
            "(Glassdoor sentiment), and social media mentions. "
            "Generate investment signals with bull/bear assessment."
        ),
        time_budget=TimeBudget.DEEP_30M,
        preferred_format=ReportFormat.FULL_REPORT,
        output_schema=ALT_DATA_SCHEMA,
        context_hints=[
            "Route by data type: jobs → LinkedIn/Indeed, reviews → Glassdoor, social → Twitter/Reddit",
            "Compare both companies side-by-side on each signal",
            "Flag divergences (e.g., heavy hiring but declining sentiment = red flag)",
        ],
        metadata={"vertical": "finance", "use_case": "alt_data"},
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
