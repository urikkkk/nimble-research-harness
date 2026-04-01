"""Example: Social media listening and engagement analysis.

Inspired by nimble-social-feeds skill — cross-platform data collection
from TikTok, Instagram, YouTube, and Twitter/X with normalized metrics.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

SOCIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "platforms_analyzed": {"type": "array", "items": {"type": "string"}},
        "top_content": {"type": "array", "items": {"type": "object", "properties": {
            "platform": {"type": "string"},
            "creator": {"type": "string"},
            "title": {"type": "string"},
            "views": {"type": "integer"},
            "engagement_rate": {"type": "number"},
            "url": {"type": "string"},
        }}},
        "top_creators": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "platform": {"type": "string"},
            "followers": {"type": "string"},
            "avg_engagement_rate": {"type": "number"},
        }}},
        "trending_hashtags": {"type": "array", "items": {"type": "string"}},
        "engagement_by_platform": {"type": "object"},
        "key_insights": {"type": "array", "items": {"type": "string"}},
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Social media analysis of 'sustainable fashion' trends across "
            "TikTok, Instagram, and YouTube. Find top creators, trending "
            "hashtags, and highest-engagement content from the last 14 days."
        ),
        time_budget=TimeBudget.STANDARD_10M,
        preferred_format=ReportFormat.FULL_REPORT,
        output_schema=SOCIAL_SCHEMA,
        context_hints=[
            "Normalize engagement metrics across platforms",
            "TikTok: views + likes + comments; Instagram: likes + comments + saves",
            "YouTube: views + likes + comments; weight engagement rate over raw counts",
        ],
        metadata={"vertical": "marketing", "use_case": "social_listening"},
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
