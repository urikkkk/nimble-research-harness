"""Example: Brand monitoring with sentiment analysis.

Inspired by nimble-brand-monitor skill — parallel search across news and
social channels, LLM-based sentiment classification, competitive comparison.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "brand": {"type": "string"},
        "overall_sentiment": {
            "type": "string",
            "enum": ["positive", "neutral", "mixed", "negative"],
        },
        "sentiment_score": {"type": "number"},
        "mention_count": {"type": "integer"},
        "top_positive_mentions": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "source": {"type": "string"},
                "quote": {"type": "string"},
                "sentiment": {"type": "number"},
            }},
        },
        "top_negative_mentions": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "source": {"type": "string"},
                "quote": {"type": "string"},
                "sentiment": {"type": "number"},
            }},
        },
        "trending_topics": {"type": "array", "items": {"type": "string"}},
        "competitor_comparison": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "brand": {"type": "string"},
                "sentiment_score": {"type": "number"},
            }},
        },
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Monitor brand sentiment for OpenAI over the last 7 days. "
            "Compare against Anthropic and Google DeepMind. "
            "Include top positive and negative mentions with source quotes."
        ),
        time_budget=TimeBudget.STANDARD_10M,
        preferred_format=ReportFormat.FULL_REPORT,
        output_schema=SENTIMENT_SCHEMA,
        context_hints=[
            "Search both news and social channels",
            "Classify sentiment per-mention before aggregating",
            "Weight authoritative sources (Reuters, Bloomberg, TechCrunch) higher",
        ],
        metadata={"vertical": "marketing", "use_case": "brand_monitor"},
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
