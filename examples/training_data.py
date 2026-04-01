"""Example: Web content collection for AI/ML training data.

Inspired by nimble-training-data skill — orchestrates map → crawl → extract
pipeline to collect clean structured content from documentation sites.
"""

import asyncio

from nimble_research_harness.models.enums import ReportFormat, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_report, format_summary
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend

TRAINING_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "target_url": {"type": "string"},
        "total_pages_collected": {"type": "integer"},
        "total_words": {"type": "integer"},
        "content_sections": {"type": "array", "items": {"type": "object", "properties": {
            "url": {"type": "string"},
            "title": {"type": "string"},
            "word_count": {"type": "integer"},
            "content_type": {
                "type": "string",
                "enum": ["documentation", "tutorial", "reference", "blog", "other"],
            },
        }}},
        "collection_stats": {"type": "object", "properties": {
            "pages_discovered": {"type": "integer"},
            "pages_extracted": {"type": "integer"},
            "avg_word_count": {"type": "integer"},
            "extraction_rate_pct": {"type": "number"},
        }},
    },
}


async def main():
    request = UserResearchRequest(
        user_query=(
            "Collect all documentation pages from https://docs.anthropic.com "
            "for use as training data. Map the site structure first, then "
            "extract each page as clean markdown. Limit to 100 pages."
        ),
        time_budget=TimeBudget.DEEP_30M,
        preferred_format=ReportFormat.JSON,
        output_schema=TRAINING_DATA_SCHEMA,
        source_policy={
            "preferred_domains": ["docs.anthropic.com"],
            "disallowed_domains": [],  # Stay on-domain
        },
        context_hints=[
            "Phase 1: nimble_map to discover site structure",
            "Phase 2: nimble_crawl for bulk collection (>50 pages) or nimble_extract for <50",
            "Output clean markdown, skip navigation/footer boilerplate",
        ],
        metadata={"vertical": "ai_ml", "use_case": "training_data"},
    )

    provider = MockNimbleProvider()
    storage = JsonStorageBackend()

    summary = await run_research(request, provider, storage)
    print(format_summary(summary))


if __name__ == "__main__":
    asyncio.run(main())
