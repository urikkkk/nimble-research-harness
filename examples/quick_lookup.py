"""Example: 30-second quick lookup."""

import asyncio

from nimble_research_harness.models.enums import TimeBudget
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.orchestrator.engine import run_research
from nimble_research_harness.reports.formatter import format_summary
from nimble_research_harness.storage.json_backend import JsonStorageBackend


async def main():
    request = UserResearchRequest(
        user_query="What is Nimble's pricing for web scraping API?",
        time_budget=TimeBudget.QUICK_30S,
    )

    provider = MockNimbleProvider()
    storage = JsonStorageBackend()

    summary = await run_research(request, provider, storage)
    print(format_summary(summary))


if __name__ == "__main__":
    asyncio.run(main())
