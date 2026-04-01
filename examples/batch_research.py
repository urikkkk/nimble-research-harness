"""Example: Batch research — run multiple queries concurrently.

Demonstrates the Parallel AI-inspired Task Groups pattern: fan-out
multiple research queries with bounded concurrency.
"""

import asyncio

from nimble_research_harness.models.enums import TimeBudget
from nimble_research_harness.orchestrator.batch import batch_research
from nimble_research_harness.nimble.mock import MockNimbleProvider
from nimble_research_harness.storage.json_backend import JsonStorageBackend


QUERIES = [
    "Due diligence on Stripe: funding, leadership, risks",
    "Brand sentiment for Tesla over the last 30 days",
    "MAP compliance for AirPods Pro on Amazon, Walmart, Target (MAP: $249)",
    "Find all coworking spaces in SoHo, Manhattan",
    "Compare hotel rates for Hilton Chicago, Aug 1-4 across Booking.com and Expedia",
]


async def main():
    provider = MockNimbleProvider()
    storage = JsonStorageBackend()

    print(f"Running batch of {len(QUERIES)} queries with concurrency=3...\n")

    result = await batch_research(
        queries=QUERIES,
        time_budget=TimeBudget.SHORT_2M,
        provider=provider,
        storage=storage,
        concurrency=3,
    )

    print(f"Batch {result.batch_id} complete:")
    print(f"  Completed: {result.completed}/{result.total_queries}")
    print(f"  Failed: {result.failed}")
    print(f"  Success rate: {result.success_rate:.0f}%")
    print()

    for i, summary in enumerate(result.summaries, 1):
        print(f"  [{i}] {summary.user_query[:50]}...")
        print(f"      Stage: {summary.final_stage.value} | "
              f"Evidence: {summary.total_evidence} | "
              f"Confidence: {summary.report_confidence or 'n/a'}")

    if result.errors:
        print("\n  Errors:")
        for err in result.errors:
            print(f"    - {err['query'][:40]}...: {err['error']}")


if __name__ == "__main__":
    asyncio.run(main())
