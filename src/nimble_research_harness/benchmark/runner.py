"""Benchmark runner — executes queries across multiple time budgets and collects results."""

from __future__ import annotations

import asyncio
import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..infra.logging import get_logger
from ..models.enums import ReportFormat, TimeBudget
from ..models.output import SessionSummary
from ..models.session import UserResearchRequest
from ..nimble.provider import NimbleProvider
from ..orchestrator.engine import run_research
from ..storage.json_backend import JsonStorageBackend

logger = get_logger(__name__)

BENCHMARK_BUDGETS = [
    TimeBudget.SHORT_2M,
    TimeBudget.MEDIUM_5M,
    TimeBudget.STANDARD_10M,
]


@dataclass
class QueryResult:
    """Result of running a single query at a single budget."""

    query_id: str
    query: str
    budget: str
    session_id: str = ""
    status: str = "pending"  # pending, completed, failed, timeout
    elapsed_seconds: float = 0.0
    total_evidence: int = 0
    total_sources: int = 0
    total_claims: int = 0
    verified_claims: int = 0
    total_tool_calls: int = 0
    confidence: str = ""
    final_stage: str = ""
    error: str = ""
    report_excerpt: str = ""
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "budget": self.budget,
            "session_id": self.session_id,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "total_evidence": self.total_evidence,
            "total_sources": self.total_sources,
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "total_tool_calls": self.total_tool_calls,
            "confidence": self.confidence,
            "final_stage": self.final_stage,
            "error": self.error,
            "report_excerpt": self.report_excerpt,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class BenchmarkRun:
    """Full benchmark run across multiple queries and budgets."""

    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    budgets: list[str] = field(default_factory=lambda: [b.value for b in BENCHMARK_BUDGETS])
    total_queries: int = 0
    total_runs: int = 0
    completed: int = 0
    failed: int = 0
    results: list[QueryResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "budgets": self.budgets,
            "total_queries": self.total_queries,
            "total_runs": self.total_runs,
            "completed": self.completed,
            "failed": self.failed,
            "success_rate": round(self.completed / max(self.total_runs, 1) * 100, 1),
            "results": [r.to_dict() for r in self.results],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


async def run_single_query(
    query: str,
    query_id: str,
    budget: TimeBudget,
    provider: NimbleProvider,
    storage: JsonStorageBackend,
) -> QueryResult:
    """Run a single query at a single budget and collect metrics."""

    result = QueryResult(
        query_id=query_id,
        query=query,
        budget=budget.value,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        request = UserResearchRequest(
            user_query=query,
            time_budget=budget,
            preferred_format=ReportFormat.FULL_REPORT,
            metadata={"benchmark_query_id": query_id, "benchmark_budget": budget.value},
        )

        summary = await run_research(request, provider, storage)
        sid = str(summary.session_id)

        result.session_id = sid
        result.status = "completed" if summary.final_stage.value == "completed" else "failed"
        result.elapsed_seconds = summary.elapsed_seconds
        result.total_evidence = summary.total_evidence
        result.total_sources = summary.total_sources
        result.total_claims = summary.total_claims
        result.verified_claims = summary.verified_claims
        result.total_tool_calls = summary.total_tool_calls
        result.confidence = summary.report_confidence or ""
        result.final_stage = summary.final_stage.value

        # Grab report excerpt
        report = await storage.load_report(sid)
        if report:
            result.report_excerpt = (report.executive_summary or "")[:500]

    except asyncio.TimeoutError:
        result.status = "timeout"
        result.error = f"Exceeded {budget.value} budget"
    except Exception as e:
        result.status = "failed"
        result.error = f"{type(e).__name__}: {str(e)[:300]}"
        logger.error("benchmark_query_failed", query_id=query_id, budget=budget.value, error=str(e))

    result.completed_at = datetime.now(timezone.utc).isoformat()
    return result


async def run_benchmark(
    queries: list[dict[str, str]],
    provider: NimbleProvider,
    output_dir: str = ".benchmark_runs",
    budgets: list[TimeBudget] | None = None,
    concurrency: int = 1,
    resume_run_id: str | None = None,
) -> BenchmarkRun:
    """Run a full benchmark: each query x each budget.

    Args:
        queries: List of {"id": "q001", "query": "..."} dicts
        provider: NimbleProvider (live or mock)
        output_dir: Where to store benchmark results
        budgets: Which budgets to test (default: 2m, 5m, 10m)
        concurrency: Max concurrent runs (default 1 for clean measurements)
        resume_run_id: Resume a previous run (skip already-completed query+budget combos)
    """
    budgets = budgets or BENCHMARK_BUDGETS
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Resume support
    completed_keys: set[str] = set()
    if resume_run_id:
        prev_path = out_path / resume_run_id / "results.jsonl"
        if prev_path.exists():
            for line in prev_path.read_text().strip().split("\n"):
                if line:
                    row = json.loads(line)
                    if row.get("status") == "completed":
                        completed_keys.add(f"{row['query_id']}_{row['budget']}")
            logger.info("benchmark_resume", skipping=len(completed_keys))

    run = BenchmarkRun(
        run_id=resume_run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        budgets=[b.value for b in budgets],
        total_queries=len(queries),
        total_runs=len(queries) * len(budgets),
    )

    run_dir = out_path / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save query manifest
    manifest_path = run_dir / "queries.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(queries, indent=2))

    # Results JSONL for streaming writes
    results_path = run_dir / "results.jsonl"
    results_file = open(results_path, "a")

    # Each query gets its own storage dir for isolation
    semaphore = asyncio.Semaphore(concurrency)
    total = len(queries) * len(budgets)
    done = 0

    async def _run_one(q: dict[str, str], budget: TimeBudget) -> None:
        nonlocal done
        qid = q.get("id", q.get("query", "")[:20])
        key = f"{qid}_{budget.value}"

        if key in completed_keys:
            done += 1
            logger.info("benchmark_skip", query_id=qid, budget=budget.value, reason="already completed")
            return

        async with semaphore:
            logger.info(
                "benchmark_start",
                query_id=qid,
                budget=budget.value,
                progress=f"{done + 1}/{total}",
            )

            # Isolated storage per query+budget
            storage = JsonStorageBackend(
                base_dir=str(run_dir / "sessions" / f"{qid}_{budget.value}")
            )

            result = await run_single_query(
                query=q["query"],
                query_id=qid,
                budget=budget,
                provider=provider,
                storage=storage,
            )

            run.results.append(result)
            if result.status == "completed":
                run.completed += 1
            else:
                run.failed += 1

            # Stream result to JSONL
            results_file.write(json.dumps(result.to_dict(), default=str) + "\n")
            results_file.flush()

            done += 1
            logger.info(
                "benchmark_done",
                query_id=qid,
                budget=budget.value,
                status=result.status,
                elapsed=f"{result.elapsed_seconds:.1f}s",
                evidence=result.total_evidence,
                progress=f"{done}/{total}",
            )

    # Run sequentially by budget tier (all queries at 2m, then 5m, then 10m)
    # This gives cleaner time comparisons
    for budget in budgets:
        budget_tasks = [_run_one(q, budget) for q in queries]
        if concurrency == 1:
            for task in budget_tasks:
                await task
        else:
            await asyncio.gather(*budget_tasks)

    results_file.close()
    run.completed_at = datetime.now(timezone.utc)

    # Write final summary
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(run.to_dict(), indent=2, default=str))

    logger.info(
        "benchmark_complete",
        run_id=run.run_id,
        total=run.total_runs,
        completed=run.completed,
        failed=run.failed,
    )

    return run
