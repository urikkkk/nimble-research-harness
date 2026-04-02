"""Benchmark analyzer — reads results and produces scorecard + insights."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_results(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load all results from a benchmark run's JSONL file."""
    results_path = Path(run_dir) / "results.jsonl"
    if not results_path.exists():
        return []
    results = []
    for line in results_path.read_text().strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))
    return results


def build_scorecard(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a scorecard comparing performance across budgets."""

    by_budget: dict[str, list[dict]] = defaultdict(list)
    by_query: dict[str, list[dict]] = defaultdict(list)

    for r in results:
        by_budget[r["budget"]].append(r)
        by_query[r["query_id"]].append(r)

    # Per-budget aggregates
    budget_stats = {}
    for budget, rows in sorted(by_budget.items()):
        completed = [r for r in rows if r["status"] == "completed"]
        budget_stats[budget] = {
            "total": len(rows),
            "completed": len(completed),
            "failed": len(rows) - len(completed),
            "success_rate": round(len(completed) / max(len(rows), 1) * 100, 1),
            "avg_elapsed_seconds": round(
                sum(r["elapsed_seconds"] for r in completed) / max(len(completed), 1), 1
            ),
            "avg_evidence": round(
                sum(r["total_evidence"] for r in completed) / max(len(completed), 1), 1
            ),
            "avg_sources": round(
                sum(r["total_sources"] for r in completed) / max(len(completed), 1), 1
            ),
            "avg_claims": round(
                sum(r["total_claims"] for r in completed) / max(len(completed), 1), 1
            ),
            "avg_verified_claims": round(
                sum(r["verified_claims"] for r in completed) / max(len(completed), 1), 1
            ),
            "avg_tool_calls": round(
                sum(r["total_tool_calls"] for r in completed) / max(len(completed), 1), 1
            ),
            "confidence_distribution": _confidence_dist(completed),
        }

    # Per-query comparison across budgets
    query_comparisons = []
    for qid, rows in sorted(by_query.items()):
        comp = {"query_id": qid, "query": rows[0]["query"][:80]}
        for r in rows:
            comp[r["budget"]] = {
                "status": r["status"],
                "elapsed": round(r["elapsed_seconds"], 1),
                "evidence": r["total_evidence"],
                "sources": r["total_sources"],
                "claims": r["total_claims"],
                "confidence": r["confidence"],
            }
        query_comparisons.append(comp)

    # Budget scaling analysis: does more time = more evidence?
    scaling = _analyze_scaling(by_budget)

    # Failure analysis
    failures = [
        {
            "query_id": r["query_id"],
            "budget": r["budget"],
            "error": r["error"][:200],
            "final_stage": r["final_stage"],
        }
        for r in results
        if r["status"] != "completed"
    ]

    return {
        "total_runs": len(results),
        "total_completed": sum(1 for r in results if r["status"] == "completed"),
        "total_failed": sum(1 for r in results if r["status"] != "completed"),
        "budget_stats": budget_stats,
        "query_comparisons": query_comparisons,
        "scaling_analysis": scaling,
        "failures": failures,
    }


def _confidence_dist(rows: list[dict]) -> dict[str, int]:
    """Count confidence levels."""
    dist: dict[str, int] = defaultdict(int)
    for r in rows:
        conf = r.get("confidence", "none") or "none"
        dist[conf] += 1
    return dict(dist)


def _analyze_scaling(by_budget: dict[str, list[dict]]) -> dict[str, Any]:
    """Analyze how metrics scale with budget."""
    budgets_ordered = ["2m", "5m", "10m", "30m", "1h"]
    metrics = ["total_evidence", "total_sources", "total_claims", "verified_claims", "total_tool_calls"]

    scaling = {}
    for metric in metrics:
        values = {}
        for b in budgets_ordered:
            if b in by_budget:
                completed = [r for r in by_budget[b] if r["status"] == "completed"]
                if completed:
                    values[b] = round(
                        sum(r[metric] for r in completed) / len(completed), 1
                    )
        if len(values) >= 2:
            keys = list(values.keys())
            first_val = values[keys[0]]
            last_val = values[keys[-1]]
            if first_val > 0:
                scaling[metric] = {
                    "by_budget": values,
                    "improvement_factor": round(last_val / first_val, 2),
                }
            else:
                scaling[metric] = {"by_budget": values, "improvement_factor": 0}

    return scaling


def format_scorecard_text(scorecard: dict[str, Any]) -> str:
    """Format scorecard as human-readable text."""
    lines = [
        "=" * 70,
        "BENCHMARK SCORECARD",
        "=" * 70,
        f"Total runs: {scorecard['total_runs']} | "
        f"Completed: {scorecard['total_completed']} | "
        f"Failed: {scorecard['total_failed']}",
        "",
    ]

    # Budget stats table
    lines.append("--- Per-Budget Performance ---")
    lines.append(f"{'Budget':<8} {'OK/Total':<10} {'Avg Time':<10} {'Evidence':<10} "
                 f"{'Sources':<10} {'Claims':<10} {'Verified':<10} {'Tool Calls':<12}")
    lines.append("-" * 80)

    for budget, stats in scorecard["budget_stats"].items():
        lines.append(
            f"{budget:<8} {stats['completed']}/{stats['total']:<8} "
            f"{stats['avg_elapsed_seconds']:<10.1f} {stats['avg_evidence']:<10.1f} "
            f"{stats['avg_sources']:<10.1f} {stats['avg_claims']:<10.1f} "
            f"{stats['avg_verified_claims']:<10.1f} {stats['avg_tool_calls']:<12.1f}"
        )

    # Confidence distribution
    lines.append("")
    lines.append("--- Confidence Distribution ---")
    for budget, stats in scorecard["budget_stats"].items():
        dist = stats["confidence_distribution"]
        dist_str = " | ".join(f"{k}: {v}" for k, v in sorted(dist.items()))
        lines.append(f"  {budget}: {dist_str}")

    # Scaling analysis
    if scorecard.get("scaling_analysis"):
        lines.append("")
        lines.append("--- Budget Scaling (more time = more data?) ---")
        for metric, data in scorecard["scaling_analysis"].items():
            by_b = " → ".join(f"{b}={v}" for b, v in data["by_budget"].items())
            lines.append(f"  {metric}: {by_b} (x{data['improvement_factor']})")

    # Failures
    if scorecard.get("failures"):
        lines.append("")
        lines.append(f"--- Failures ({len(scorecard['failures'])}) ---")
        for f in scorecard["failures"][:20]:
            lines.append(f"  [{f['budget']}] {f['query_id']}: {f['error'][:80]} (stage: {f['final_stage']})")

    # Query comparison (top 10)
    if scorecard.get("query_comparisons"):
        lines.append("")
        lines.append("--- Per-Query Comparison (sample) ---")
        for comp in scorecard["query_comparisons"][:10]:
            lines.append(f"\n  {comp['query_id']}: {comp['query']}")
            for budget in ["2m", "5m", "10m"]:
                if budget in comp:
                    d = comp[budget]
                    lines.append(
                        f"    {budget}: {d['status']} | {d['elapsed']:.0f}s | "
                        f"ev={d['evidence']} src={d['sources']} claims={d['claims']} "
                        f"conf={d['confidence']}"
                    )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def format_scorecard_csv(scorecard: dict[str, Any]) -> str:
    """Format per-query results as CSV for spreadsheet analysis."""
    lines = [
        "query_id,query,budget,status,elapsed_s,evidence,sources,claims,verified,tool_calls,confidence"
    ]
    for comp in scorecard["query_comparisons"]:
        for budget in ["2m", "5m", "10m"]:
            if budget in comp:
                d = comp[budget]
                q = comp["query"].replace(",", ";").replace('"', "'")
                lines.append(
                    f'{comp["query_id"]},"{q}",{budget},{d["status"]},'
                    f'{d["elapsed"]},{d["evidence"]},{d["sources"]},'
                    f'{d["claims"]},0,0,{d["confidence"]}'
                )
    return "\n".join(lines)
