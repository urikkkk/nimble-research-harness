"""BrowseComp benchmark runner and scorer.

Runs OpenAI's BrowseComp (1,266 web browsing questions) through the research harness,
scores answers using an LLM judge, and produces detailed analysis.
"""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import anthropic

from ..infra.logging import get_logger
from ..models.enums import ReportFormat, TimeBudget
from ..models.session import UserResearchRequest
from ..nimble.provider import NimbleProvider
from ..orchestrator.engine import run_research
from ..storage.json_backend import JsonStorageBackend

logger = get_logger(__name__)

# --- Decryption (from OpenAI's browsecomp_eval.py) ---


def _derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def load_browsecomp(csv_path: str, limit: int | None = None) -> list[dict[str, str]]:
    """Load and decrypt BrowseComp questions from CSV."""
    questions = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            questions.append({
                "id": f"bc{i+1:04d}",
                "question": _decrypt(row["problem"], row["canary"]),
                "answer": _decrypt(row["answer"], row["canary"]),
                "topic": row.get("problem_topic", ""),
            })
    return questions


# --- Query template (from OpenAI's eval) ---

QUERY_TEMPLATE = """Research the following question thoroughly using web search. Find the specific, precise answer.

{question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""


# --- LLM Grader ---

GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


async def grade_answer(
    question: str,
    correct_answer: str,
    response: str,
    grader_model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """Grade a response using an LLM judge."""
    client = anthropic.AsyncAnthropic()
    prompt = GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )

    try:
        result = await client.messages.create(
            model=grader_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        text = result.content[0].text

        correct_match = re.search(r"correct:\s*(yes|no)", text, re.IGNORECASE)
        is_correct = correct_match.group(1).lower() == "yes" if correct_match else False

        answer_match = re.search(r"extracted_final_answer:\s*(.+?)(?:\n|$)", text)
        extracted = answer_match.group(1).strip() if answer_match else ""

        conf_match = re.search(r"confidence:\s*(\d+)", text)
        confidence = int(conf_match.group(1)) if conf_match else 0

        return {
            "is_correct": is_correct,
            "extracted_answer": extracted,
            "confidence": confidence,
            "grader_reasoning": text,
        }
    except Exception as e:
        logger.error("grader_failed", error=str(e))
        return {
            "is_correct": False,
            "extracted_answer": "",
            "confidence": 0,
            "grader_reasoning": f"Grader error: {e}",
        }


# --- Result dataclass ---


@dataclass
class BrowseCompResult:
    id: str
    question: str
    correct_answer: str
    topic: str = ""
    budget: str = ""
    session_id: str = ""
    status: str = "pending"
    model_response: str = ""
    extracted_answer: str = ""
    is_correct: bool = False
    confidence: int = 0
    grader_reasoning: str = ""
    elapsed_seconds: float = 0.0
    total_evidence: int = 0
    total_sources: int = 0
    total_tool_calls: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "topic": self.topic,
            "budget": self.budget,
            "session_id": self.session_id,
            "status": self.status,
            "model_response": self.model_response[:500],
            "extracted_answer": self.extracted_answer,
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "grader_reasoning": self.grader_reasoning[:500],
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "total_evidence": self.total_evidence,
            "total_sources": self.total_sources,
            "total_tool_calls": self.total_tool_calls,
            "error": self.error,
        }


@dataclass
class BrowseCompRun:
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    budget: str = "2m"
    total_questions: int = 0
    completed: int = 0
    correct: int = 0
    failed: int = 0
    results: list[BrowseCompResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    @property
    def accuracy(self) -> float:
        if self.completed == 0:
            return 0.0
        return self.correct / self.completed * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "budget": self.budget,
            "total_questions": self.total_questions,
            "completed": self.completed,
            "correct": self.correct,
            "failed": self.failed,
            "accuracy": round(self.accuracy, 2),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# --- Main runner ---


async def run_browsecomp(
    questions: list[dict[str, str]],
    provider: NimbleProvider,
    budget: TimeBudget = TimeBudget.SHORT_2M,
    output_dir: str = ".browsecomp_runs",
    concurrency: int = 2,
    grader_model: str = "claude-sonnet-4-6",
    resume_run_id: str | None = None,
    mode: str = "standard",  # "standard" or "deep"
) -> BrowseCompRun:
    """Run BrowseComp benchmark through the research harness.

    For each question:
    1. Run research pipeline (standard or deep mode) with the question
    2. Extract the answer from the report/session
    3. Grade against ground truth using LLM judge

    mode="deep" uses the multi-hop deep research engine instead of the standard pipeline.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    run = BrowseCompRun(
        run_id=resume_run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        budget=budget.value,
        total_questions=len(questions),
    )

    run_dir = out_path / run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save questions manifest
    manifest_path = run_dir / "questions.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(questions, indent=2, ensure_ascii=False))

    # Resume support — skip already-graded questions
    completed_ids: set[str] = set()
    results_path = run_dir / "results.jsonl"
    if results_path.exists():
        for line in results_path.read_text().strip().split("\n"):
            if line.strip():
                row = json.loads(line)
                if row.get("status") in ("correct", "incorrect", "failed"):
                    completed_ids.add(row["id"])
                    # Rebuild counters
                    run.completed += 1
                    if row.get("is_correct"):
                        run.correct += 1
                    if row.get("status") == "failed":
                        run.failed += 1
        logger.info("browsecomp_resume", skipping=len(completed_ids))

    results_file = open(results_path, "a")
    semaphore = asyncio.Semaphore(concurrency)
    done = len(completed_ids)
    total = len(questions)

    async def _run_one(q: dict[str, str]) -> None:
        nonlocal done
        if q["id"] in completed_ids:
            done += 1
            return

        async with semaphore:
            result = BrowseCompResult(
                id=q["id"],
                question=q["question"],
                correct_answer=q["answer"],
                topic=q.get("topic", ""),
                budget=budget.value,
            )

            logger.info("browsecomp_start", id=q["id"], progress=f"{done+1}/{total}")

            try:
                if mode == "deep":
                    # Deep research mode: multi-hop constraint-driven search
                    from ..deepresearch.engine import deep_research

                    timeout = budget.seconds - 60  # Leave margin
                    dr_session = await deep_research(
                        question=q["question"],
                        provider=provider,
                        max_hops=5,
                        max_queries_per_hop=6,
                        max_parallel=4,
                        extract_top_n=3,
                        timeout_seconds=max(120, timeout),
                    )
                    result.elapsed_seconds = dr_session.elapsed_seconds
                    result.total_evidence = len([f for h in dr_session.hops for f in h.findings])
                    result.total_sources = dr_session.total_searches
                    result.total_tool_calls = dr_session.total_searches + dr_session.total_extracts

                    # Build response from deep research session
                    if dr_session.final_answer:
                        result.model_response = (
                            f"Explanation: After {len(dr_session.hops)} hops of multi-hop research, "
                            f"decomposing the question into {len(dr_session.constraints)} constraints "
                            f"and evaluating {len(dr_session.candidates)} candidates.\n\n"
                            f"Exact Answer: {dr_session.final_answer}\n"
                            f"Confidence: {int(dr_session.final_confidence * 100)}%"
                        )
                    else:
                        best = dr_session.best_candidate
                        if best:
                            result.model_response = (
                                f"Explanation: Best guess after {len(dr_session.hops)} hops. "
                                f"Candidate meets {len(best.constraints_met)} constraints.\n\n"
                                f"Exact Answer: {best.answer}\n"
                                f"Confidence: {int(best.confidence * 100)}%"
                            )
                        else:
                            result.model_response = (
                                f"Explanation: No candidate found after {len(dr_session.hops)} hops.\n\n"
                                f"Exact Answer: None\n"
                                f"Confidence: 0%"
                            )

                    # Save deep research session
                    session_dir = run_dir / "sessions" / q["id"]
                    session_dir.mkdir(parents=True, exist_ok=True)
                    (session_dir / "deep_session.json").write_text(
                        json.dumps(dr_session.to_dict(), indent=2, default=str)
                    )

                else:
                    # Standard mode: full research pipeline
                    storage = JsonStorageBackend(
                        base_dir=str(run_dir / "sessions" / q["id"])
                    )
                    request = UserResearchRequest(
                        user_query=q["question"],
                        time_budget=budget,
                        preferred_format=ReportFormat.FULL_REPORT,
                        metadata={"browsecomp_id": q["id"]},
                    )

                    summary = await run_research(request, provider, storage)
                    result.session_id = str(summary.session_id)
                    result.elapsed_seconds = summary.elapsed_seconds
                    result.total_evidence = summary.total_evidence
                    result.total_sources = summary.total_sources
                    result.total_tool_calls = summary.total_tool_calls

                    # Extract answer from report
                    report = await storage.load_report(str(summary.session_id))
                    if report:
                        result.model_response = (
                            f"Explanation: {report.executive_summary}\n\n"
                            f"Key findings: {'; '.join(report.key_findings[:5])}\n\n"
                            f"Detailed: {report.detailed_analysis[:2000]}\n\n"
                            f"Exact Answer: {report.executive_summary[:200]}\n"
                            f"Confidence: 50%"
                        )
                    else:
                        result.model_response = f"Research completed but no report generated. Evidence count: {summary.total_evidence}"

                # Grade
                grade = await grade_answer(
                    question=q["question"],
                    correct_answer=q["answer"],
                    response=result.model_response,
                    grader_model=grader_model,
                )
                result.is_correct = grade["is_correct"]
                result.extracted_answer = grade["extracted_answer"]
                result.confidence = grade["confidence"]
                result.grader_reasoning = grade["grader_reasoning"]
                result.status = "correct" if result.is_correct else "incorrect"

                run.completed += 1
                if result.is_correct:
                    run.correct += 1

            except Exception as e:
                result.status = "failed"
                result.error = f"{type(e).__name__}: {str(e)[:300]}"
                run.failed += 1
                logger.error("browsecomp_failed", id=q["id"], error=str(e))

            run.results.append(result)
            results_file.write(json.dumps(result.to_dict(), default=str) + "\n")
            results_file.flush()

            done += 1
            status_icon = "+" if result.is_correct else "-" if result.status == "incorrect" else "X"
            logger.info(
                "browsecomp_done",
                id=q["id"],
                status=f"{status_icon} {result.status}",
                extracted=result.extracted_answer[:60],
                correct=result.correct_answer[:60],
                elapsed=f"{result.elapsed_seconds:.0f}s",
                progress=f"{done}/{total}",
            )

    # Run all questions
    tasks = [_run_one(q) for q in questions]
    await asyncio.gather(*tasks)

    results_file.close()
    run.completed_at = datetime.now(timezone.utc)

    # Write summary
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(run.to_dict(), indent=2, default=str))

    logger.info(
        "browsecomp_complete",
        run_id=run.run_id,
        accuracy=f"{run.accuracy:.1f}%",
        correct=run.correct,
        total=run.completed,
        failed=run.failed,
    )

    return run


# --- Analysis ---


def analyze_browsecomp_run(run_dir: str | Path) -> dict[str, Any]:
    """Analyze a BrowseComp run and produce a detailed scorecard."""
    run_dir = Path(run_dir)
    results_path = run_dir / "results.jsonl"

    results = []
    for line in results_path.read_text().strip().split("\n"):
        if line.strip():
            results.append(json.loads(line))

    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    incorrect = sum(1 for r in results if r.get("status") == "incorrect")
    failed = sum(1 for r in results if r.get("status") == "failed")
    completed = correct + incorrect

    # Timing stats
    elapsed = [r["elapsed_seconds"] for r in results if r["elapsed_seconds"] > 0]
    avg_elapsed = sum(elapsed) / max(len(elapsed), 1)

    # Evidence stats
    evidence = [r["total_evidence"] for r in results if r["total_evidence"] > 0]
    avg_evidence = sum(evidence) / max(len(evidence), 1)

    # Confidence distribution
    conf_buckets = {"0-25": 0, "26-50": 0, "51-75": 0, "76-100": 0}
    for r in results:
        c = r.get("confidence", 0)
        if c <= 25: conf_buckets["0-25"] += 1
        elif c <= 50: conf_buckets["26-50"] += 1
        elif c <= 75: conf_buckets["51-75"] += 1
        else: conf_buckets["76-100"] += 1

    # Topic breakdown
    by_topic: dict[str, dict] = {}
    for r in results:
        topic = r.get("topic", "unknown") or "unknown"
        if topic not in by_topic:
            by_topic[topic] = {"total": 0, "correct": 0}
        by_topic[topic]["total"] += 1
        if r.get("is_correct"):
            by_topic[topic]["correct"] += 1

    # Failure examples
    wrong_examples = [
        {
            "id": r["id"],
            "question": r["question"][:120],
            "correct_answer": r["correct_answer"],
            "extracted_answer": r.get("extracted_answer", "")[:100],
        }
        for r in results
        if r.get("status") == "incorrect"
    ][:20]

    correct_examples = [
        {
            "id": r["id"],
            "question": r["question"][:120],
            "correct_answer": r["correct_answer"],
            "extracted_answer": r.get("extracted_answer", "")[:100],
        }
        for r in results
        if r.get("is_correct")
    ][:10]

    return {
        "total": total,
        "completed": completed,
        "correct": correct,
        "incorrect": incorrect,
        "failed": failed,
        "accuracy": round(correct / max(completed, 1) * 100, 2),
        "avg_elapsed_seconds": round(avg_elapsed, 1),
        "avg_evidence": round(avg_evidence, 1),
        "confidence_distribution": conf_buckets,
        "by_topic": {k: {**v, "accuracy": round(v["correct"] / max(v["total"], 1) * 100, 1)} for k, v in sorted(by_topic.items())},
        "correct_examples": correct_examples,
        "wrong_examples": wrong_examples,
    }


def format_browsecomp_report(analysis: dict[str, Any]) -> str:
    """Format analysis as readable text."""
    lines = [
        "=" * 70,
        "BROWSECOMP BENCHMARK RESULTS",
        "=" * 70,
        f"Total: {analysis['total']} | Completed: {analysis['completed']} | "
        f"Failed: {analysis['failed']}",
        f"Correct: {analysis['correct']} | Incorrect: {analysis['incorrect']}",
        f"ACCURACY: {analysis['accuracy']:.1f}%",
        f"Avg time: {analysis['avg_elapsed_seconds']:.0f}s | Avg evidence: {analysis['avg_evidence']:.0f}",
        "",
        "--- Confidence Distribution ---",
    ]
    for bucket, count in analysis["confidence_distribution"].items():
        lines.append(f"  {bucket}%: {count}")

    if analysis.get("by_topic"):
        lines.extend(["", "--- By Topic ---"])
        for topic, stats in analysis["by_topic"].items():
            lines.append(f"  {topic}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.0f}%)")

    if analysis.get("correct_examples"):
        lines.extend(["", "--- Correct Examples ---"])
        for ex in analysis["correct_examples"][:5]:
            lines.append(f"  [{ex['id']}] Q: {ex['question'][:80]}...")
            lines.append(f"    A: {ex['correct_answer']} | Got: {ex['extracted_answer'][:60]}")

    if analysis.get("wrong_examples"):
        lines.extend(["", "--- Wrong Examples ---"])
        for ex in analysis["wrong_examples"][:10]:
            lines.append(f"  [{ex['id']}] Q: {ex['question'][:80]}...")
            lines.append(f"    Expected: {ex['correct_answer']} | Got: {ex['extracted_answer'][:60]}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)
