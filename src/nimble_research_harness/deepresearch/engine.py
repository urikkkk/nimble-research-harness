"""Deep research engine — multi-hop iterative search for factual answers.

This is a completely separate pipeline from the main research harness.
It reuses the Nimble provider but has its own search/verify/refine loop
optimized for BrowseComp-style needle-in-a-haystack problems.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..infra.logging import get_logger
from ..nimble.provider import NimbleProvider
from .candidates import analyze_gaps, extract_candidates, generate_initial_queries, generate_refined_queries
from .decomposer import decompose_question
from .searcher import search_hop
from .state import Candidate, DeepResearchSession, HopState
from .verifier import verify_candidate

logger = get_logger(__name__)


async def deep_research(
    question: str,
    provider: NimbleProvider,
    max_hops: int = 5,
    max_queries_per_hop: int = 6,
    max_parallel: int = 4,
    extract_top_n: int = 3,
    timeout_seconds: float = 540.0,
) -> DeepResearchSession:
    """Multi-hop iterative search for a specific factual answer.

    Algorithm:
    1. Decompose question into constraints
    2. For each hop:
       a. Generate search queries (initial or refined)
       b. Execute parallel searches + extract top results
       c. Extract candidate answers from findings
       d. Verify best candidate against ALL constraints
       e. If verified → return. If not → gap analysis → next hop.
    3. After max_hops, return best candidate.

    Args:
        question: The full question text
        provider: NimbleProvider (live or mock)
        max_hops: Maximum number of search-refine iterations
        max_queries_per_hop: Search queries per hop
        max_parallel: Concurrent search limit
        extract_top_n: How many URLs to deep-extract per hop
        timeout_seconds: Hard wall-clock limit
    """
    start = time.time()
    session = DeepResearchSession(question=question)

    def _elapsed() -> float:
        return time.time() - start

    def _remaining() -> float:
        return max(0, timeout_seconds - _elapsed())

    # --- Step 1: Decompose question into constraints ---
    logger.info("deep_research_start", question=question[:80])

    try:
        constraints = await asyncio.wait_for(
            decompose_question(question),
            timeout=min(30, _remaining()),
        )
    except asyncio.TimeoutError:
        logger.warning("decompose_timeout")
        constraints = []

    session.constraints = constraints
    session.total_llm_calls += 1
    logger.info("constraints_decomposed", count=len(constraints))

    all_findings = []
    search_history: list[str] = []

    # --- Step 2: Multi-hop search loop ---
    for hop in range(max_hops):
        if _remaining() < 30:
            logger.info("deep_research_time_limit", hop=hop, remaining=f"{_remaining():.0f}s")
            break

        hop_start = time.time()
        hop_state = HopState(hop=hop)

        # Generate queries
        if hop == 0:
            queries = await asyncio.wait_for(
                generate_initial_queries(question, constraints, num_queries=max_queries_per_hop),
                timeout=min(20, _remaining()),
            )
        else:
            gap = session.hops[-1].gap_analysis if session.hops else ""
            queries = await asyncio.wait_for(
                generate_refined_queries(
                    question, constraints, session.candidates,
                    search_history, gap, num_queries=max_queries_per_hop,
                ),
                timeout=min(20, _remaining()),
            )

        session.total_llm_calls += 1
        hop_state.queries_used = queries
        search_history.extend(queries)
        logger.info("hop_queries", hop=hop, queries=queries)

        # Execute parallel searches
        findings, s_count, e_count = await asyncio.wait_for(
            search_hop(queries, provider, max_parallel=max_parallel, extract_top_n=extract_top_n),
            timeout=min(90, _remaining()),
        )
        hop_state.findings = findings
        all_findings.extend(findings)
        session.total_searches += s_count
        session.total_extracts += e_count

        # Extract candidates
        new_candidates = await asyncio.wait_for(
            extract_candidates(question, constraints, findings, session.candidates, hop=hop),
            timeout=min(30, _remaining()),
        )
        session.total_llm_calls += 1
        session.candidates.extend(new_candidates)
        hop_state.candidates_found = new_candidates

        logger.info(
            "hop_candidates",
            hop=hop,
            new=len(new_candidates),
            total=len(session.candidates),
            top=session.candidates[0].answer[:50] if session.candidates else "(none)",
        )

        # Verify best candidate if we have any
        if session.candidates:
            # Sort by confidence descending
            session.candidates.sort(key=lambda c: c.confidence, reverse=True)
            best = session.candidates[0]

            all_met, confidence, updated_constraints = await asyncio.wait_for(
                verify_candidate(question, best, constraints, all_findings),
                timeout=min(30, _remaining()),
            )
            session.total_llm_calls += 1
            constraints = updated_constraints
            session.constraints = constraints

            best.confidence = confidence
            best.constraints_unmet = [c.text for c in constraints if not c.is_met]

            if all_met and confidence >= 0.6:
                session.final_answer = best.answer
                session.final_confidence = confidence
                hop_state.elapsed_seconds = time.time() - hop_start
                session.hops.append(hop_state)
                session.elapsed_seconds = _elapsed()
                logger.info(
                    "deep_research_answer_found",
                    answer=best.answer,
                    confidence=confidence,
                    hop=hop,
                    elapsed=f"{_elapsed():.0f}s",
                )
                return session

        # Gap analysis for next hop
        if _remaining() > 60:
            gap = await asyncio.wait_for(
                analyze_gaps(question, constraints, search_history, session.candidates),
                timeout=min(15, _remaining()),
            )
            session.total_llm_calls += 1
            hop_state.gap_analysis = gap
            logger.info("gap_analysis", hop=hop, gap=gap[:100])

        hop_state.elapsed_seconds = time.time() - hop_start
        session.hops.append(hop_state)

    # --- Step 3: Return best guess ---
    session.elapsed_seconds = _elapsed()

    if session.candidates:
        best = max(session.candidates, key=lambda c: c.confidence)
        session.final_answer = best.answer
        session.final_confidence = best.confidence
        logger.info(
            "deep_research_best_guess",
            answer=best.answer,
            confidence=best.confidence,
            hops=len(session.hops),
        )
    else:
        session.final_answer = ""
        session.final_confidence = 0.0
        logger.warning("deep_research_no_answer", hops=len(session.hops))

    return session
