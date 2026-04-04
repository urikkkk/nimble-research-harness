"""Deep research engine V5 — V1 simplicity + specialized source routing.

Lesson from 4 benchmark runs: V1 (simplest) scored highest at 15.8%.
Every added complexity (type filtering, entity discovery, stricter thresholds)
REDUCED accuracy by adding LLM calls that consumed time without finding
better candidates.

V5 strategy: restore V1's fast/simple loop, add ONLY the proven win
(specialized site: queries from domain detection).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..infra.logging import get_logger
from ..nimble.provider import NimbleProvider
from .candidates import (
    analyze_gaps,
    detect_domain,
    extract_candidates,
    generate_initial_queries,
    generate_refined_queries,
)
from .decomposer import decompose_question
from .searcher import search_hop
from .state import Candidate, DeepResearchSession, HopState
from .verifier import verify_candidate

logger = get_logger(__name__)


async def deep_research(
    question: str,
    provider: NimbleProvider,
    max_hops: int = 5,
    max_queries_per_hop: int = 8,
    max_parallel: int = 4,
    extract_top_n: int = 5,
    timeout_seconds: float = 540.0,
) -> DeepResearchSession:
    """Fast multi-hop search with specialized source routing.

    Simple loop (proven best in benchmarks):
      1. Decompose into constraints
      2. For each hop: generate queries → parallel search → extract candidates → verify
      3. Inject specialized site: queries based on domain detection
      4. Always return best candidate (never "None")
    """
    start = time.time()
    session = DeepResearchSession(question=question)

    def _elapsed() -> float:
        return time.time() - start

    def _remaining() -> float:
        return max(0, timeout_seconds - _elapsed())

    # --- Step 1: Decompose ---
    logger.info("deep_research_start", question=question[:80])

    try:
        constraints, answer_type = await asyncio.wait_for(
            decompose_question(question),
            timeout=min(30, _remaining()),
        )
    except asyncio.TimeoutError:
        constraints, answer_type = [], "other"

    session.constraints = constraints
    session.expected_answer_type = answer_type
    session.total_llm_calls += 1

    # Detect specialized sources
    specialized_sites = detect_domain(constraints, answer_type)
    logger.info("decomposed", count=len(constraints), answer_type=answer_type, sites=specialized_sites[:3])

    all_findings = []
    search_history: list[str] = []

    # --- Step 2: Multi-hop loop (V1 simplicity) ---
    for hop in range(max_hops):
        if _remaining() < 30:
            break

        hop_start = time.time()
        hop_state = HopState(hop=hop)

        # Generate queries
        try:
            if hop == 0:
                queries = await asyncio.wait_for(
                    generate_initial_queries(question, constraints, num_queries=max_queries_per_hop, answer_type=answer_type),
                    timeout=min(20, _remaining()),
                )
                # Inject specialized site: queries
                if specialized_sites and constraints:
                    top = constraints[0].text[:40]
                    site_queries = [f"{s} {top}" for s in specialized_sites[:2]]
                    queries = queries[:max_queries_per_hop - 2] + site_queries
            else:
                gap = session.hops[-1].gap_analysis if session.hops else ""
                queries = await asyncio.wait_for(
                    generate_refined_queries(
                        question, constraints, session.candidates,
                        search_history, gap,
                        num_queries=max_queries_per_hop, answer_type=answer_type,
                    ),
                    timeout=min(20, _remaining()),
                )
        except asyncio.TimeoutError:
            break

        session.total_llm_calls += 1
        hop_state.queries_used = queries
        search_history.extend(queries)
        logger.info("hop_queries", hop=hop, queries=queries)

        # Parallel search + extract
        try:
            findings, s_count, e_count = await asyncio.wait_for(
                search_hop(queries, provider, max_parallel=max_parallel, extract_top_n=extract_top_n),
                timeout=min(90, _remaining()),
            )
        except asyncio.TimeoutError:
            findings, s_count, e_count = [], 0, 0

        hop_state.findings = findings
        all_findings.extend(findings)
        session.total_searches += s_count
        session.total_extracts += e_count

        # Extract candidates
        try:
            new_candidates = await asyncio.wait_for(
                extract_candidates(question, constraints, findings, session.candidates, hop=hop, answer_type=answer_type),
                timeout=min(30, _remaining()),
            )
        except asyncio.TimeoutError:
            new_candidates = []

        session.total_llm_calls += 1
        session.candidates.extend(new_candidates)
        hop_state.candidates_found = new_candidates

        # Sort by confidence
        session.candidates.sort(key=lambda c: c.confidence, reverse=True)

        logger.info(
            "hop_candidates",
            hop=hop,
            new=len(new_candidates),
            total=len(session.candidates),
            top=session.candidates[0].answer[:50] if session.candidates else "(none)",
        )

        # Verify best candidate
        if session.candidates:
            best = session.candidates[0]
            try:
                all_met, confidence, updated_constraints = await asyncio.wait_for(
                    verify_candidate(question, best, constraints, all_findings, answer_type=answer_type),
                    timeout=min(30, _remaining()),
                )
            except asyncio.TimeoutError:
                all_met, confidence, updated_constraints = False, 0.0, constraints

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
                logger.info("answer_found", answer=best.answer, confidence=confidence, hop=hop)
                return session

        # Gap analysis
        if _remaining() > 60:
            try:
                gap = await asyncio.wait_for(
                    analyze_gaps(question, constraints, search_history, session.candidates, answer_type=answer_type),
                    timeout=min(15, _remaining()),
                )
            except asyncio.TimeoutError:
                gap = ""
            session.total_llm_calls += 1
            hop_state.gap_analysis = gap
            logger.info("gap_analysis", hop=hop, gap=gap[:100])

        hop_state.elapsed_seconds = time.time() - hop_start
        session.hops.append(hop_state)

    # --- Return best guess (always) ---
    session.elapsed_seconds = _elapsed()

    if session.candidates:
        best = max(session.candidates, key=lambda c: c.confidence)
        session.final_answer = best.answer
        session.final_confidence = best.confidence
        logger.info("best_guess", answer=best.answer, confidence=best.confidence, hops=len(session.hops))
    else:
        session.final_answer = ""
        session.final_confidence = 0.0
        logger.warning("no_answer", hops=len(session.hops))

    return session
