"""Deep research engine V4 — entity-first multi-hop search.

Key insight from V1 benchmark analysis: the 3 correct answers all followed
the same pattern:
  1. Identify the ENTITY (person, movie, place) from constraints
  2. Search for the SPECIFIC ANSWER about that entity

V4 makes this systematic with a two-phase hop strategy:
  Phase A (hops 0-1): Entity Discovery — find WHO/WHAT the question is about
  Phase B (hops 2+): Answer Extraction — find the specific detail about the entity
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
    discover_entities,
    extract_answer_from_entity,
    extract_candidates,
    generate_entity_pivot_queries,
    generate_initial_queries,
    generate_refined_queries,
)
from .decomposer import decompose_question
from .searcher import search_hop
from .state import Candidate, DeepResearchSession, HopState
from .verifier import verify_candidate

logger = get_logger(__name__)


def _rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Rank candidates by (constraints_met count, confidence)."""
    return sorted(
        candidates,
        key=lambda c: (len(c.constraints_met), c.confidence),
        reverse=True,
    )


async def deep_research(
    question: str,
    provider: NimbleProvider,
    max_hops: int = 5,
    max_queries_per_hop: int = 8,
    max_parallel: int = 4,
    extract_top_n: int = 5,
    timeout_seconds: float = 540.0,
) -> DeepResearchSession:
    """Entity-first multi-hop search for specific factual answers.

    Phase A (hops 0-1): Entity Discovery
      - Broad constraint search + specialized database queries
      - Extract entity names (person, movie, place) from findings
      - Use site: prefixes for domain-specific sources

    Phase B (hops 2+): Answer Extraction
      - If entity found: "[entity] [answer detail]" queries + extract entity pages
      - If no entity: fall back to direct constraint search (V1 approach)

    Always returns best candidate (never "None" if any candidate exists).
    """
    start = time.time()
    session = DeepResearchSession(question=question)

    def _elapsed() -> float:
        return time.time() - start

    def _remaining() -> float:
        return max(0, timeout_seconds - _elapsed())

    # --- Step 1: Decompose + detect domain ---
    logger.info("deep_research_v4_start", question=question[:80])

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

    # Detect specialized sources based on domain
    specialized_sites = detect_domain(constraints, answer_type)
    logger.info(
        "decomposed",
        count=len(constraints),
        answer_type=answer_type,
        specialized_sites=specialized_sites[:3],
    )

    all_findings = []
    search_history: list[str] = []

    # --- Step 2: Multi-hop loop ---
    for hop in range(max_hops):
        if _remaining() < 30:
            logger.info("time_limit", hop=hop, remaining=f"{_remaining():.0f}s")
            break

        hop_start = time.time()
        hop_state = HopState(hop=hop)

        # --- Query Generation ---
        try:
            if hop == 0:
                # Phase A: Entity discovery — broad + specialized queries
                base_queries = await asyncio.wait_for(
                    generate_initial_queries(
                        question, constraints,
                        num_queries=max_queries_per_hop - len(specialized_sites),
                        answer_type=answer_type,
                    ),
                    timeout=min(20, _remaining()),
                )
                # Add specialized site: queries using top constraints
                top_constraint = constraints[0].text if constraints else question[:50]
                site_queries = [f"{site} {top_constraint[:40]}" for site in specialized_sites[:3]]
                queries = base_queries + site_queries

            elif hop == 1 and not session.identified_entity:
                # Phase A continued: try different constraint combos + Wikipedia lists
                gap = session.hops[-1].gap_analysis if session.hops else ""
                base_queries = await asyncio.wait_for(
                    generate_refined_queries(
                        question, constraints, session.candidates,
                        search_history, gap,
                        num_queries=max_queries_per_hop - 2,
                        answer_type=answer_type,
                    ),
                    timeout=min(20, _remaining()),
                )
                # Add Wikipedia list query
                list_query = f"list of {answer_type}s {constraints[0].text[:30]}" if constraints else f"list of {answer_type}s"
                queries = base_queries + [list_query, f"site:wikipedia.org list {constraints[0].text[:30]}" if constraints else ""]
                queries = [q for q in queries if q]

            elif session.identified_entity:
                # Phase B: Entity found — search for specific answer
                entity_queries = generate_entity_pivot_queries(
                    session.identified_entity, answer_type, constraints,
                )
                gap = session.hops[-1].gap_analysis if session.hops else ""
                extra_queries = await asyncio.wait_for(
                    generate_refined_queries(
                        question, constraints, session.candidates,
                        search_history, gap,
                        num_queries=max_queries_per_hop - len(entity_queries),
                        answer_type=answer_type,
                    ),
                    timeout=min(20, _remaining()),
                )
                queries = entity_queries + extra_queries

            else:
                # Fallback: standard refinement
                gap = session.hops[-1].gap_analysis if session.hops else ""
                queries = await asyncio.wait_for(
                    generate_refined_queries(
                        question, constraints, session.candidates,
                        search_history, gap,
                        num_queries=max_queries_per_hop,
                        answer_type=answer_type,
                    ),
                    timeout=min(20, _remaining()),
                )
        except asyncio.TimeoutError:
            logger.warning("query_gen_timeout", hop=hop)
            break

        session.total_llm_calls += 1
        hop_state.queries_used = queries
        search_history.extend(queries)
        logger.info("hop_queries", hop=hop, count=len(queries), phase="entity_discovery" if hop <= 1 and not session.identified_entity else "answer_extraction")

        # --- Execute searches ---
        try:
            findings, s_count, e_count = await asyncio.wait_for(
                search_hop(queries, provider, max_parallel=max_parallel, extract_top_n=extract_top_n),
                timeout=min(120, _remaining()),
            )
        except asyncio.TimeoutError:
            findings, s_count, e_count = [], 0, 0

        hop_state.findings = findings
        all_findings.extend(findings)
        session.total_searches += s_count
        session.total_extracts += e_count

        # --- Entity Discovery (hops 0-1) ---
        if hop <= 1 and not session.identified_entity and findings:
            try:
                entities = await asyncio.wait_for(
                    discover_entities(question, constraints, findings),
                    timeout=min(20, _remaining()),
                )
                session.total_llm_calls += 1

                if entities:
                    best_entity = entities[0]
                    session.identified_entity = best_entity["entity"]
                    session.identified_entity_type = best_entity.get("entity_type", "")
                    logger.info(
                        "entity_discovered",
                        entity=session.identified_entity,
                        type=session.identified_entity_type,
                        constraints_matched=best_entity.get("constraints_matched", []),
                    )

                    # Try to extract answer directly from entity
                    answer_candidate = await asyncio.wait_for(
                        extract_answer_from_entity(question, session.identified_entity, answer_type, all_findings),
                        timeout=min(15, _remaining()),
                    )
                    session.total_llm_calls += 1
                    if answer_candidate:
                        session.candidates.append(answer_candidate)
                        logger.info("answer_from_entity", answer=answer_candidate.answer, confidence=answer_candidate.confidence)

            except asyncio.TimeoutError:
                logger.warning("entity_discovery_timeout", hop=hop)

        # --- Standard Candidate Extraction ---
        try:
            new_candidates = await asyncio.wait_for(
                extract_candidates(
                    question, constraints, findings, session.candidates,
                    hop=hop, answer_type=answer_type,
                ),
                timeout=min(30, _remaining()),
            )
        except asyncio.TimeoutError:
            new_candidates = []

        session.total_llm_calls += 1
        session.candidates.extend(new_candidates)
        hop_state.candidates_found = new_candidates
        session.candidates = _rank_candidates(session.candidates)

        logger.info(
            "hop_result",
            hop=hop,
            new_candidates=len(new_candidates),
            total_candidates=len(session.candidates),
            entity=session.identified_entity or "(none)",
            top=session.candidates[0].answer[:50] if session.candidates else "(none)",
        )

        # --- Verify best candidate ---
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

            if all_met and confidence >= 0.5:
                session.final_answer = best.answer
                session.final_confidence = confidence
                hop_state.elapsed_seconds = time.time() - hop_start
                session.hops.append(hop_state)
                session.elapsed_seconds = _elapsed()
                logger.info("answer_verified", answer=best.answer, confidence=confidence, hop=hop)
                return session

        # --- Gap Analysis ---
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

        hop_state.elapsed_seconds = time.time() - hop_start
        session.hops.append(hop_state)

    # --- Return best guess ---
    session.elapsed_seconds = _elapsed()

    if session.candidates:
        session.candidates = _rank_candidates(session.candidates)
        best = session.candidates[0]
        session.final_answer = best.answer
        session.final_confidence = best.confidence
        logger.info(
            "best_guess",
            answer=best.answer,
            confidence=best.confidence,
            constraints_met=len(best.constraints_met),
            entity=session.identified_entity or "(none)",
            hops=len(session.hops),
        )
    else:
        session.final_answer = ""
        session.final_confidence = 0.0
        logger.warning("no_answer", hops=len(session.hops))

    return session
