"""Follow-up research — chain queries with prior session context."""

from __future__ import annotations

from typing import Optional

from ..infra.logging import get_logger
from ..models.enums import TimeBudget
from ..models.session import UserResearchRequest
from ..nimble.provider import NimbleProvider
from ..storage.backend import StorageBackend
from .engine import run_research

logger = get_logger(__name__)


async def follow_up_research(
    previous_session_id: str,
    new_query: str,
    time_budget: TimeBudget,
    provider: NimbleProvider,
    storage: StorageBackend,
) -> "SessionSummary":
    """Run a follow-up research session that builds on a prior session's findings.

    Loads the previous session's evidence and findings as context hints,
    so the new research can deepen or pivot without re-running the original.
    """
    from ..models.output import SessionSummary

    # Load prior session context
    prior_config = await storage.load_session(previous_session_id)
    if not prior_config:
        raise ValueError(f"Previous session {previous_session_id} not found")

    prior_evidence = await storage.get_evidence(previous_session_id)
    prior_claims = await storage.get_claims(previous_session_id)
    prior_report = await storage.load_report(previous_session_id)

    # Build context hints from prior session
    context_hints = [
        f"FOLLOW-UP: This builds on prior research about: {prior_config.user_query}",
    ]

    if prior_report:
        if prior_report.executive_summary:
            context_hints.append(f"Prior summary: {prior_report.executive_summary[:500]}")
        for finding in prior_report.key_findings[:5]:
            context_hints.append(f"Prior finding: {finding[:200]}")

    if prior_claims:
        for claim in prior_claims[:10]:
            context_hints.append(
                f"Prior claim [{claim.confidence.value}]: {claim.statement[:200]}"
            )

    # Key source domains from prior research
    prior_domains = list({e.source_domain for e in prior_evidence if e.source_domain})
    if prior_domains:
        context_hints.append(f"Prior sources included: {', '.join(prior_domains[:10])}")

    request = UserResearchRequest(
        user_query=new_query,
        time_budget=time_budget,
        context_hints=context_hints,
        previous_session_id=previous_session_id,
    )

    logger.info(
        "follow_up_started",
        prior_session=previous_session_id,
        new_query=new_query,
        context_hints_count=len(context_hints),
    )

    return await run_research(request, provider, storage)
