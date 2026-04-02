"""Main orchestrator engine — enhanced 13-stage research pipeline.

Enhancements inspired by Parallel AI Task API and Claude Code architecture:
- Hook system for budget/domain/rate enforcement
- SSE event streaming for real-time progress
- Interactive gates at critical decision points
- Per-field provenance (FieldBasis) on report outputs
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from ..agents.analyst import analyze_and_report
from ..agents.intake import normalize_request
from ..agents.monitor import BudgetMonitor
from ..agents.planner import assess_evidence_sufficiency, create_followup_plan, create_plan
from ..agents.researcher import execute_research
from ..agents.skill_builder import build_skill
from ..agents.verifier import verify_claims
from ..infra.context import RunContext, get_context, set_context
from ..infra.errors import AgentAbortError, AgentTimeoutError
from ..infra.events import EventStream
from ..infra.hooks import HookContext, HookDecision, HookRegistry, build_hooks
from ..infra.logging import get_logger
from ..models.discovery import AgentFitScore
from ..models.enums import ExecutionMode, ExecutionStage, TimeBudget
from ..models.output import ResearchReport, SessionSummary
from ..models.session import SessionConfig, UserResearchRequest
from ..nimble.provider import NimbleProvider
from ..skillgen.deployer import deploy_skill
from ..storage.backend import StorageBackend
from ..tools.definitions import build_registry
from ..wsa.catalog import WSACatalog
from ..wsa.strategy import ExecutionStrategy
from .gates import GateDecision, GateRegistry, GateResult

logger = get_logger(__name__)


async def run_research(
    request: UserResearchRequest,
    provider: NimbleProvider,
    storage: StorageBackend,
    resume_session_id: Optional[str] = None,
    gate_registry: Optional[GateRegistry] = None,
    event_stream: Optional[EventStream] = None,
) -> SessionSummary:
    """Execute the full research pipeline with hooks, events, and gates."""

    gates = gate_registry or GateRegistry()

    # --- Stage 0: Discovery ---
    catalog = WSACatalog(provider)
    await catalog.load()
    logger.info("wsa_catalog_ready", agent_count=catalog.count)

    # --- Stage 1: Intake ---
    config = normalize_request(request)
    fast_mode = request.fast_mode or request.time_budget in (
        TimeBudget.QUICK_30S,
        TimeBudget.SHORT_2M,
        TimeBudget.MEDIUM_5M,
    )
    if fast_mode:
        logger.info("fast_mode_enabled", budget=request.time_budget.value)
    ctx = RunContext(session_id=config.session_id, storage=storage)
    set_context(ctx)

    monitor = BudgetMonitor(config)
    monitor.set_stage(ExecutionStage.INTAKE)

    if event_stream:
        await event_stream.session_started(
            query=config.user_query,
            budget=config.time_budget.label,
        )

    # Build hook registry with source policy from request
    disallowed = []
    preferred = []
    if request.source_policy:
        disallowed = request.source_policy.get("disallowed_domains", [])
        preferred = request.source_policy.get("preferred_domains", [])

    hooks = build_hooks(
        wall_clock_limit=config.policy.stop_conditions.wall_clock_seconds,
        start_time=monitor.start_time,
        disallowed_domains=disallowed or None,
        preferred_domains=preferred or None,
        max_content_length=config.policy.stop_conditions.wall_clock_seconds * 30,
        max_concurrent=config.policy.concurrency_limit,
    )

    # Handle resume
    if resume_session_id:
        existing = await storage.load_session(resume_session_id)
        if existing:
            config = existing
            ctx = RunContext(session_id=config.session_id, storage=storage)
            set_context(ctx)
            checkpoint = await storage.load_latest_checkpoint(resume_session_id)
            if checkpoint:
                logger.info("resuming_session", stage=checkpoint.stage.value)

    session_id_str = str(config.session_id)
    await storage.create_session(config)
    await monitor.create_checkpoint(ExecutionStage.INTAKE, 0)

    registry = build_registry(provider)
    wsa_matches: list[AgentFitScore] = []

    try:
        # --- Stage 0b: WSA Strategy ---
        monitor.set_stage(ExecutionStage.DISCOVERY)
        if event_stream:
            await event_stream.stage_entered("discovery", monitor.elapsed_seconds)

        if config.execution_mode != ExecutionMode.RAW_TOOLS:
            strategy = ExecutionStrategy(catalog)
            mode, wsa_matches = await strategy.resolve(
                target_domains=config.target_domains,
                target_verticals=[],
                target_entity_types=[],
                required_output_fields=[],
                available_input_params={},
            )
            if config.execution_mode == ExecutionMode.HYBRID:
                config = config.model_copy(update={"execution_mode": mode})
            logger.info(
                "execution_strategy",
                mode=config.execution_mode.value,
                wsa_matches=len(wsa_matches),
            )
        await monitor.create_checkpoint(ExecutionStage.DISCOVERY, 1)

        # --- Stage 2: Skill Generation ---
        monitor.set_stage(ExecutionStage.SKILL_GEN)
        if event_stream:
            await event_stream.stage_entered("skill_gen", monitor.elapsed_seconds)

        if monitor.is_over_budget:
            raise AgentTimeoutError("skill_gen", int(monitor.elapsed_seconds * 1000))

        skill_timeout = max(30, monitor.remaining_seconds * 0.15) if fast_mode else max(90, monitor.remaining_seconds * 0.4)
        skill = await asyncio.wait_for(
            build_skill(config, fast_mode=fast_mode),
            timeout=skill_timeout,
        )
        logger.info("skill_generated", title=skill.title, subquestions=len(skill.subquestions))
        await monitor.create_checkpoint(ExecutionStage.SKILL_GEN, 2)

        # --- Gate: Skill Approval ---
        gate_result = await gates.check(
            "skill_gen",
            f"Generated skill: {skill.title} with {len(skill.subquestions)} sub-questions",
            skill.model_dump(mode="json"),
        )
        if gate_result.decision == GateDecision.ABORT:
            raise AgentAbortError(f"Skill generation aborted: {gate_result.feedback}")

        # --- Stage 3: Deployment ---
        monitor.set_stage(ExecutionStage.DEPLOYMENT)
        if event_stream:
            await event_stream.stage_entered("deployment", monitor.elapsed_seconds)

        deployment = await deploy_skill(skill)
        logger.info("skill_deployed", deployment_id=str(deployment.deployment_id))
        await monitor.create_checkpoint(ExecutionStage.DEPLOYMENT, 3)

        # --- Stage 4: Planning ---
        monitor.set_stage(ExecutionStage.PLANNING)
        if event_stream:
            await event_stream.stage_entered("planning", monitor.elapsed_seconds)

        if monitor.is_over_budget:
            raise AgentTimeoutError("planning", int(monitor.elapsed_seconds * 1000))

        plan_timeout = max(30, monitor.remaining_seconds * 0.15) if fast_mode else max(90, monitor.remaining_seconds * 0.4)
        plan = await asyncio.wait_for(
            create_plan(config, skill, wsa_matches, fast_mode=fast_mode),
            timeout=plan_timeout,
        )
        await storage.save_plan(plan)
        logger.info("plan_created", steps=plan.total_steps)
        await monitor.create_checkpoint(ExecutionStage.PLANNING, 4, total_steps=plan.total_steps)

        # --- Gate: Plan Approval ---
        gate_result = await gates.check(
            "planning",
            f"Research plan with {plan.total_steps} steps",
            plan.model_dump(mode="json"),
        )
        if gate_result.decision == GateDecision.ABORT:
            raise AgentAbortError(f"Planning aborted: {gate_result.feedback}")

        # --- Stage 5: Research ---
        monitor.set_stage(ExecutionStage.RESEARCH)
        if event_stream:
            await event_stream.stage_entered("research", monitor.elapsed_seconds)

        if monitor.is_over_budget:
            raise AgentTimeoutError("research", int(monitor.elapsed_seconds * 1000))

        research_pct = 0.7 if fast_mode else 0.6
        research_result = await asyncio.wait_for(
            execute_research(config, plan, registry),
            timeout=max(30, monitor.remaining_seconds * research_pct),
        )
        evidence = await storage.get_evidence(session_id_str)

        if event_stream:
            for e in evidence[-5:]:
                await event_stream.finding_added(e.content[:200])

        logger.info(
            "research_complete",
            completed=research_result["completed"],
            errors=research_result["errors"],
            evidence_count=len(evidence),
        )
        await monitor.create_checkpoint(
            ExecutionStage.RESEARCH,
            5,
            completed_steps=research_result["completed"],
            total_steps=research_result["total_steps"],
            evidence_count=len(evidence),
        )

        # Budget warning at 70%
        if event_stream and monitor.budget_utilization_pct >= 70:
            await event_stream.budget_warning(100 - monitor.budget_utilization_pct)

        # --- Stage 6: Extraction (optional deep extract) ---
        # Skip extraction if search already yielded rich evidence
        skip_extraction = len(evidence) >= 80 and fast_mode
        if skip_extraction:
            logger.info("extraction_skipped", reason="sufficient_search_evidence", evidence_count=len(evidence))

        if not skip_extraction and not monitor.should_skip_stage(ExecutionStage.EXTRACTION):
            monitor.set_stage(ExecutionStage.EXTRACTION)
            if event_stream:
                await event_stream.stage_entered("extraction", monitor.elapsed_seconds)

            priority_urls = skill.extraction_strategy.priority_urls[:5]
            for url in priority_urls:
                if monitor.is_over_budget:
                    break

                # Run pre-hooks for domain filtering
                hook_ctx = HookContext(
                    tool_name="nimble_extract",
                    params={"url": url},
                    session_id=session_id_str,
                    elapsed_seconds=monitor.elapsed_seconds,
                    budget_remaining_seconds=monitor.remaining_seconds,
                )
                hook_result = await hooks.run_pre_hooks(hook_ctx)
                if hook_result.decision == HookDecision.BLOCK:
                    logger.info("extraction_blocked_by_hook", url=url, reason=hook_result.reason)
                    continue

                try:
                    await asyncio.wait_for(
                        registry.dispatch("nimble_extract", {"url": url}),
                        timeout=30,
                    )
                    if event_stream:
                        await event_stream.tool_completed("nimble_extract", f"Extracted {url}", 0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning("extraction_failed", url=url, error=str(e))
            await monitor.create_checkpoint(ExecutionStage.EXTRACTION, 6)

        # --- Stage 6b: Evidence Deepening (budget >= 10m only) ---
        # Reserve 25% of total budget for analysis (min 120s)
        analysis_reserve = max(120, config.policy.stop_conditions.wall_clock_seconds * 0.25)
        can_deepen = (
            config.time_budget in (TimeBudget.STANDARD_10M, TimeBudget.DEEP_30M, TimeBudget.EXHAUSTIVE_1H)
            and not monitor.is_over_budget
            and monitor.remaining_seconds > analysis_reserve
        )

        if can_deepen:
            iteration = 0
            prev_count = len(evidence)
            available = monitor.remaining_seconds - analysis_reserve
            logger.info("deepening_started", reserve=f"{analysis_reserve:.0f}s", available=f"{available:.0f}s")

            while monitor.remaining_seconds > analysis_reserve:
                if event_stream:
                    await event_stream.stage_entered(f"deepening_round_{iteration + 1}", monitor.elapsed_seconds)

                try:
                    assessment = await asyncio.wait_for(
                        assess_evidence_sufficiency(config, skill, evidence, fast_mode=fast_mode),
                        timeout=min(30, monitor.remaining_seconds * 0.1),
                    )
                except asyncio.TimeoutError:
                    logger.warning("sufficiency_check_timeout")
                    break

                if assessment["sufficient"]:
                    logger.info("evidence_sufficient", reason=assessment["reason"], evidence_count=len(evidence))
                    break

                logger.info("evidence_insufficient", reason=assessment["reason"], suggestions=len(assessment.get("suggested_queries", [])))

                try:
                    followup_plan = await asyncio.wait_for(
                        create_followup_plan(
                            config, skill, evidence,
                            suggested_queries=assessment.get("suggested_queries", []),
                            fast_mode=fast_mode,
                            iteration=iteration,
                        ),
                        timeout=min(30, monitor.remaining_seconds * 0.15),
                    )
                except asyncio.TimeoutError:
                    logger.warning("followup_plan_timeout")
                    break

                if not followup_plan or followup_plan.total_steps == 0:
                    logger.info("no_followup_steps")
                    break

                try:
                    result = await asyncio.wait_for(
                        execute_research(config, followup_plan, registry),
                        timeout=max(30, monitor.remaining_seconds * 0.4),
                    )
                except asyncio.TimeoutError:
                    logger.warning("deepening_research_timeout")
                    break

                evidence = await storage.get_evidence(session_id_str)
                new_count = len(evidence)
                growth = (new_count - prev_count) / max(prev_count, 1)
                logger.info(
                    "deepening_iteration",
                    iteration=iteration,
                    new_evidence=new_count - prev_count,
                    total=new_count,
                    growth=f"{growth:.1%}",
                )

                if growth < 0.05:
                    logger.info("deepening_converged", growth=f"{growth:.1%}")
                    break

                prev_count = new_count
                iteration += 1

        # --- Stage 7: Analysis ---
        monitor.set_stage(ExecutionStage.ANALYSIS)
        if event_stream:
            await event_stream.stage_entered("analysis", monitor.elapsed_seconds)

        if monitor.is_over_budget:
            raise AgentTimeoutError("analysis", int(monitor.elapsed_seconds * 1000))

        analysis_pct = 0.8 if fast_mode else 0.7
        await asyncio.wait_for(
            analyze_and_report(config, skill, registry, fast_mode=fast_mode),
            timeout=max(60, monitor.remaining_seconds * analysis_pct),
        )
        claims = await storage.get_claims(session_id_str)
        logger.info("analysis_complete", claims=len(claims))
        await monitor.create_checkpoint(
            ExecutionStage.ANALYSIS, 7, evidence_count=len(evidence)
        )

        # --- Gate: Findings Approval ---
        findings_summary = {
            "claims_count": len(claims),
            "evidence_count": len(evidence),
            "claims": [c.model_dump(mode="json") for c in claims[:10]],
        }
        gate_result = await gates.check(
            "analysis",
            f"Analysis produced {len(claims)} claims from {len(evidence)} evidence items",
            findings_summary,
        )
        if gate_result.decision == GateDecision.ABORT:
            raise AgentAbortError(f"Analysis aborted: {gate_result.feedback}")

        # --- Stage 8: Verification ---
        if not monitor.should_skip_stage(ExecutionStage.VERIFICATION):
            monitor.set_stage(ExecutionStage.VERIFICATION)
            if event_stream:
                await event_stream.stage_entered("verification", monitor.elapsed_seconds)
            try:
                await asyncio.wait_for(
                    verify_claims(config, registry, fast_mode=fast_mode),
                    timeout=max(10, monitor.remaining_seconds * 0.5),
                )
            except asyncio.TimeoutError:
                logger.warning("verification_timeout")
            await monitor.create_checkpoint(ExecutionStage.VERIFICATION, 8)

        # --- Stage 9: Reporting ---
        monitor.set_stage(ExecutionStage.REPORTING)
        if event_stream:
            await event_stream.stage_entered("reporting", monitor.elapsed_seconds)

        report = await storage.load_report(session_id_str)
        tool_calls = await storage.get_tool_calls(session_id_str)
        verifications = await storage.get_verifications(session_id_str)
        final_claims = await storage.get_claims(session_id_str)

        verified_count = sum(
            1 for v in verifications if v.status.value == "verified"
        )

        summary = SessionSummary(
            session_id=config.session_id,
            user_query=config.user_query,
            time_budget=config.time_budget,
            execution_mode=config.execution_mode,
            final_stage=ExecutionStage.COMPLETED,
            total_tool_calls=len(tool_calls),
            total_evidence=len(evidence),
            total_sources=len(set(e.source_url for e in evidence)),
            total_claims=len(final_claims),
            verified_claims=verified_count,
            elapsed_seconds=monitor.elapsed_seconds,
            report_confidence=report.confidence_rating if report else "low",
            skill_title=skill.title,
            wsa_agents_used=[s.agent_name for s in wsa_matches if s.is_strong_match],
        )
        await storage.save_summary(summary)
        await monitor.create_checkpoint(ExecutionStage.COMPLETED, 9)

        if event_stream:
            await event_stream.session_completed({
                "session_id": session_id_str,
                "elapsed": round(monitor.elapsed_seconds, 1),
                "evidence": len(evidence),
                "claims": len(final_claims),
                "verified": verified_count,
                "confidence": report.confidence_rating if report else "low",
            })

        logger.info(
            "research_complete",
            session_id=session_id_str,
            elapsed=f"{monitor.elapsed_seconds:.1f}s",
            evidence=len(evidence),
            claims=len(final_claims),
            verified=verified_count,
        )

        return summary

    except (AgentTimeoutError, AgentAbortError) as e:
        msg = e.phase_name if isinstance(e, AgentTimeoutError) else e.reason
        logger.warning("research_stopped", reason=msg)
        evidence = await storage.get_evidence(session_id_str)
        claims = await storage.get_claims(session_id_str)
        tool_calls = await storage.get_tool_calls(session_id_str)

        summary = SessionSummary(
            session_id=config.session_id,
            user_query=config.user_query,
            time_budget=config.time_budget,
            execution_mode=config.execution_mode,
            final_stage=ExecutionStage.FAILED,
            total_tool_calls=len(tool_calls),
            total_evidence=len(evidence),
            total_sources=len(set(e.source_url for e in evidence)),
            total_claims=len(claims),
            elapsed_seconds=monitor.elapsed_seconds,
        )
        await storage.save_summary(summary)

        if event_stream:
            await event_stream.session_failed(str(e))

        return summary

    except Exception as e:
        logger.error("research_failed", error=str(e))

        summary = SessionSummary(
            session_id=config.session_id,
            user_query=config.user_query,
            time_budget=config.time_budget,
            execution_mode=config.execution_mode,
            final_stage=ExecutionStage.FAILED,
            elapsed_seconds=monitor.elapsed_seconds,
        )
        await storage.save_summary(summary)

        if event_stream:
            await event_stream.session_failed(str(e))

        raise
