"""Main orchestrator engine — 10-stage research pipeline."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from ..agents.analyst import analyze_and_report
from ..agents.intake import normalize_request
from ..agents.monitor import BudgetMonitor
from ..agents.planner import create_plan
from ..agents.researcher import execute_research
from ..agents.skill_builder import build_skill
from ..agents.verifier import verify_claims
from ..infra.context import RunContext, get_context, set_context
from ..infra.errors import AgentAbortError, AgentTimeoutError
from ..infra.logging import get_logger
from ..models.discovery import AgentFitScore
from ..models.enums import ExecutionMode, ExecutionStage
from ..models.output import ResearchReport, SessionSummary
from ..models.session import SessionConfig, UserResearchRequest
from ..nimble.provider import NimbleProvider
from ..skillgen.deployer import deploy_skill
from ..storage.backend import StorageBackend
from ..tools.definitions import build_registry
from ..wsa.catalog import WSACatalog
from ..wsa.strategy import ExecutionStrategy

logger = get_logger(__name__)


async def run_research(
    request: UserResearchRequest,
    provider: NimbleProvider,
    storage: StorageBackend,
    resume_session_id: Optional[str] = None,
) -> SessionSummary:
    """Execute the full 10-stage research pipeline."""

    # --- Stage 0: Discovery ---
    catalog = WSACatalog(provider)
    await catalog.load()
    logger.info("wsa_catalog_ready", agent_count=catalog.count)

    # --- Stage 1: Intake ---
    config = normalize_request(request)
    ctx = RunContext(session_id=config.session_id, storage=storage)
    set_context(ctx)

    monitor = BudgetMonitor(config)
    monitor.set_stage(ExecutionStage.INTAKE)

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
        if monitor.is_over_budget:
            raise AgentTimeoutError("skill_gen", int(monitor.elapsed_seconds * 1000))

        skill = await asyncio.wait_for(
            build_skill(config),
            timeout=min(30, monitor.remaining_seconds),
        )
        logger.info("skill_generated", title=skill.title, subquestions=len(skill.subquestions))
        await monitor.create_checkpoint(ExecutionStage.SKILL_GEN, 2)

        # --- Stage 3: Deployment ---
        monitor.set_stage(ExecutionStage.DEPLOYMENT)
        deployment = await deploy_skill(skill)
        logger.info("skill_deployed", deployment_id=str(deployment.deployment_id))
        await monitor.create_checkpoint(ExecutionStage.DEPLOYMENT, 3)

        # --- Stage 4: Planning ---
        monitor.set_stage(ExecutionStage.PLANNING)
        if monitor.is_over_budget:
            raise AgentTimeoutError("planning", int(monitor.elapsed_seconds * 1000))

        plan = await asyncio.wait_for(
            create_plan(config, skill, wsa_matches),
            timeout=min(30, monitor.remaining_seconds),
        )
        await storage.save_plan(plan)
        logger.info("plan_created", steps=plan.total_steps)
        await monitor.create_checkpoint(ExecutionStage.PLANNING, 4, total_steps=plan.total_steps)

        # --- Stage 5: Research ---
        monitor.set_stage(ExecutionStage.RESEARCH)
        if monitor.is_over_budget:
            raise AgentTimeoutError("research", int(monitor.elapsed_seconds * 1000))

        research_result = await asyncio.wait_for(
            execute_research(config, plan, registry),
            timeout=max(10, monitor.remaining_seconds * 0.6),
        )
        evidence = await storage.get_evidence(session_id_str)
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

        # --- Stage 6: Extraction (optional deep extract) ---
        if not monitor.should_skip_stage(ExecutionStage.EXTRACTION):
            monitor.set_stage(ExecutionStage.EXTRACTION)
            # Extract from priority URLs if we have budget
            priority_urls = skill.extraction_strategy.priority_urls[:5]
            for url in priority_urls:
                if monitor.is_over_budget:
                    break
                try:
                    await asyncio.wait_for(
                        registry.dispatch("nimble_extract", {"url": url}),
                        timeout=30,
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning("extraction_failed", url=url, error=str(e))
            await monitor.create_checkpoint(ExecutionStage.EXTRACTION, 6)

        # --- Stage 7: Analysis ---
        monitor.set_stage(ExecutionStage.ANALYSIS)
        if monitor.is_over_budget:
            raise AgentTimeoutError("analysis", int(monitor.elapsed_seconds * 1000))

        await asyncio.wait_for(
            analyze_and_report(config, skill, registry),
            timeout=max(15, monitor.remaining_seconds * 0.7),
        )
        claims = await storage.get_claims(session_id_str)
        logger.info("analysis_complete", claims=len(claims))
        await monitor.create_checkpoint(
            ExecutionStage.ANALYSIS, 7, evidence_count=len(evidence)
        )

        # --- Stage 8: Verification ---
        if not monitor.should_skip_stage(ExecutionStage.VERIFICATION):
            monitor.set_stage(ExecutionStage.VERIFICATION)
            try:
                await asyncio.wait_for(
                    verify_claims(config, registry),
                    timeout=max(10, monitor.remaining_seconds * 0.5),
                )
            except asyncio.TimeoutError:
                logger.warning("verification_timeout")
            await monitor.create_checkpoint(ExecutionStage.VERIFICATION, 8)

        # --- Stage 9: Reporting ---
        monitor.set_stage(ExecutionStage.REPORTING)
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

        logger.info(
            "research_complete",
            session_id=session_id_str,
            elapsed=f"{monitor.elapsed_seconds:.1f}s",
            evidence=len(evidence),
            claims=len(final_claims),
            verified=verified_count,
        )

        return summary

    except AgentTimeoutError as e:
        logger.warning("budget_exceeded", phase=e.phase_name)
        # Attempt graceful completion with whatever we have
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
        raise
