"""Skill deployment — registers a generated skill into the runtime."""

from __future__ import annotations

from datetime import datetime, timezone

from ..infra.context import get_context
from ..infra.logging import get_logger
from ..models.enums import DeploymentStatus
from ..models.skill import DeploymentRecord, DynamicSkillSpec

logger = get_logger(__name__)


async def deploy_skill(skill: DynamicSkillSpec) -> DeploymentRecord:
    """Register a generated skill into the runtime and persist the deployment record."""
    ctx = get_context()

    record = DeploymentRecord(
        skill_id=skill.skill_id,
        session_id=skill.session_id,
        status=DeploymentStatus.DEPLOYING,
        registered_tools=[t.value for t in skill.tool_permissions],
    )

    logger.info(
        "deploying_skill",
        skill_id=str(skill.skill_id),
        title=skill.title,
        tools=record.registered_tools,
    )

    await ctx.storage.save_skill(skill)
    record.status = DeploymentStatus.READY
    record.deployed_at = datetime.now(timezone.utc)
    await ctx.storage.save_deployment(record)

    logger.info("skill_deployed", skill_id=str(skill.skill_id))
    return record
