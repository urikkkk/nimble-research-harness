"""Interactive gates — user approval checkpoints at critical stages."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Awaitable, Optional

from pydantic import BaseModel

from ..infra.logging import get_logger

logger = get_logger(__name__)


class GateDecision(str, Enum):
    APPROVE = "approve"
    MODIFY = "modify"
    SKIP = "skip"
    ABORT = "abort"


class GateResult(BaseModel):
    """Result of a gate check."""

    decision: GateDecision
    feedback: str = ""
    modifications: dict[str, Any] | None = None


# Type for gate handler functions (CLI prompt, API callback, auto-approve, etc.)
GateHandler = Callable[[str, str, dict[str, Any]], Awaitable[GateResult]]


async def auto_approve_gate(stage: str, description: str, artifact: dict[str, Any]) -> GateResult:
    """Default gate handler that auto-approves everything."""
    logger.info("gate_auto_approved", stage=stage)
    return GateResult(decision=GateDecision.APPROVE)


class GateRegistry:
    """Manages interactive approval gates at critical pipeline stages."""

    GATE_STAGES = ("skill_gen", "planning", "analysis")

    def __init__(self, handler: GateHandler | None = None) -> None:
        self._handler = handler or auto_approve_gate
        self._enabled = handler is not None

    @property
    def is_interactive(self) -> bool:
        return self._enabled

    async def check(
        self,
        stage: str,
        description: str,
        artifact: dict[str, Any],
    ) -> GateResult:
        """Run a gate check at the given stage.

        Args:
            stage: Pipeline stage name (e.g., "skill_gen", "planning", "analysis")
            description: Human-readable description of what's being approved
            artifact: The artifact to review (skill spec, plan, findings)

        Returns:
            GateResult with the user's decision
        """
        if not self._enabled:
            return GateResult(decision=GateDecision.APPROVE)

        if stage not in self.GATE_STAGES:
            return GateResult(decision=GateDecision.APPROVE)

        logger.info("gate_check", stage=stage, description=description)
        result = await self._handler(stage, description, artifact)
        logger.info("gate_result", stage=stage, decision=result.decision.value)

        if result.decision == GateDecision.ABORT:
            logger.warning("gate_aborted", stage=stage, feedback=result.feedback)

        return result


def build_cli_gate_handler() -> GateHandler:
    """Build an interactive CLI gate handler using Rich."""

    async def _handler(
        stage: str, description: str, artifact: dict[str, Any]
    ) -> GateResult:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt
        import json

        console = Console()

        # Display the artifact
        stage_labels = {
            "skill_gen": "Generated Skill Spec",
            "planning": "Research Plan",
            "analysis": "Research Findings",
        }
        title = stage_labels.get(stage, stage)

        console.print()
        console.print(Panel(
            json.dumps(artifact, indent=2, default=str)[:3000],
            title=f"[bold]{title}[/bold]",
            subtitle=description,
            border_style="yellow",
        ))

        choice = Prompt.ask(
            "\n[bold]Decision[/bold]",
            choices=["approve", "skip", "abort"],
            default="approve",
        )

        if choice == "abort":
            feedback = Prompt.ask("Reason for aborting", default="")
            return GateResult(decision=GateDecision.ABORT, feedback=feedback)

        return GateResult(decision=GateDecision(choice))

    return _handler
