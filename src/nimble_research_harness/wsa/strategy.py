"""Execution mode strategy selection."""

from __future__ import annotations

from ..models.discovery import AgentFitScore, WSACandidate
from ..models.enums import ExecutionMode
from .catalog import WSACatalog
from .scorer import score_candidate


class ExecutionStrategy:
    """Determines execution mode: hybrid, raw_tools, or wsa_only."""

    def __init__(self, catalog: WSACatalog):
        self.catalog = catalog

    async def resolve(
        self,
        target_domains: list[str],
        target_verticals: list[str],
        target_entity_types: list[str],
        required_output_fields: list[str],
        available_input_params: dict[str, str],
        min_score: float = 0.4,
    ) -> tuple[ExecutionMode, list[AgentFitScore]]:
        """Score all relevant WSAs and choose execution mode."""
        candidates: list[WSACandidate] = []

        for domain in target_domains:
            candidates.extend(self.catalog.search_by_domain(domain))
        for vertical in target_verticals:
            candidates.extend(self.catalog.search_by_vertical(vertical))

        seen = set()
        unique: list[WSACandidate] = []
        for c in candidates:
            if c.name not in seen:
                seen.add(c.name)
                unique.append(c)

        scores = [
            score_candidate(
                c,
                target_domains,
                target_verticals,
                target_entity_types,
                required_output_fields,
                available_input_params,
            )
            for c in unique
        ]

        strong = [s for s in scores if s.composite_score >= min_score]
        strong.sort(key=lambda s: s.composite_score, reverse=True)

        if not strong:
            return ExecutionMode.RAW_TOOLS, []

        top_score = strong[0].composite_score
        if top_score >= 0.8 and all(s.composite_score >= 0.6 for s in strong[:3]):
            return ExecutionMode.WSA_ONLY, strong

        return ExecutionMode.HYBRID, strong
