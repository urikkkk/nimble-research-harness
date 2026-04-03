"""Deep research module — multi-hop iterative search for BrowseComp-style problems."""

from .engine import deep_research
from .state import Candidate, Constraint, DeepResearchSession, HopState

__all__ = [
    "deep_research",
    "Candidate",
    "Constraint",
    "DeepResearchSession",
    "HopState",
]
