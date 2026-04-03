"""State models for multi-hop deep research sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Constraint:
    """A single searchable constraint extracted from the question."""

    text: str  # "Brazilian referee"
    category: str = ""  # "person", "event", "property", "temporal", "location"
    is_met: bool = False
    evidence: str = ""  # What we found that satisfies this


@dataclass
class SearchFinding:
    """A single finding from a web search or extraction."""

    query: str
    url: str = ""
    title: str = ""
    snippet: str = ""
    full_content: str = ""  # From nimble_extract if used
    relevance: float = 0.5


@dataclass
class Candidate:
    """A potential answer to the question."""

    answer: str  # "Ireland v Romania"
    confidence: float = 0.0  # 0.0-1.0
    source_url: str = ""
    source_snippet: str = ""
    constraints_met: list[str] = field(default_factory=list)
    constraints_unmet: list[str] = field(default_factory=list)
    hop_found: int = 0
    verification_notes: str = ""


@dataclass
class HopState:
    """State of a single hop in the multi-hop search."""

    hop: int
    queries_used: list[str] = field(default_factory=list)
    findings: list[SearchFinding] = field(default_factory=list)
    candidates_found: list[Candidate] = field(default_factory=list)
    gap_analysis: str = ""
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop": self.hop,
            "queries": self.queries_used,
            "findings_count": len(self.findings),
            "candidates_count": len(self.candidates_found),
            "gap_analysis": self.gap_analysis[:300],
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


@dataclass
class DeepResearchSession:
    """Full session state for a multi-hop deep research run."""

    question: str
    constraints: list[Constraint] = field(default_factory=list)
    hops: list[HopState] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    final_answer: str = ""
    final_confidence: float = 0.0
    total_searches: int = 0
    total_extracts: int = 0
    total_llm_calls: int = 0
    elapsed_seconds: float = 0.0

    @property
    def best_candidate(self) -> Candidate | None:
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "constraints": [{"text": c.text, "category": c.category, "is_met": c.is_met} for c in self.constraints],
            "hops": [h.to_dict() for h in self.hops],
            "candidates": [
                {"answer": c.answer, "confidence": c.confidence, "constraints_met": c.constraints_met}
                for c in self.candidates
            ],
            "final_answer": self.final_answer,
            "final_confidence": self.final_confidence,
            "total_searches": self.total_searches,
            "total_extracts": self.total_extracts,
            "total_llm_calls": self.total_llm_calls,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }
