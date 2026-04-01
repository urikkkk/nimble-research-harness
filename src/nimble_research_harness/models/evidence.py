"""Evidence, claim, verification, and field-basis models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import ClaimConfidence


class Citation(BaseModel):
    """A citation linking a report field to its source."""

    model_config = ConfigDict(frozen=True)

    url: str
    title: Optional[str] = None
    excerpts: list[str] = Field(default_factory=list)
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FieldBasis(BaseModel):
    """Per-field provenance tracking — inspired by Parallel AI's research basis."""

    model_config = ConfigDict(frozen=True)

    field: str  # Field name or dot-path (e.g., "competitors.0")
    citations: list[Citation] = Field(default_factory=list)
    reasoning: str = ""  # How data was derived/reconciled
    confidence: ClaimConfidence = Field(default=ClaimConfidence.UNRESOLVED)


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    call_id: Optional[uuid.UUID] = None
    source_url: str
    source_domain: Optional[str] = None
    title: Optional[str] = None
    content: str
    content_type: str = Field(default="text")
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def derive_domain(cls, data):
        if isinstance(data, dict) and not data.get("source_domain"):
            url = data.get("source_url", "")
            if url:
                try:
                    data["source_domain"] = urlparse(url).netloc or None
                except Exception:
                    pass
        return data


class Claim(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    session_id: uuid.UUID
    statement: str
    evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    confidence: ClaimConfidence = Field(default=ClaimConfidence.UNRESOLVED)
    category: Optional[str] = None
    importance: int = Field(default=1, ge=1, le=5)


class VerificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    verification_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    claim_id: uuid.UUID
    session_id: uuid.UUID
    status: ClaimConfidence
    corroborating_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    conflicting_evidence_ids: list[uuid.UUID] = Field(default_factory=list)
    verification_call_ids: list[uuid.UUID] = Field(default_factory=list)
    notes: Optional[str] = None
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
