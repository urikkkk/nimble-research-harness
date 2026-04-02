"""WSA discovery and scoring models."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


class WSACandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    vertical: Optional[str] = None
    entity_type: Optional[str] = None
    domain: Optional[str] = None
    managed_by: Optional[str] = None
    is_public: bool = True
    input_properties: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    feature_flags: dict[str, bool] = Field(default_factory=dict)


class AgentFitScore(BaseModel):
    agent_name: str
    agent_domain: Optional[str] = None
    agent_entity_type: Optional[str] = None
    agent_description: Optional[str] = None
    input_properties: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    domain_match: float = Field(default=0.0, ge=0.0, le=1.0)
    entity_type_match: float = Field(default=0.0, ge=0.0, le=1.0)
    vertical_match: float = Field(default=0.0, ge=0.0, le=1.0)
    output_field_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    input_feasibility: float = Field(default=0.0, ge=0.0, le=1.0)

    domain_weight: float = Field(default=0.35)
    entity_type_weight: float = Field(default=0.25)
    vertical_weight: float = Field(default=0.20)
    output_weight: float = Field(default=0.10)
    input_weight: float = Field(default=0.10)

    @computed_field
    @property
    def composite_score(self) -> float:
        return round(
            self.domain_match * self.domain_weight
            + self.entity_type_match * self.entity_type_weight
            + self.vertical_match * self.vertical_weight
            + self.output_field_coverage * self.output_weight
            + self.input_feasibility * self.input_weight,
            3,
        )

    @computed_field
    @property
    def is_strong_match(self) -> bool:
        return self.composite_score >= 0.6
