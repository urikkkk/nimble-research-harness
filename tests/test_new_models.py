"""Tests for new/enhanced model fields."""

import uuid

import pytest

from nimble_research_harness.models.evidence import Citation, FieldBasis
from nimble_research_harness.models.enums import ClaimConfidence, TimeBudget
from nimble_research_harness.models.output import ResearchReport
from nimble_research_harness.models.session import UserResearchRequest
from nimble_research_harness.models.skill import DynamicSkillSpec, SourcePolicy, VerificationPolicy


class TestCitation:
    def test_basic_creation(self):
        c = Citation(url="https://example.com", title="Example", excerpts=["text"])
        assert c.url == "https://example.com"
        assert len(c.excerpts) == 1

    def test_roundtrip(self):
        c = Citation(url="https://x.com")
        data = c.model_dump(mode="json")
        c2 = Citation(**data)
        assert c2.url == c.url


class TestFieldBasis:
    def test_basic_creation(self):
        fb = FieldBasis(
            field="competitors.0",
            citations=[Citation(url="https://src.com", excerpts=["found it"])],
            reasoning="Cross-referenced two sources",
            confidence=ClaimConfidence.VERIFIED,
        )
        assert fb.field == "competitors.0"
        assert fb.confidence == ClaimConfidence.VERIFIED

    def test_default_confidence(self):
        fb = FieldBasis(field="test")
        assert fb.confidence == ClaimConfidence.UNRESOLVED

    def test_roundtrip(self):
        fb = FieldBasis(
            field="price",
            citations=[Citation(url="https://a.com"), Citation(url="https://b.com")],
            reasoning="Average of two sources",
            confidence=ClaimConfidence.PARTIALLY_VERIFIED,
        )
        data = fb.model_dump(mode="json")
        fb2 = FieldBasis(**data)
        assert fb2.field == "price"
        assert len(fb2.citations) == 2


class TestEnhancedUserResearchRequest:
    def test_new_fields_default(self):
        r = UserResearchRequest(user_query="test query")
        assert r.output_schema is None
        assert r.source_policy is None
        assert r.metadata == {}
        assert r.previous_session_id is None
        assert r.fast_mode is False

    def test_new_fields_set(self):
        r = UserResearchRequest(
            user_query="test query",
            output_schema={"type": "object", "properties": {"name": {"type": "string"}}},
            source_policy={"preferred_domains": ["example.com"]},
            metadata={"user_id": "123", "project": "test"},
            previous_session_id="abc-123",
            fast_mode=True,
        )
        assert r.output_schema is not None
        assert r.fast_mode is True
        assert r.metadata["user_id"] == "123"
        assert r.previous_session_id == "abc-123"


class TestEnhancedSourcePolicy:
    def test_new_domain_fields(self):
        sp = SourcePolicy(
            preferred_domains=["reuters.com", "bloomberg.com"],
            disallowed_domains=["reddit.com"],
        )
        assert len(sp.preferred_domains) == 2
        assert sp.disallowed_domains == ["reddit.com"]

    def test_defaults_empty(self):
        sp = SourcePolicy()
        assert sp.preferred_domains == []
        assert sp.disallowed_domains == []


class TestEnhancedVerificationPolicy:
    def test_confidence_threshold(self):
        vp = VerificationPolicy(confidence_threshold=0.7)
        assert vp.confidence_threshold == 0.7

    def test_default_threshold(self):
        vp = VerificationPolicy()
        assert vp.confidence_threshold == 0.0


class TestEnhancedResearchReport:
    def test_field_basis(self):
        r = ResearchReport(
            session_id=uuid.uuid4(),
            title="Test",
            field_basis=[
                FieldBasis(field="revenue", reasoning="From 10-K", confidence=ClaimConfidence.VERIFIED),
            ],
        )
        assert len(r.field_basis) == 1

    def test_structured_output(self):
        r = ResearchReport(
            session_id=uuid.uuid4(),
            title="Test",
            structured_output={"name": "Acme Corp", "revenue": "$1B"},
        )
        assert r.structured_output["name"] == "Acme Corp"

    def test_defaults_empty(self):
        r = ResearchReport(session_id=uuid.uuid4(), title="Test")
        assert r.field_basis == []
        assert r.structured_output is None


class TestEnhancedDynamicSkillSpec:
    def test_output_schema(self):
        s = DynamicSkillSpec(
            session_id=uuid.uuid4(),
            title="Test Skill",
            user_objective="test",
            task_type="factual_lookup",
            time_budget="5m",
            output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        )
        assert s.output_schema is not None

    def test_default_no_schema(self):
        s = DynamicSkillSpec(
            session_id=uuid.uuid4(),
            title="Test",
            user_objective="test",
            task_type="factual_lookup",
            time_budget="5m",
        )
        assert s.output_schema is None
