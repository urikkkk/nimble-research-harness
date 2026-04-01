"""Tests for evidence models."""

import uuid

from nimble_research_harness.models.enums import ClaimConfidence
from nimble_research_harness.models.evidence import Claim, EvidenceItem, VerificationResult


class TestEvidenceItem:
    def test_domain_derived_from_url(self):
        e = EvidenceItem(
            session_id=uuid.uuid4(),
            source_url="https://www.example.com/page",
            content="test",
        )
        assert e.source_domain == "www.example.com"

    def test_explicit_domain_preserved(self):
        e = EvidenceItem(
            session_id=uuid.uuid4(),
            source_url="https://example.com/page",
            source_domain="custom.domain",
            content="test",
        )
        assert e.source_domain == "custom.domain"

    def test_serialization(self):
        e = EvidenceItem(
            session_id=uuid.uuid4(),
            source_url="https://example.com",
            content="test content",
        )
        data = e.model_dump(mode="json")
        restored = EvidenceItem(**data)
        assert restored.evidence_id == e.evidence_id


class TestClaim:
    def test_default_unresolved(self):
        c = Claim(session_id=uuid.uuid4(), statement="Test claim")
        assert c.confidence == ClaimConfidence.UNRESOLVED

    def test_with_evidence_links(self):
        eid = uuid.uuid4()
        c = Claim(
            session_id=uuid.uuid4(),
            statement="Test",
            evidence_ids=[eid],
            confidence=ClaimConfidence.VERIFIED,
        )
        assert eid in c.evidence_ids
