"""Tests for WSA scorer."""

from nimble_research_harness.models.discovery import WSACandidate
from nimble_research_harness.wsa.scorer import score_candidate


class TestScorer:
    def test_exact_domain_match(self):
        candidate = WSACandidate(
            name="amazon_search",
            domain="amazon.com",
            vertical="Ecommerce",
            entity_type="Search Engine Results Page (SERP)",
        )
        score = score_candidate(
            candidate,
            target_domains=["amazon.com"],
            target_verticals=["ecommerce"],
            target_entity_types=["serp"],
            required_output_fields=[],
            available_input_params={},
        )
        assert score.domain_match == 1.0
        assert score.vertical_match == 1.0
        assert score.composite_score > 0.5

    def test_no_match(self):
        candidate = WSACandidate(
            name="zillow_property",
            domain="zillow.com",
            vertical="Real Estate",
        )
        score = score_candidate(
            candidate,
            target_domains=["amazon.com"],
            target_verticals=["ecommerce"],
            target_entity_types=[],
            required_output_fields=[],
            available_input_params={},
        )
        assert score.domain_match == 0.0
        assert score.vertical_match == 0.0

    def test_partial_match(self):
        candidate = WSACandidate(
            name="walmart_search",
            domain="walmart.com",
            vertical="Ecommerce",
            entity_type="Search Engine Results Page (SERP)",
        )
        score = score_candidate(
            candidate,
            target_domains=["target.com"],
            target_verticals=["ecommerce"],
            target_entity_types=["serp"],
            required_output_fields=[],
            available_input_params={},
        )
        assert score.domain_match == 0.0
        assert score.vertical_match == 1.0
        assert score.entity_type_match == 1.0
