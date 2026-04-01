"""Tests for WSA discovery models."""

from nimble_research_harness.models.discovery import AgentFitScore, WSACandidate


class TestAgentFitScore:
    def test_perfect_score(self):
        score = AgentFitScore(
            agent_name="test",
            domain_match=1.0,
            entity_type_match=1.0,
            vertical_match=1.0,
            output_field_coverage=1.0,
            input_feasibility=1.0,
        )
        assert score.composite_score == 1.0
        assert score.is_strong_match is True

    def test_zero_score(self):
        score = AgentFitScore(agent_name="test")
        assert score.composite_score == 0.0
        assert score.is_strong_match is False

    def test_partial_score(self):
        score = AgentFitScore(
            agent_name="test",
            domain_match=1.0,
            entity_type_match=0.5,
        )
        assert 0.0 < score.composite_score < 1.0

    def test_strong_match_threshold(self):
        score = AgentFitScore(
            agent_name="test",
            domain_match=1.0,
            entity_type_match=1.0,
            vertical_match=0.5,
        )
        assert score.is_strong_match is True


class TestWSACandidate:
    def test_basic_creation(self):
        c = WSACandidate(name="amazon_search", domain="amazon.com")
        assert c.name == "amazon_search"
        assert c.is_public is True
