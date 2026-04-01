"""Tests for intake agent."""

from nimble_research_harness.agents.intake import (
    classify_task_type,
    extract_target_domains,
    normalize_request,
)
from nimble_research_harness.models.enums import TaskType, TimeBudget
from nimble_research_harness.models.session import UserResearchRequest


class TestClassifyTaskType:
    def test_market_research(self):
        assert classify_task_type("market analysis of EV industry") == TaskType.MARKET_RESEARCH

    def test_competitive(self):
        assert classify_task_type("compare Slack vs Teams") == TaskType.COMPETITIVE_INTEL

    def test_company(self):
        assert classify_task_type("who is Nimble?") == TaskType.COMPANY_DEEP_DIVE

    def test_factual(self):
        assert classify_task_type("what is web scraping?") == TaskType.FACTUAL_LOOKUP

    def test_default(self):
        assert classify_task_type("something random") == TaskType.OPEN_EXPLORATION


class TestExtractDomains:
    def test_finds_urls(self):
        domains = extract_target_domains("check amazon.com and walmart.com")
        assert "amazon.com" in domains
        assert "walmart.com" in domains

    def test_no_domains(self):
        assert extract_target_domains("general research topic") == []


class TestNormalizeRequest:
    def test_basic_normalization(self):
        req = UserResearchRequest(
            user_query="What is Nimble",
            time_budget=TimeBudget.SHORT_2M,
        )
        config = normalize_request(req)
        assert config.normalized_objective.endswith(".")
        assert config.time_budget == TimeBudget.SHORT_2M
        assert config.policy.budget == TimeBudget.SHORT_2M
