"""Tests for session models."""

import pytest
from pydantic import ValidationError

from nimble_research_harness.models.enums import TimeBudget
from nimble_research_harness.models.session import (
    SessionConfig,
    TimeBudgetPolicy,
    UserResearchRequest,
)


class TestTimeBudgetPolicy:
    def test_all_presets_valid(self):
        for budget in TimeBudget:
            policy = TimeBudgetPolicy.from_budget(budget)
            assert policy.budget == budget
            assert policy.planning_depth >= 1
            assert policy.stop_conditions.wall_clock_seconds == budget.seconds

    def test_30s_is_minimal(self):
        policy = TimeBudgetPolicy.from_budget(TimeBudget.QUICK_30S)
        assert policy.max_searches == 3
        assert policy.max_crawl_pages == 0
        assert policy.verification_strictness == 0
        assert policy.max_refinement_loops == 0

    def test_1h_is_maximal(self):
        policy = TimeBudgetPolicy.from_budget(TimeBudget.EXHAUSTIVE_1H)
        assert policy.max_searches == 60  # Rebalanced: depth > breadth
        assert policy.max_crawl_pages == 2000
        assert policy.verification_strictness == 3
        assert policy.concurrency_limit == 20

    def test_monotonic_increase(self):
        budgets = list(TimeBudget)
        policies = [TimeBudgetPolicy.from_budget(b) for b in budgets]
        for i in range(1, len(policies)):
            assert policies[i].max_searches >= policies[i - 1].max_searches
            assert policies[i].concurrency_limit >= policies[i - 1].concurrency_limit

    def test_serialization_roundtrip(self):
        policy = TimeBudgetPolicy.from_budget(TimeBudget.MEDIUM_5M)
        data = policy.model_dump(mode="json")
        restored = TimeBudgetPolicy(**data)
        assert restored == policy


class TestUserResearchRequest:
    def test_basic_creation(self):
        req = UserResearchRequest(user_query="What is Nimble?")
        assert req.user_query == "What is Nimble?"
        assert req.time_budget == TimeBudget.MEDIUM_5M

    def test_whitespace_stripped(self):
        req = UserResearchRequest(user_query="  hello world  ")
        assert req.user_query == "hello world"

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            UserResearchRequest(user_query="")

    def test_too_short_query_rejected(self):
        with pytest.raises(ValidationError):
            UserResearchRequest(user_query="ab")

    def test_serialization(self):
        req = UserResearchRequest(
            user_query="Test query",
            time_budget=TimeBudget.DEEP_30M,
        )
        data = req.model_dump(mode="json")
        assert data["time_budget"] == "30m"
        restored = UserResearchRequest(**data)
        assert restored.user_query == req.user_query
