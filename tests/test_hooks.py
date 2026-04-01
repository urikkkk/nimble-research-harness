"""Tests for the hook system."""

import asyncio
import time

import pytest

from nimble_research_harness.infra.hooks import (
    HookContext,
    HookDecision,
    HookRegistry,
    HookResult,
    budget_enforcement_hook,
    build_hooks,
    content_size_hook,
    domain_filter_hook,
    rate_limit_hook,
)


@pytest.fixture
def hook_ctx():
    return HookContext(
        tool_name="nimble_extract",
        params={"url": "https://example.com"},
        session_id="test-session",
        elapsed_seconds=10.0,
        budget_remaining_seconds=50.0,
    )


class TestBudgetEnforcement:
    @pytest.mark.asyncio
    async def test_allows_within_budget(self, hook_ctx):
        hook = budget_enforcement_hook(wall_clock_limit=60.0, start_time=time.time())
        result = await hook(hook_ctx)
        assert result.decision == HookDecision.ALLOW

    @pytest.mark.asyncio
    async def test_blocks_over_budget(self, hook_ctx):
        hook = budget_enforcement_hook(wall_clock_limit=1.0, start_time=time.time() - 10)
        result = await hook(hook_ctx)
        assert result.decision == HookDecision.BLOCK
        assert "Budget exhausted" in result.reason


class TestDomainFilter:
    @pytest.mark.asyncio
    async def test_allows_clean_domain(self, hook_ctx):
        hook = domain_filter_hook(disallowed_domains=["evil.com"])
        result = await hook(hook_ctx)
        assert result.decision == HookDecision.ALLOW

    @pytest.mark.asyncio
    async def test_blocks_disallowed_domain(self, hook_ctx):
        hook = domain_filter_hook(disallowed_domains=["example.com"])
        result = await hook(hook_ctx)
        assert result.decision == HookDecision.BLOCK

    @pytest.mark.asyncio
    async def test_blocks_subdomain(self):
        hook = domain_filter_hook(disallowed_domains=["example.com"])
        ctx = HookContext(
            tool_name="nimble_extract",
            params={"url": "https://sub.example.com/page"},
            session_id="test",
        )
        result = await hook(ctx)
        assert result.decision == HookDecision.BLOCK

    @pytest.mark.asyncio
    async def test_allows_no_url(self, hook_ctx):
        hook = domain_filter_hook(disallowed_domains=["example.com"])
        hook_ctx.params = {"query": "test"}
        result = await hook(hook_ctx)
        assert result.decision == HookDecision.ALLOW


class TestContentSize:
    @pytest.mark.asyncio
    async def test_truncates_large_content(self):
        hook = content_size_hook(max_content_length=100)
        ctx = HookContext(
            tool_name="write_finding",
            params={"content": "x" * 500},
            session_id="test",
        )
        result = await hook(ctx)
        assert result.decision == HookDecision.ALLOW
        assert result.modified_params is not None
        assert len(result.modified_params["content"]) == 100

    @pytest.mark.asyncio
    async def test_passes_small_content(self):
        hook = content_size_hook(max_content_length=1000)
        ctx = HookContext(
            tool_name="write_finding",
            params={"content": "small"},
            session_id="test",
        )
        result = await hook(ctx)
        assert result.decision == HookDecision.ALLOW
        assert result.modified_params is None


class TestHookRegistry:
    @pytest.mark.asyncio
    async def test_first_block_wins(self, hook_ctx):
        registry = HookRegistry()

        async def always_block(ctx):
            return HookResult(decision=HookDecision.BLOCK, reason="blocked")

        async def always_allow(ctx):
            return HookResult(decision=HookDecision.ALLOW)

        registry.add_pre_hook("blocker", always_block)
        registry.add_pre_hook("allower", always_allow)

        result = await registry.run_pre_hooks(hook_ctx)
        assert result.decision == HookDecision.BLOCK

    @pytest.mark.asyncio
    async def test_all_allow(self, hook_ctx):
        registry = HookRegistry()

        async def allow(ctx):
            return HookResult(decision=HookDecision.ALLOW)

        registry.add_pre_hook("h1", allow)
        registry.add_pre_hook("h2", allow)

        result = await registry.run_pre_hooks(hook_ctx)
        assert result.decision == HookDecision.ALLOW


class TestBuildHooks:
    def test_builds_without_domains(self):
        hooks = build_hooks(wall_clock_limit=60, start_time=time.time())
        assert len(hooks._pre_hooks) == 3  # budget, content_size, rate_limit

    def test_builds_with_domains(self):
        hooks = build_hooks(
            wall_clock_limit=60,
            start_time=time.time(),
            disallowed_domains=["evil.com"],
        )
        assert len(hooks._pre_hooks) == 4  # budget, domain_filter, content_size, rate_limit
