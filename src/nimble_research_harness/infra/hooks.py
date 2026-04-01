"""Hook system — pre/post tool execution hooks for budget, domain, and rate limiting."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Optional
from urllib.parse import urlparse

from .logging import get_logger

logger = get_logger(__name__)


class HookDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"


@dataclass
class HookContext:
    """Context passed to hook functions."""

    tool_name: str
    params: dict[str, Any]
    session_id: str
    elapsed_seconds: float = 0.0
    budget_remaining_seconds: float = float("inf")
    tool_calls_made: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HookResult:
    """Result from a hook execution."""

    decision: HookDecision
    reason: str = ""
    modified_params: dict[str, Any] | None = None


HookFn = Callable[[HookContext], Awaitable[HookResult]]


class HookRegistry:
    """Central registry for pre/post tool execution hooks."""

    def __init__(self) -> None:
        self._pre_hooks: list[tuple[str, HookFn]] = []
        self._post_hooks: list[tuple[str, Callable]] = []

    def add_pre_hook(self, name: str, hook: HookFn) -> None:
        self._pre_hooks.append((name, hook))
        logger.debug("pre_hook_registered", name=name)

    def add_post_hook(self, name: str, hook: Callable) -> None:
        self._post_hooks.append((name, hook))
        logger.debug("post_hook_registered", name=name)

    async def run_pre_hooks(self, ctx: HookContext) -> HookResult:
        """Run all pre-hooks. First BLOCK wins."""
        for name, hook in self._pre_hooks:
            try:
                result = await hook(ctx)
                if result.decision == HookDecision.BLOCK:
                    logger.info("hook_blocked", hook=name, tool=ctx.tool_name, reason=result.reason)
                    return result
                if result.modified_params is not None:
                    ctx.params = result.modified_params
            except Exception as e:
                logger.warning("hook_error", hook=name, error=str(e))
        return HookResult(decision=HookDecision.ALLOW)

    async def run_post_hooks(self, ctx: HookContext, result: dict[str, Any]) -> None:
        """Run all post-hooks (informational, cannot block)."""
        for name, hook in self._post_hooks:
            try:
                await hook(ctx, result)
            except Exception as e:
                logger.warning("post_hook_error", hook=name, error=str(e))


# --- Built-in Hooks ---


def budget_enforcement_hook(
    wall_clock_limit: float,
    start_time: float,
) -> HookFn:
    """Block tool calls when time budget is exhausted."""

    async def _hook(ctx: HookContext) -> HookResult:
        elapsed = time.time() - start_time
        if elapsed >= wall_clock_limit:
            return HookResult(
                decision=HookDecision.BLOCK,
                reason=f"Budget exhausted: {elapsed:.0f}s / {wall_clock_limit:.0f}s",
            )
        return HookResult(decision=HookDecision.ALLOW)

    return _hook


def domain_filter_hook(
    disallowed_domains: list[str],
    preferred_domains: list[str] | None = None,
) -> HookFn:
    """Block extract/crawl on disallowed domains; boost preferred domains."""

    _blocked = {d.lower().strip() for d in disallowed_domains}

    async def _hook(ctx: HookContext) -> HookResult:
        url = ctx.params.get("url", "")
        if not url:
            return HookResult(decision=HookDecision.ALLOW)

        try:
            domain = urlparse(url).netloc.lower()
        except Exception:
            return HookResult(decision=HookDecision.ALLOW)

        for blocked in _blocked:
            if domain == blocked or domain.endswith(f".{blocked}"):
                return HookResult(
                    decision=HookDecision.BLOCK,
                    reason=f"Domain '{domain}' is disallowed",
                )

        return HookResult(decision=HookDecision.ALLOW)

    return _hook


def content_size_hook(max_content_length: int = 10000) -> HookFn:
    """Truncate oversized content params before tool execution."""

    async def _hook(ctx: HookContext) -> HookResult:
        modified = False
        new_params = dict(ctx.params)
        for key in ("content", "text", "body"):
            if key in new_params and isinstance(new_params[key], str):
                if len(new_params[key]) > max_content_length:
                    new_params[key] = new_params[key][:max_content_length]
                    modified = True

        if modified:
            return HookResult(
                decision=HookDecision.ALLOW,
                modified_params=new_params,
            )
        return HookResult(decision=HookDecision.ALLOW)

    return _hook


def rate_limit_hook(max_concurrent: int = 10) -> HookFn:
    """Enforce per-tool concurrency limits via a semaphore."""

    _semaphore = asyncio.Semaphore(max_concurrent)

    async def _hook(ctx: HookContext) -> HookResult:
        if _semaphore.locked():
            return HookResult(
                decision=HookDecision.BLOCK,
                reason=f"Rate limit: {max_concurrent} concurrent tool calls",
            )
        return HookResult(decision=HookDecision.ALLOW)

    return _hook


def build_hooks(
    wall_clock_limit: float,
    start_time: float,
    disallowed_domains: list[str] | None = None,
    preferred_domains: list[str] | None = None,
    max_content_length: int = 10000,
    max_concurrent: int = 10,
) -> HookRegistry:
    """Build a HookRegistry with all standard hooks wired up."""
    registry = HookRegistry()

    registry.add_pre_hook(
        "budget_enforcement",
        budget_enforcement_hook(wall_clock_limit, start_time),
    )

    if disallowed_domains:
        registry.add_pre_hook(
            "domain_filter",
            domain_filter_hook(disallowed_domains, preferred_domains),
        )

    registry.add_pre_hook(
        "content_size",
        content_size_hook(max_content_length),
    )

    registry.add_pre_hook(
        "rate_limit",
        rate_limit_hook(max_concurrent),
    )

    return registry
