"""Tests for interactive gates."""

import pytest

from nimble_research_harness.orchestrator.gates import (
    GateDecision,
    GateRegistry,
    GateResult,
    auto_approve_gate,
)


class TestAutoApproveGate:
    @pytest.mark.asyncio
    async def test_always_approves(self):
        result = await auto_approve_gate("skill_gen", "test", {})
        assert result.decision == GateDecision.APPROVE


class TestGateRegistry:
    @pytest.mark.asyncio
    async def test_no_handler_auto_approves(self):
        gates = GateRegistry()
        result = await gates.check("skill_gen", "test", {})
        assert result.decision == GateDecision.APPROVE

    @pytest.mark.asyncio
    async def test_non_gate_stage_auto_approves(self):
        called = False

        async def handler(stage, desc, artifact):
            nonlocal called
            called = True
            return GateResult(decision=GateDecision.APPROVE)

        gates = GateRegistry(handler=handler)
        result = await gates.check("non_gate_stage", "test", {})
        assert result.decision == GateDecision.APPROVE
        assert not called

    @pytest.mark.asyncio
    async def test_custom_handler_called(self):
        async def handler(stage, desc, artifact):
            return GateResult(decision=GateDecision.ABORT, feedback="bad plan")

        gates = GateRegistry(handler=handler)
        result = await gates.check("planning", "test plan", {"steps": 5})
        assert result.decision == GateDecision.ABORT
        assert result.feedback == "bad plan"

    @pytest.mark.asyncio
    async def test_is_interactive(self):
        assert not GateRegistry().is_interactive
        assert GateRegistry(handler=auto_approve_gate).is_interactive
