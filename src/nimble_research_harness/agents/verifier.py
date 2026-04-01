"""Verifier agent — validates claims against evidence."""

from __future__ import annotations

import uuid
from typing import Any

from ..infra.context import get_context
from ..models.enums import ClaimConfidence
from ..models.evidence import VerificationResult
from ..models.session import SessionConfig
from ..tools.registry import ToolDefinition, ToolRegistry
from .base import run_agent_loop

SYSTEM_PROMPT = """You are a research claim verifier. Your job is to validate claims made during research.

For each claim that needs verification:
1. Read the evidence using `read_evidence`
2. If the claim has weak support, search for corroborating evidence using `nimble_search`
3. Record your verification result using `record_verification`

Verification standards:
- "verified": Claim is supported by 2+ independent sources with consistent information
- "partially_verified": Claim is supported by 1 strong source or 2+ weak sources
- "weak_support": Only indirect or tangential evidence
- "unresolved": Cannot confirm or deny with available evidence

Look for contradictions. If you find conflicting evidence, note it.
Be conservative — do not upgrade confidence without strong justification."""


async def verify_claims(
    config: SessionConfig,
    registry: ToolRegistry,
) -> list[VerificationResult]:
    """Verify claims against collected evidence."""
    results: list[VerificationResult] = []

    async def handle_record_verification(params: dict[str, Any]) -> dict[str, Any]:
        ctx = get_context()
        vr = VerificationResult(
            claim_id=uuid.UUID(params["claim_id"]),
            session_id=ctx.session_id,
            status=ClaimConfidence(params["status"]),
            corroborating_evidence_ids=[
                uuid.UUID(e) for e in params.get("corroborating_ids", [])
            ],
            conflicting_evidence_ids=[
                uuid.UUID(e) for e in params.get("conflicting_ids", [])
            ],
            notes=params.get("notes"),
        )
        await ctx.storage.insert_verification(vr)
        results.append(vr)
        return {"status": "ok", "verification_id": str(vr.verification_id)}

    registry.register(
        ToolDefinition(
            name="record_verification",
            description="Record the verification result for a claim.",
            input_schema={
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string", "description": "UUID of the claim to verify"},
                    "status": {
                        "type": "string",
                        "enum": ["verified", "partially_verified", "weak_support", "unresolved"],
                    },
                    "corroborating_ids": {"type": "array", "items": {"type": "string"}, "default": []},
                    "conflicting_ids": {"type": "array", "items": {"type": "string"}, "default": []},
                    "notes": {"type": "string"},
                },
                "required": ["claim_id", "status"],
            },
            handler=handle_record_verification,
        )
    )

    # Get existing claims
    ctx = get_context()
    claims = await ctx.storage.get_claims(str(ctx.session_id))
    if not claims:
        return results

    claims_text = "\n".join(
        f"- [{c.claim_id}] (importance: {c.importance}) {c.statement}"
        for c in claims
    )

    verification_budget = config.policy.verification_strictness
    tool_names = ["read_evidence", "nimble_search", "record_verification"]

    if verification_budget == 0:
        return results

    user_prompt = f"""Verify the following research claims. Verification strictness: {verification_budget}/3

Claims to verify:
{claims_text}

Steps:
1. Call `read_evidence` to see all supporting evidence
2. For claims with weak support, use `nimble_search` to find corroborating sources
3. For each claim, call `record_verification` with your assessment

Focus on the most important claims first (higher importance scores).
{"Verify all claims." if verification_budget >= 3 else "Verify key claims only." if verification_budget >= 2 else "Spot-check 1-2 claims."}"""

    await run_agent_loop(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        registry=registry,
        tool_names=tool_names,
        max_turns=min(10, len(claims) * 2 + 3),
    )

    return results
