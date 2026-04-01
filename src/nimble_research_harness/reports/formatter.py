"""Report formatters for different output modes."""

from __future__ import annotations

import json
from typing import Any

from ..models.enums import ReportFormat
from ..models.output import ResearchReport, SessionSummary


def format_report(report: ResearchReport, fmt: ReportFormat = ReportFormat.FULL_REPORT) -> str:
    if fmt == ReportFormat.JSON:
        return json.dumps(report.model_dump(mode="json"), indent=2, default=str)

    if fmt == ReportFormat.BRIEF:
        return _format_brief(report)

    if fmt == ReportFormat.EVIDENCE_TABLE:
        return _format_evidence_table(report)

    if fmt == ReportFormat.SOURCE_PACK:
        return _format_source_pack(report)

    return _format_full(report)


def _format_full(report: ResearchReport) -> str:
    lines = [
        f"# {report.title}",
        f"*Confidence: {report.confidence_rating} | Sources: {report.total_sources}*\n",
        "## Executive Summary",
        report.executive_summary or "(no summary)",
        "",
        "## Key Findings",
    ]
    for i, f in enumerate(report.key_findings, 1):
        lines.append(f"  {i}. {f}")

    if report.detailed_analysis:
        lines.extend(["", "## Detailed Analysis", report.detailed_analysis])

    if report.claims:
        lines.extend(["", "## Claims & Evidence"])
        for c in report.claims:
            conf = c.confidence.value.replace("_", " ").title()
            lines.append(f"  - [{conf}] {c.statement}")

    if report.known_unknowns:
        lines.extend(["", "## Known Unknowns"])
        for u in report.known_unknowns:
            lines.append(f"  - {u}")

    if report.limitations:
        lines.extend(["", "## Limitations"])
        for l in report.limitations:
            lines.append(f"  - {l}")

    if report.methodology:
        lines.extend(["", "## Methodology", report.methodology])

    return "\n".join(lines)


def _format_brief(report: ResearchReport) -> str:
    lines = [f"**{report.title}** (confidence: {report.confidence_rating})\n"]
    if report.executive_summary:
        lines.append(report.executive_summary)
    if report.key_findings:
        lines.append("")
        for f in report.key_findings[:5]:
            lines.append(f"- {f}")
    return "\n".join(lines)


def _format_evidence_table(report: ResearchReport) -> str:
    lines = [
        f"# Evidence Table: {report.title}",
        "",
        "| # | Source | Content | Relevance |",
        "|---|--------|---------|-----------|",
    ]
    for i, e in enumerate(report.evidence, 1):
        domain = e.source_domain or "unknown"
        content = e.content[:80].replace("|", " ").replace("\n", " ")
        lines.append(f"| {i} | {domain} | {content} | {e.relevance_score:.1f} |")
    return "\n".join(lines)


def _format_source_pack(report: ResearchReport) -> str:
    lines = [f"# Source Pack: {report.title}", ""]
    seen = set()
    for e in report.evidence:
        if e.source_url not in seen:
            seen.add(e.source_url)
            lines.append(f"- [{e.title or e.source_domain or 'Source'}]({e.source_url})")
    return "\n".join(lines)


def format_summary(summary: SessionSummary) -> str:
    return "\n".join([
        f"Session: {summary.session_id}",
        f"Query: {summary.user_query}",
        f"Budget: {summary.time_budget.label} (used {summary.budget_utilization_pct:.0f}%)",
        f"Mode: {summary.execution_mode.value}",
        f"Stage: {summary.final_stage.value}",
        f"Tool calls: {summary.total_tool_calls}",
        f"Evidence: {summary.total_evidence} items from {summary.total_sources} sources",
        f"Claims: {summary.total_claims} ({summary.verified_claims} verified)",
        f"Elapsed: {summary.elapsed_seconds:.1f}s",
        f"Confidence: {summary.report_confidence or 'n/a'}",
        f"Skill: {summary.skill_title or 'n/a'}",
        f"WSAs used: {', '.join(summary.wsa_agents_used) or 'none'}",
    ])
