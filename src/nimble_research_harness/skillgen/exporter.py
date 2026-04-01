"""Skill markdown export — render DynamicSkillSpec as human-readable markdown."""

from __future__ import annotations

from ..models.skill import DynamicSkillSpec


def export_skill_markdown(skill: DynamicSkillSpec) -> str:
    """Render a DynamicSkillSpec as a human-readable markdown document."""
    lines = [
        "---",
        f"title: {skill.title}",
        f"task_type: {skill.task_type.value}",
        f"time_budget: {skill.time_budget.value}",
        f"execution_mode: {skill.execution_mode.value}",
        f"skill_id: {skill.skill_id}",
        f"session_id: {skill.session_id}",
        "---",
        "",
        f"# {skill.title}",
        "",
        "## Objective",
        skill.user_objective,
        "",
    ]

    if skill.subquestions:
        lines.append("## Sub-questions")
        for i, q in enumerate(skill.subquestions, 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    if skill.target_entities:
        lines.append("## Target Entities")
        for entity in skill.target_entities:
            lines.append(f"- {entity}")
        lines.append("")

    if skill.likely_source_types:
        lines.append("## Likely Source Types")
        for src in skill.likely_source_types:
            lines.append(f"- {src}")
        lines.append("")

    # Policies
    lines.extend([
        "## Policies",
        "",
        "### Planning",
        f"- Depth: {skill.planning_policy.depth}",
        f"- Max subquestions: {skill.planning_policy.max_subquestions}",
        f"- Entity extraction: {'yes' if skill.planning_policy.require_entity_extraction else 'no'}",
        f"- Replanning: {'allowed' if skill.planning_policy.allow_replanning else 'disabled'}",
        "",
        "### Sources",
        f"- Min sources: {skill.source_policy.min_sources}",
        f"- Require diversity: {'yes' if skill.source_policy.require_diversity else 'no'}",
    ])

    if skill.source_policy.domain_include:
        lines.append(f"- Include domains: {', '.join(skill.source_policy.domain_include)}")
    if skill.source_policy.domain_exclude:
        lines.append(f"- Exclude domains: {', '.join(skill.source_policy.domain_exclude)}")
    if skill.source_policy.freshness_days:
        lines.append(f"- Freshness: {skill.source_policy.freshness_days} days")
    if skill.source_policy.preferred_domains:
        lines.append(f"- Preferred domains: {', '.join(skill.source_policy.preferred_domains)}")
    if skill.source_policy.disallowed_domains:
        lines.append(f"- Disallowed domains: {', '.join(skill.source_policy.disallowed_domains)}")

    lines.extend([
        "",
        "### Extraction",
        f"- Mode: {skill.extraction_policy.extract_mode}",
        f"- Max content length: {skill.extraction_policy.max_content_length}",
        f"- Render JS: {'yes' if skill.extraction_policy.render_js else 'no'}",
        "",
        "### Synthesis",
        f"- Style: {skill.synthesis_policy.style}",
        f"- Max findings: {skill.synthesis_policy.max_findings}",
        f"- Comparisons: {'required' if skill.synthesis_policy.require_comparisons else 'optional'}",
        "",
        "### Verification",
        f"- Strictness: {skill.verification_policy.strictness}/3",
        f"- Corroboration: {'required' if skill.verification_policy.require_corroboration else 'optional'}",
        f"- Flag contradictions: {'yes' if skill.verification_policy.flag_contradictions else 'no'}",
        "",
        "### Report",
        f"- Format: {skill.report_policy.format.value}",
        f"- Sections: {', '.join(skill.report_policy.sections)}",
        f"- Evidence table: {'yes' if skill.report_policy.include_evidence_table else 'no'}",
        "",
    ])

    # Tool permissions
    lines.append("## Tool Permissions")
    for tool in skill.tool_permissions:
        lines.append(f"- `{tool.value}`")
    lines.append("")

    # Search strategy
    if skill.search_strategy.queries:
        lines.append("## Search Strategy")
        lines.append(f"- Focus modes: {', '.join(f.value for f in skill.search_strategy.focus_modes)}")
        lines.append(f"- Max results per query: {skill.search_strategy.max_results_per_query}")
        lines.append("- Queries:")
        for q in skill.search_strategy.queries:
            lines.append(f"  - {q}")
        lines.append("")

    # Deliverables
    if skill.deliverables:
        lines.append("## Deliverables")
        for d in skill.deliverables:
            lines.append(f"- {d}")
        lines.append("")

    return "\n".join(lines)


def export_skill_yaml(skill: DynamicSkillSpec) -> str:
    """Export skill spec as YAML."""
    import yaml
    data = skill.model_dump(mode="json")
    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
