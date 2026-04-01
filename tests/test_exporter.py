"""Tests for skill markdown/YAML export."""

import uuid

import pytest

from nimble_research_harness.models.skill import DynamicSkillSpec
from nimble_research_harness.skillgen.exporter import export_skill_markdown, export_skill_yaml


@pytest.fixture
def sample_skill():
    return DynamicSkillSpec(
        session_id=uuid.uuid4(),
        title="Market Research: Electric Vehicles",
        user_objective="Analyze the EV market landscape in 2026",
        task_type="market_research",
        time_budget="10m",
        subquestions=["What are the top EV manufacturers?", "What are the growth trends?"],
        target_entities=["Tesla", "BYD", "Rivian"],
        likely_source_types=["news", "financial_reports"],
        deliverables=["research_report", "comparison_table"],
    )


class TestMarkdownExport:
    def test_contains_frontmatter(self, sample_skill):
        md = export_skill_markdown(sample_skill)
        assert "---" in md
        assert "title: Market Research: Electric Vehicles" in md
        assert "task_type: market_research" in md

    def test_contains_objective(self, sample_skill):
        md = export_skill_markdown(sample_skill)
        assert "## Objective" in md
        assert "Analyze the EV market" in md

    def test_contains_subquestions(self, sample_skill):
        md = export_skill_markdown(sample_skill)
        assert "## Sub-questions" in md
        assert "1. What are the top EV manufacturers?" in md

    def test_contains_entities(self, sample_skill):
        md = export_skill_markdown(sample_skill)
        assert "## Target Entities" in md
        assert "- Tesla" in md

    def test_contains_policies(self, sample_skill):
        md = export_skill_markdown(sample_skill)
        assert "## Policies" in md
        assert "### Planning" in md
        assert "### Sources" in md
        assert "### Verification" in md

    def test_contains_tool_permissions(self, sample_skill):
        md = export_skill_markdown(sample_skill)
        assert "## Tool Permissions" in md
        assert "`nimble_search`" in md

    def test_contains_deliverables(self, sample_skill):
        md = export_skill_markdown(sample_skill)
        assert "## Deliverables" in md
        assert "- research_report" in md


class TestYamlExport:
    def test_produces_valid_yaml(self, sample_skill):
        import yaml
        yml = export_skill_yaml(sample_skill)
        data = yaml.safe_load(yml)
        assert data["title"] == "Market Research: Electric Vehicles"
        assert data["task_type"] == "market_research"

    def test_roundtrip(self, sample_skill):
        import yaml
        yml = export_skill_yaml(sample_skill)
        data = yaml.safe_load(yml)
        assert len(data["subquestions"]) == 2
        assert "Tesla" in data["target_entities"]
