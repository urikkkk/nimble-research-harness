"""Tests for CLI commands."""

from typer.testing import CliRunner

from nimble_research_harness.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Nimble Research Harness" in result.stdout


def test_session_list_empty():
    result = runner.invoke(app, ["session", "list"])
    assert result.exit_code == 0


def test_research_start_help():
    result = runner.invoke(app, ["research", "start", "--help"])
    assert result.exit_code == 0
    assert "query" in result.stdout.lower() or "QUERY" in result.stdout
