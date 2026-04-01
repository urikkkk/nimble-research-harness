"""CLI interface for the research harness."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .infra.logging import setup_logging
from .models.enums import ReportFormat, TimeBudget
from .models.session import UserResearchRequest
from .nimble.client import NimbleClient
from .nimble.mock import MockNimbleProvider
from .orchestrator.engine import run_research
from .reports.formatter import format_report, format_summary
from .storage.json_backend import JsonStorageBackend

load_dotenv()
setup_logging()

app = typer.Typer(name="nrh", help="Nimble Research Harness — Dynamic web research agent platform")
console = Console()

research_app = typer.Typer(help="Research session commands")
skill_app = typer.Typer(help="Skill inspection commands")
session_app = typer.Typer(help="Session management commands")

app.add_typer(research_app, name="research")
app.add_typer(skill_app, name="skill")
app.add_typer(session_app, name="session")


BUDGET_OPTIONS = {
    "1": TimeBudget.QUICK_30S,
    "2": TimeBudget.SHORT_2M,
    "3": TimeBudget.MEDIUM_5M,
    "4": TimeBudget.STANDARD_10M,
    "5": TimeBudget.DEEP_30M,
    "6": TimeBudget.EXHAUSTIVE_1H,
}


def _ask_time_budget() -> TimeBudget:
    console.print("\n[bold]How long do you want the search to run?[/bold]\n")
    console.print("  [1] 30 seconds")
    console.print("  [2] 2 minutes")
    console.print("  [3] 5 minutes")
    console.print("  [4] 10 minutes")
    console.print("  [5] 30 minutes")
    console.print("  [6] 1 hour")
    console.print()

    choice = Prompt.ask("Select duration", choices=["1", "2", "3", "4", "5", "6"], default="3")
    budget = BUDGET_OPTIONS[choice]
    console.print(f"\n[green]Selected: {budget.label}[/green]\n")
    return budget


def _get_provider():
    api_key = os.environ.get("NIMBLE_API_KEY", "")
    if not api_key or api_key == "your_nimble_api_key_here":
        console.print("[yellow]No NIMBLE_API_KEY found — using mock provider[/yellow]")
        return MockNimbleProvider()
    return NimbleClient(api_key=api_key)


def _get_storage():
    return JsonStorageBackend()


@research_app.command("start")
def research_start(
    query: str = typer.Argument(..., help="Research question or objective"),
    budget: Optional[str] = typer.Option(None, "--budget", "-b", help="Time budget (30s/2m/5m/10m/30m/1h)"),
    format: str = typer.Option("full_report", "--format", "-f", help="Report format"),
    mock: bool = typer.Option(False, "--mock", help="Use mock Nimble provider"),
):
    """Start a new research session."""
    if budget:
        time_budget = TimeBudget(budget)
    else:
        time_budget = _ask_time_budget()

    request = UserResearchRequest(
        user_query=query,
        time_budget=time_budget,
        preferred_format=ReportFormat(format),
    )

    provider = MockNimbleProvider() if mock else _get_provider()
    storage = _get_storage()

    console.print(Panel(
        f"[bold]{query}[/bold]\n\nBudget: {time_budget.label} | Format: {format}",
        title="Research Session",
        border_style="blue",
    ))

    async def _run():
        summary = await run_research(request, provider, storage)
        return summary

    summary = asyncio.run(_run())

    console.print("\n")
    console.print(Panel(format_summary(summary), title="Session Summary", border_style="green"))

    # Show report if available
    report = asyncio.run(storage.load_report(str(summary.session_id)))
    if report:
        console.print("\n")
        formatted = format_report(report, ReportFormat(format))
        console.print(Panel(formatted, title="Research Report", border_style="cyan"))


@research_app.command("resume")
def research_resume(
    session_id: str = typer.Argument(..., help="Session ID to resume"),
):
    """Resume an interrupted research session."""
    storage = _get_storage()
    provider = _get_provider()

    config = asyncio.run(storage.load_session(session_id))
    if not config:
        console.print(f"[red]Session {session_id} not found[/red]")
        raise typer.Exit(1)

    request = UserResearchRequest(
        user_query=config.user_query,
        time_budget=config.time_budget,
    )

    console.print(f"[yellow]Resuming session {session_id}...[/yellow]")
    summary = asyncio.run(run_research(request, provider, storage, resume_session_id=session_id))
    console.print(Panel(format_summary(summary), title="Session Summary", border_style="green"))


@research_app.command("inspect")
def research_inspect(
    session_id: str = typer.Argument(..., help="Session ID to inspect"),
):
    """Inspect the state of a research session."""
    storage = _get_storage()

    config = asyncio.run(storage.load_session(session_id))
    if not config:
        console.print(f"[red]Session {session_id} not found[/red]")
        raise typer.Exit(1)

    checkpoint = asyncio.run(storage.load_latest_checkpoint(session_id))

    table = Table(title=f"Session: {session_id}")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Query", config.user_query)
    table.add_row("Budget", config.time_budget.label)
    table.add_row("Mode", config.execution_mode.value)
    table.add_row("Task Type", config.task_type.value)
    if checkpoint:
        table.add_row("Stage", checkpoint.stage.value)
        table.add_row("Progress", f"{checkpoint.progress_pct:.0f}%")
        table.add_row("Evidence", str(checkpoint.evidence_count))
        table.add_row("Elapsed", f"{checkpoint.elapsed_seconds:.1f}s")
    console.print(table)


@research_app.command("report")
def research_report(
    session_id: str = typer.Argument(..., help="Session ID"),
    format: str = typer.Option("full_report", "--format", "-f"),
):
    """Display or regenerate a research report."""
    storage = _get_storage()
    report = asyncio.run(storage.load_report(session_id))
    if not report:
        console.print(f"[red]No report found for session {session_id}[/red]")
        raise typer.Exit(1)

    formatted = format_report(report, ReportFormat(format))
    console.print(formatted)


@skill_app.command("inspect")
def skill_inspect(
    session_id: str = typer.Argument(..., help="Session ID"),
):
    """Show the generated skill spec for a session."""
    storage = _get_storage()
    skill = asyncio.run(storage.load_skill(session_id))
    if not skill:
        console.print(f"[red]No skill found for session {session_id}[/red]")
        raise typer.Exit(1)

    console.print_json(json.dumps(skill.model_dump(mode="json"), indent=2, default=str))


@skill_app.command("export")
def skill_export(
    session_id: str = typer.Argument(..., help="Session ID"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Export a generated skill spec as JSON."""
    storage = _get_storage()
    skill = asyncio.run(storage.load_skill(session_id))
    if not skill:
        console.print(f"[red]No skill found for session {session_id}[/red]")
        raise typer.Exit(1)

    data = json.dumps(skill.model_dump(mode="json"), indent=2, default=str)
    if output:
        with open(output, "w") as f:
            f.write(data)
        console.print(f"[green]Exported to {output}[/green]")
    else:
        console.print(data)


@session_app.command("list")
def session_list():
    """List all research sessions."""
    storage = _get_storage()
    sessions = asyncio.run(storage.list_sessions())

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Research Sessions")
    table.add_column("Session ID", max_width=36)
    table.add_column("Query", max_width=40)
    table.add_column("Budget")
    table.add_column("Report")
    table.add_column("Created")

    for s in sessions:
        table.add_row(
            s["session_id"][:12] + "...",
            s["user_query"][:40],
            s.get("time_budget", ""),
            "yes" if s.get("has_report") else "no",
            str(s.get("created_at", ""))[:19],
        )
    console.print(table)


@session_app.command("summary")
def session_summary(
    session_id: str = typer.Argument(..., help="Session ID"),
):
    """Show session summary."""
    storage = _get_storage()
    from pathlib import Path

    summary_path = Path(storage.base_dir) / session_id / "summary.json"
    if not summary_path.exists():
        console.print(f"[red]No summary found for session {session_id}[/red]")
        raise typer.Exit(1)

    from .models.output import SessionSummary

    data = json.loads(summary_path.read_text())
    summary = SessionSummary(**data)
    console.print(format_summary(summary))


if __name__ == "__main__":
    app()
