"""CLI interface for the research harness — enhanced with follow-up, batch, and streaming."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from .infra.events import EventStream
from .infra.logging import setup_logging
from .models.enums import ReportFormat, TimeBudget
from .models.session import UserResearchRequest
from .nimble.client import NimbleClient
from .nimble.mock import MockNimbleProvider
from .orchestrator.batch import batch_research
from .orchestrator.engine import run_research
from .orchestrator.followup import follow_up_research
from .orchestrator.gates import GateRegistry, build_cli_gate_handler
from .reports.formatter import format_report, format_summary
from .skillgen.exporter import export_skill_markdown, export_skill_yaml
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
    watch: bool = typer.Option(False, "--watch", "-w", help="Live progress display via SSE"),
    fast: bool = typer.Option(False, "--fast", help="Fast mode: trade freshness for speed"),
    output_schema: Optional[str] = typer.Option(None, "--output-schema", help="Path to JSON schema file"),
    prefer_domains: Optional[str] = typer.Option(None, "--prefer-domains", help="Comma-separated preferred domains"),
    block_domains: Optional[str] = typer.Option(None, "--block-domains", help="Comma-separated blocked domains"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Enable approval gates"),
):
    """Start a new research session."""
    if budget:
        time_budget = TimeBudget(budget)
    else:
        time_budget = _ask_time_budget()

    # Build source policy from CLI flags
    source_policy = None
    if prefer_domains or block_domains:
        source_policy = {}
        if prefer_domains:
            source_policy["preferred_domains"] = [d.strip() for d in prefer_domains.split(",")]
        if block_domains:
            source_policy["disallowed_domains"] = [d.strip() for d in block_domains.split(",")]

    # Load output schema if provided
    schema = None
    if output_schema:
        schema = json.loads(Path(output_schema).read_text())

    request = UserResearchRequest(
        user_query=query,
        time_budget=time_budget,
        preferred_format=ReportFormat(format),
        fast_mode=fast,
        output_schema=schema,
        source_policy=source_policy,
    )

    provider = MockNimbleProvider() if mock else _get_provider()
    storage = _get_storage()

    # Build gate registry
    gate_registry = None
    if interactive:
        gate_registry = GateRegistry(handler=build_cli_gate_handler())

    console.print(Panel(
        f"[bold]{query}[/bold]\n\nBudget: {time_budget.label} | Format: {format}"
        + (f" | Fast mode" if fast else "")
        + (f" | Interactive" if interactive else ""),
        title="Research Session",
        border_style="blue",
    ))

    async def _run():
        event_stream = EventStream(session_id="pending") if watch else None

        if watch and event_stream:
            # Run research and stream events concurrently
            async def _display_events():
                async for event in event_stream.listen():
                    _type = event.event_type
                    data = event.data
                    if _type == "stage.entered":
                        console.print(f"  [cyan]Stage:[/cyan] {data.get('stage', '')} ({data.get('elapsed_seconds', 0):.1f}s)")
                    elif _type == "tool.completed":
                        console.print(f"    [dim]{data.get('tool', '')}: {data.get('summary', '')}[/dim]")
                    elif _type == "finding.added":
                        console.print(f"    [green]+[/green] {data.get('summary', '')[:100]}")
                    elif _type == "budget.warning":
                        console.print(f"  [yellow]Budget: {data.get('remaining_pct', 0):.0f}% remaining[/yellow]")
                    elif _type == "session.completed":
                        console.print(f"  [green]Completed![/green]")
                    elif _type == "session.failed":
                        console.print(f"  [red]Failed: {data.get('error', '')}[/red]")

            research_task = asyncio.create_task(
                run_research(request, provider, storage, gate_registry=gate_registry, event_stream=event_stream)
            )
            display_task = asyncio.create_task(_display_events())

            summary = await research_task
            await event_stream.close()
            await display_task
            return summary
        else:
            return await run_research(request, provider, storage, gate_registry=gate_registry)

    summary = asyncio.run(_run())

    console.print("\n")
    console.print(Panel(format_summary(summary), title="Session Summary", border_style="green"))

    # Show report if available
    report = asyncio.run(storage.load_report(str(summary.session_id)))
    if report:
        console.print("\n")
        formatted = format_report(report, ReportFormat(format))
        console.print(Panel(formatted, title="Research Report", border_style="cyan"))


@research_app.command("follow-up")
def research_follow_up(
    session_id: str = typer.Argument(..., help="Previous session ID to build on"),
    query: str = typer.Argument(..., help="Follow-up research question"),
    budget: Optional[str] = typer.Option(None, "--budget", "-b", help="Time budget"),
):
    """Run follow-up research that builds on a prior session's findings."""
    if budget:
        time_budget = TimeBudget(budget)
    else:
        time_budget = _ask_time_budget()

    provider = _get_provider()
    storage = _get_storage()

    console.print(Panel(
        f"[bold]Follow-up:[/bold] {query}\n\n[dim]Building on session: {session_id}[/dim]",
        title="Follow-up Research",
        border_style="yellow",
    ))

    async def _run():
        return await follow_up_research(
            previous_session_id=session_id,
            new_query=query,
            time_budget=time_budget,
            provider=provider,
            storage=storage,
        )

    summary = asyncio.run(_run())
    console.print(Panel(format_summary(summary), title="Session Summary", border_style="green"))


@research_app.command("batch")
def research_batch(
    queries_file: str = typer.Argument(..., help="Path to JSONL file with queries"),
    budget: str = typer.Option("5m", "--budget", "-b", help="Time budget per query"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Max concurrent sessions"),
    mock: bool = typer.Option(False, "--mock", help="Use mock provider"),
):
    """Run batch research on multiple queries from a JSONL file."""
    queries_path = Path(queries_file)
    if not queries_path.exists():
        console.print(f"[red]File not found: {queries_file}[/red]")
        raise typer.Exit(1)

    queries = []
    for line in queries_path.read_text().strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            queries.append(data.get("query", data.get("q", line)))
        except json.JSONDecodeError:
            queries.append(line)

    if not queries:
        console.print("[red]No queries found in file[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Batch Research[/bold]\n\n"
        f"Queries: {len(queries)} | Budget: {budget} each | Concurrency: {concurrency}",
        title="Batch Research",
        border_style="blue",
    ))

    provider = MockNimbleProvider() if mock else _get_provider()
    storage = _get_storage()

    async def _run():
        return await batch_research(
            queries=queries,
            time_budget=TimeBudget(budget),
            provider=provider,
            storage=storage,
            concurrency=concurrency,
        )

    result = asyncio.run(_run())

    table = Table(title=f"Batch Results — {result.batch_id}")
    table.add_column("#", style="bold")
    table.add_column("Query", max_width=40)
    table.add_column("Status")
    table.add_column("Confidence")
    table.add_column("Evidence")

    for i, s in enumerate(result.summaries, 1):
        table.add_row(
            str(i),
            s.user_query[:40],
            s.final_stage.value,
            s.report_confidence or "n/a",
            str(s.total_evidence),
        )
    for i, err in enumerate(result.errors, len(result.summaries) + 1):
        table.add_row(str(i), err["query"][:40], "[red]failed[/red]", "-", "-")

    console.print(table)
    console.print(f"\n[bold]Success rate: {result.success_rate:.0f}%[/bold]")


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
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Export format: json, yaml, markdown"),
):
    """Export a generated skill spec."""
    storage = _get_storage()
    skill = asyncio.run(storage.load_skill(session_id))
    if not skill:
        console.print(f"[red]No skill found for session {session_id}[/red]")
        raise typer.Exit(1)

    if format == "markdown":
        data = export_skill_markdown(skill)
    elif format == "yaml":
        data = export_skill_yaml(skill)
    else:
        data = json.dumps(skill.model_dump(mode="json"), indent=2, default=str)

    if output:
        Path(output).write_text(data)
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
