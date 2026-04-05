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

benchmark_app = typer.Typer(help="Benchmark commands")

app.add_typer(research_app, name="research")
app.add_typer(skill_app, name="skill")
app.add_typer(session_app, name="session")
app.add_typer(benchmark_app, name="benchmark")


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


@research_app.command("export")
def research_export(
    session_id: str = typer.Argument(..., help="Session ID"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Export session as structured JSON (reference format: input, output, outputBasis)."""
    from .reports.formatter import export_session_json

    storage = _get_storage()
    report = asyncio.run(storage.load_report(session_id))
    if not report:
        console.print(f"[red]No report found for session {session_id}[/red]")
        raise typer.Exit(1)

    config = asyncio.run(storage.load_session(session_id))
    claims_data = asyncio.run(storage.get_claims(session_id))
    evidence_data = asyncio.run(storage.get_evidence(session_id))

    result = export_session_json(
        user_query=config.user_query if config else "",
        report=report,
        claims=claims_data,
        evidence=evidence_data,
    )

    data = json.dumps(result, indent=2, default=str)
    if output:
        Path(output).write_text(data)
        console.print(f"[green]Exported to {output}[/green]")
    else:
        console.print(data)


@research_app.command("excel")
def research_excel(
    session_id: str = typer.Argument(..., help="Session ID"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Export session as McKinsey-level Excel report."""
    from .reports.excel_export import export_excel

    storage = _get_storage()
    report_data = asyncio.run(storage.load_report(session_id))
    if not report_data:
        console.print(f"[red]No report found for session {session_id}[/red]")
        raise typer.Exit(1)

    config = asyncio.run(storage.load_session(session_id))
    claims_data = asyncio.run(storage.get_claims(session_id))
    evidence_data = asyncio.run(storage.get_evidence(session_id))

    report_dict = report_data.model_dump(mode="json") if hasattr(report_data, "model_dump") else report_data
    claims_list = [c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in claims_data]
    evidence_list = [e.model_dump(mode="json") if hasattr(e, "model_dump") else e for e in evidence_data]

    meta = {
        "time_budget": config.time_budget.label if config else "",
        "elapsed_seconds": 0,
    }

    out_path = output or f"{session_id[:12]}_report.xlsx"
    export_excel(
        output_path=out_path,
        user_query=config.user_query if config else "",
        report=report_dict,
        claims=claims_list,
        evidence=evidence_list,
        session_meta=meta,
    )
    console.print(f"[green]Excel report saved: {out_path}[/green]")


def _resolve_skill_session(storage, name_or_id: str, require_skill_json: bool = True) -> str:
    """Resolve a name-or-id to a session_id. Exits with error if not found."""
    match = asyncio.run(storage.find_skill(name_or_id))
    if match:
        sid = match["session_id"]
        if require_skill_json:
            skill = asyncio.run(storage.load_skill(sid))
            if not skill:
                console.print(f"[red]Session '{name_or_id}' exists but has no skill spec (skill generation may have been skipped).[/red]")
                raise typer.Exit(1)
        return sid
    # Fall back to raw session_id (for backward compat)
    skill = asyncio.run(storage.load_skill(name_or_id))
    if skill:
        return name_or_id
    console.print(f"[red]No skill found for '{name_or_id}'[/red]")
    console.print("[dim]Use 'nrh skill list' to see available skills.[/dim]")
    raise typer.Exit(1)


@skill_app.command("list")
def skill_list():
    """List all generated research skills by name."""
    storage = _get_storage()
    skills = asyncio.run(storage.list_skills())

    if not skills:
        console.print("[dim]No skills found.[/dim]")
        return

    table = Table(title="Research Skills")
    table.add_column("Name (slug)", style="cyan", max_width=40)
    table.add_column("Title", max_width=50)
    table.add_column("Budget")
    table.add_column("Report")
    table.add_column("Created")

    for s in skills:
        table.add_row(
            s.get("slug", "")[:40],
            s.get("title", "")[:50],
            s.get("time_budget", ""),
            "[green]yes[/green]" if s.get("has_report") else "[dim]no[/dim]",
            str(s.get("created_at", ""))[:19],
        )
    console.print(table)


@skill_app.command("show")
def skill_show(
    name_or_id: str = typer.Argument(..., help="Skill slug, session ID, or prefix"),
):
    """Show a skill spec by name or session ID."""
    storage = _get_storage()
    session_id = _resolve_skill_session(storage, name_or_id)
    skill = asyncio.run(storage.load_skill(session_id))

    console.print(Panel(
        f"[bold]{skill.title}[/bold]\n"
        f"Slug: [cyan]{skill.slug}[/cyan] | Session: {session_id[:12]}...\n"
        f"Task: {skill.task_type.value} | Budget: {skill.time_budget.value}\n"
        f"Created: {skill.created_at}",
        title="Skill Spec",
        border_style="blue",
    ))

    if skill.subquestions:
        console.print("\n[bold]Subquestions:[/bold]")
        for i, sq in enumerate(skill.subquestions, 1):
            console.print(f"  {i}. {sq}")

    if skill.target_entities:
        console.print(f"\n[bold]Target Entities:[/bold] {', '.join(skill.target_entities[:20])}")

    if skill.search_strategy.queries:
        console.print("\n[bold]Search Queries:[/bold]")
        for q in skill.search_strategy.queries[:10]:
            console.print(f"  - {q}")

    console.print(f"\n[bold]Policies:[/bold]")
    console.print(f"  Sources: min={skill.source_policy.min_sources}, diverse={skill.source_policy.require_diversity}")
    console.print(f"  Verification: strictness={skill.verification_policy.strictness}")
    console.print(f"  Synthesis: style={skill.synthesis_policy.style}, max_findings={skill.synthesis_policy.max_findings}")
    console.print(f"  Tools: {', '.join(t.value for t in skill.tool_permissions)}")


@skill_app.command("inspect")
def skill_inspect(
    name_or_id: str = typer.Argument(..., help="Skill slug, session ID, or prefix"),
):
    """Show the raw JSON skill spec for a session."""
    storage = _get_storage()
    session_id = _resolve_skill_session(storage, name_or_id)
    skill = asyncio.run(storage.load_skill(session_id))

    console.print_json(json.dumps(skill.model_dump(mode="json"), indent=2, default=str))


@skill_app.command("export")
def skill_export(
    name_or_id: str = typer.Argument(..., help="Skill slug, session ID, or prefix"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Export format: json, yaml, markdown"),
):
    """Export a generated skill spec."""
    storage = _get_storage()
    session_id = _resolve_skill_session(storage, name_or_id)
    skill = asyncio.run(storage.load_skill(session_id))

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


@skill_app.command("edit")
def skill_edit(
    name_or_id: str = typer.Argument(..., help="Skill slug, session ID, or prefix"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output YAML file path"),
):
    """Export a skill as editable YAML for modification."""
    storage = _get_storage()
    session_id = _resolve_skill_session(storage, name_or_id)
    skill = asyncio.run(storage.load_skill(session_id))

    data = export_skill_yaml(skill)
    out_path = output or f"{skill.slug}.yaml"
    Path(out_path).write_text(data)
    console.print(f"[green]Skill exported to {out_path}[/green]")
    console.print(f"[dim]Edit the file, then run: nrh skill import {out_path} --budget 5m[/dim]")


@skill_app.command("run")
def skill_run(
    name_or_id: str = typer.Argument(..., help="Skill slug, session ID, or prefix"),
    budget: Optional[str] = typer.Option(None, "--budget", "-b", help="Override time budget"),
    format: str = typer.Option("full_report", "--format", "-f", help="Report format"),
    mock: bool = typer.Option(False, "--mock", help="Use mock Nimble provider"),
):
    """Re-run a previously generated skill (optionally with a different budget)."""
    storage = _get_storage()
    session_id = _resolve_skill_session(storage, name_or_id)
    skill = asyncio.run(storage.load_skill(session_id))

    time_budget = TimeBudget(budget) if budget else skill.time_budget

    request = UserResearchRequest(
        user_query=skill.user_objective,
        time_budget=time_budget,
        preferred_format=ReportFormat(format),
    )

    provider = MockNimbleProvider() if mock else _get_provider()

    console.print(Panel(
        f"[bold]Re-running skill:[/bold] {skill.title}\n"
        f"Slug: [cyan]{skill.slug}[/cyan]\n"
        f"Budget: {time_budget.label} | Format: {format}\n"
        f"Subquestions: {len(skill.subquestions)} | Entities: {len(skill.target_entities)}",
        title="Skill Re-Run",
        border_style="green",
    ))

    async def _run():
        summary = await run_research(
            request=request,
            provider=provider,
            storage=storage,
            skill_override=skill,
        )
        return summary

    summary = asyncio.run(_run())
    console.print(format_summary(summary))

    # Show report
    report = asyncio.run(storage.load_report(str(summary.session_id)))
    if report:
        console.print(format_report(report))


@skill_app.command("import")
def skill_import_cmd(
    file: str = typer.Argument(..., help="Path to YAML or JSON skill file"),
    budget: Optional[str] = typer.Option(None, "--budget", "-b", help="Time budget"),
    format: str = typer.Option("full_report", "--format", "-f", help="Report format"),
    mock: bool = typer.Option(False, "--mock", help="Use mock Nimble provider"),
):
    """Import and run a skill from a YAML/JSON file."""
    import uuid as _uuid

    import yaml

    from .models.skill import DynamicSkillSpec

    path = Path(file)
    if not path.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    raw = path.read_text()
    if file.endswith((".yaml", ".yml")):
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    # Assign new IDs
    data["session_id"] = str(_uuid.uuid4())
    data["skill_id"] = str(_uuid.uuid4())

    skill = DynamicSkillSpec(**data)
    time_budget = TimeBudget(budget) if budget else skill.time_budget

    request = UserResearchRequest(
        user_query=skill.user_objective,
        time_budget=time_budget,
        preferred_format=ReportFormat(format),
    )

    provider = MockNimbleProvider() if mock else _get_provider()
    storage = _get_storage()

    console.print(Panel(
        f"[bold]Imported skill:[/bold] {skill.title}\n"
        f"Budget: {time_budget.label} | Format: {format}",
        title="Skill Import & Run",
        border_style="green",
    ))

    async def _run():
        return await run_research(
            request=request,
            provider=provider,
            storage=storage,
            skill_override=skill,
        )

    summary = asyncio.run(_run())
    console.print(format_summary(summary))

    report = asyncio.run(storage.load_report(str(summary.session_id)))
    if report:
        console.print(format_report(report))


@session_app.command("list")
def session_list():
    """List all research sessions."""
    storage = _get_storage()
    sessions = asyncio.run(storage.list_sessions())

    if not sessions:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title="Research Sessions")
    table.add_column("Skill Name", style="cyan", max_width=30)
    table.add_column("Session ID", max_width=14)
    table.add_column("Query", max_width=40)
    table.add_column("Budget")
    table.add_column("Report")
    table.add_column("Created")

    for s in sessions:
        table.add_row(
            s.get("slug", "")[:30] or "[dim]-[/dim]",
            s["session_id"][:12] + "...",
            s["user_query"][:40],
            s.get("time_budget", ""),
            "[green]yes[/green]" if s.get("has_report") else "[dim]no[/dim]",
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


# --- Benchmark Commands ---


@benchmark_app.command("run")
def benchmark_run(
    queries_file: str = typer.Argument(..., help="Path to queries JSONL file"),
    budgets: str = typer.Option("2m,5m,10m", "--budgets", help="Comma-separated budgets to test"),
    concurrency: int = typer.Option(1, "--concurrency", "-c", help="Max concurrent runs"),
    output_dir: str = typer.Option(".benchmark_runs", "--output-dir", "-o", help="Output directory"),
    mock: bool = typer.Option(False, "--mock", help="Use mock provider"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume a previous run ID"),
):
    """Run a benchmark: each query x each budget tier.

    Queries file: JSONL with {"id": "q001", "query": "..."} per line,
    or plain text (one query per line, auto-numbered).
    """
    from .benchmark.runner import run_benchmark

    queries_path = Path(queries_file)
    if not queries_path.exists():
        console.print(f"[red]File not found: {queries_file}[/red]")
        raise typer.Exit(1)

    # Parse queries
    queries = []
    for i, line in enumerate(queries_path.read_text().strip().split("\n"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            queries.append({
                "id": data.get("id", f"q{i:03d}"),
                "query": data.get("query", data.get("q", line)),
            })
        except json.JSONDecodeError:
            queries.append({"id": f"q{i:03d}", "query": line})

    if not queries:
        console.print("[red]No queries found in file[/red]")
        raise typer.Exit(1)

    budget_list = [TimeBudget(b.strip()) for b in budgets.split(",")]
    provider = MockNimbleProvider() if mock else _get_provider()

    total_runs = len(queries) * len(budget_list)
    console.print(Panel(
        f"[bold]Benchmark Run[/bold]\n\n"
        f"Queries: {len(queries)} | Budgets: {budgets} | Total runs: {total_runs}\n"
        f"Concurrency: {concurrency} | Output: {output_dir}\n"
        f"Provider: {'mock' if mock else 'live'}",
        title="Benchmark",
        border_style="blue",
    ))

    async def _run():
        return await run_benchmark(
            queries=queries,
            provider=provider,
            output_dir=output_dir,
            budgets=budget_list,
            concurrency=concurrency,
            resume_run_id=resume,
        )

    result = asyncio.run(_run())

    console.print(f"\n[green]Benchmark complete: {result.run_id}[/green]")
    console.print(f"  Completed: {result.completed}/{result.total_runs}")
    console.print(f"  Failed: {result.failed}")
    console.print(f"  Results: {output_dir}/{result.run_id}/")
    console.print(f"\nRun [bold]nrh benchmark scorecard {result.run_id}[/bold] to see analysis.")


@benchmark_app.command("scorecard")
def benchmark_scorecard(
    run_id: str = typer.Argument(..., help="Benchmark run ID"),
    output_dir: str = typer.Option(".benchmark_runs", "--output-dir", "-o"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json, csv"),
):
    """Show scorecard for a benchmark run."""
    from .benchmark.analyzer import (
        build_scorecard,
        format_scorecard_csv,
        format_scorecard_text,
        load_results,
    )

    run_dir = Path(output_dir) / run_id
    if not run_dir.exists():
        console.print(f"[red]Run not found: {run_dir}[/red]")
        raise typer.Exit(1)

    results = load_results(run_dir)
    if not results:
        console.print("[red]No results found[/red]")
        raise typer.Exit(1)

    scorecard = build_scorecard(results)

    if format == "json":
        console.print_json(json.dumps(scorecard, indent=2, default=str))
    elif format == "csv":
        csv_text = format_scorecard_csv(scorecard)
        console.print(csv_text)
        # Also save to file
        csv_path = run_dir / "scorecard.csv"
        csv_path.write_text(csv_text)
        console.print(f"\n[dim]Saved to {csv_path}[/dim]")
    else:
        text = format_scorecard_text(scorecard)
        console.print(text)
        # Also save to file
        text_path = run_dir / "scorecard.txt"
        text_path.write_text(text)
        console.print(f"\n[dim]Saved to {text_path}[/dim]")


@benchmark_app.command("list")
def benchmark_list(
    output_dir: str = typer.Option(".benchmark_runs", "--output-dir", "-o"),
):
    """List all benchmark runs."""
    runs_dir = Path(output_dir)
    if not runs_dir.exists():
        console.print("[dim]No benchmark runs found.[/dim]")
        return

    table = Table(title="Benchmark Runs")
    table.add_column("Run ID")
    table.add_column("Queries")
    table.add_column("Budgets")
    table.add_column("Completed")
    table.add_column("Failed")
    table.add_column("Rate")

    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir():
            summary_path = d / "summary.json"
            if summary_path.exists():
                data = json.loads(summary_path.read_text())
                table.add_row(
                    data.get("run_id", d.name),
                    str(data.get("total_queries", "?")),
                    ", ".join(data.get("budgets", [])),
                    str(data.get("completed", "?")),
                    str(data.get("failed", "?")),
                    f"{data.get('success_rate', 0):.0f}%",
                )
            else:
                # In-progress run — count JSONL lines
                results_path = d / "results.jsonl"
                count = 0
                if results_path.exists():
                    count = sum(1 for line in results_path.read_text().strip().split("\n") if line.strip())
                table.add_row(d.name, "?", "?", str(count), "?", "[yellow]in progress[/yellow]")

    console.print(table)


@benchmark_app.command("inspect")
def benchmark_inspect(
    run_id: str = typer.Argument(..., help="Benchmark run ID"),
    query_id: Optional[str] = typer.Option(None, "--query", "-q", help="Filter by query ID"),
    budget: Optional[str] = typer.Option(None, "--budget", "-b", help="Filter by budget"),
    output_dir: str = typer.Option(".benchmark_runs", "--output-dir", "-o"),
):
    """Inspect individual results from a benchmark run."""
    from .benchmark.analyzer import load_results

    run_dir = Path(output_dir) / run_id
    results = load_results(run_dir)

    if query_id:
        results = [r for r in results if r["query_id"] == query_id]
    if budget:
        results = [r for r in results if r["budget"] == budget]

    if not results:
        console.print("[red]No matching results[/red]")
        raise typer.Exit(1)

    for r in results:
        status_color = "green" if r["status"] == "completed" else "red"
        console.print(Panel(
            f"Query: {r['query'][:100]}\n"
            f"Status: [{status_color}]{r['status']}[/{status_color}] | "
            f"Elapsed: {r['elapsed_seconds']:.1f}s\n"
            f"Evidence: {r['total_evidence']} | Sources: {r['total_sources']} | "
            f"Claims: {r['total_claims']} ({r['verified_claims']} verified)\n"
            f"Tool calls: {r['total_tool_calls']} | Confidence: {r['confidence']}\n"
            f"Session: {r['session_id']}\n"
            + (f"Error: {r['error']}" if r["error"] else "")
            + (f"\n\nExcerpt: {r['report_excerpt'][:300]}" if r.get("report_excerpt") else ""),
            title=f"[bold]{r['query_id']}[/bold] @ {r['budget']}",
            border_style="cyan",
        ))


# --- BrowseComp Commands ---

browsecomp_app = typer.Typer(help="BrowseComp benchmark commands")
app.add_typer(browsecomp_app, name="browsecomp")


@browsecomp_app.command("run")
def browsecomp_run(
    budget: str = typer.Option("2m", "--budget", "-b", help="Time budget per question"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit number of questions"),
    concurrency: int = typer.Option(2, "--concurrency", "-c", help="Max concurrent runs"),
    output_dir: str = typer.Option(".browsecomp_runs", "--output-dir", "-o"),
    csv_path: str = typer.Option("benchmarks/browse_comp_test_set.csv", "--csv", help="Path to BrowseComp CSV"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume a previous run ID"),
    mock: bool = typer.Option(False, "--mock", help="Use mock provider"),
    mode: str = typer.Option("standard", "--mode", "-m", help="Mode: standard or deep (multi-hop)"),
):
    """Run BrowseComp benchmark: research + grade against ground truth."""
    from .benchmark.browsecomp import load_browsecomp, run_browsecomp

    if not Path(csv_path).exists():
        console.print(f"[red]CSV not found: {csv_path}[/red]")
        console.print("Download with: curl -O https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv")
        raise typer.Exit(1)

    questions = load_browsecomp(csv_path, limit=limit)
    provider = MockNimbleProvider() if mock else _get_provider()
    time_budget = TimeBudget(budget)

    console.print(Panel(
        f"[bold]BrowseComp Benchmark[/bold]\n\n"
        f"Questions: {len(questions)} | Budget: {budget} | Concurrency: {concurrency}\n"
        f"Mode: [bold]{mode}[/bold] | Provider: {'mock' if mock else 'live'}",
        title="BrowseComp",
        border_style="blue",
    ))

    async def _run():
        return await run_browsecomp(
            questions=questions,
            provider=provider,
            budget=time_budget,
            output_dir=output_dir,
            concurrency=concurrency,
            resume_run_id=resume,
            mode=mode,
        )

    result = asyncio.run(_run())

    console.print(f"\n[green]BrowseComp complete: {result.run_id}[/green]")
    console.print(f"  Accuracy: [bold]{result.accuracy:.1f}%[/bold] ({result.correct}/{result.completed})")
    console.print(f"  Failed: {result.failed}")
    console.print(f"\nRun [bold]nrh browsecomp report {result.run_id}[/bold] for detailed analysis.")


@browsecomp_app.command("report")
def browsecomp_report(
    run_id: str = typer.Argument(..., help="BrowseComp run ID"),
    output_dir: str = typer.Option(".browsecomp_runs", "--output-dir", "-o"),
    format: str = typer.Option("text", "--format", "-f", help="Output format: text, json"),
):
    """Show detailed report for a BrowseComp run."""
    from .benchmark.browsecomp import analyze_browsecomp_run, format_browsecomp_report

    run_dir = Path(output_dir) / run_id
    if not run_dir.exists():
        console.print(f"[red]Run not found: {run_dir}[/red]")
        raise typer.Exit(1)

    analysis = analyze_browsecomp_run(run_dir)

    if format == "json":
        console.print_json(json.dumps(analysis, indent=2, default=str))
    else:
        text = format_browsecomp_report(analysis)
        console.print(text)
        report_path = run_dir / "report.txt"
        report_path.write_text(text)
        console.print(f"\n[dim]Saved to {report_path}[/dim]")


@browsecomp_app.command("list")
def browsecomp_list(
    output_dir: str = typer.Option(".browsecomp_runs", "--output-dir", "-o"),
):
    """List all BrowseComp runs."""
    runs_dir = Path(output_dir)
    if not runs_dir.exists():
        console.print("[dim]No BrowseComp runs found.[/dim]")
        return

    table = Table(title="BrowseComp Runs")
    table.add_column("Run ID")
    table.add_column("Budget")
    table.add_column("Questions")
    table.add_column("Correct")
    table.add_column("Accuracy")

    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir():
            summary_path = d / "summary.json"
            if summary_path.exists():
                data = json.loads(summary_path.read_text())
                table.add_row(
                    data.get("run_id", d.name),
                    data.get("budget", "?"),
                    str(data.get("total_questions", "?")),
                    f"{data.get('correct', '?')}/{data.get('completed', '?')}",
                    f"{data.get('accuracy', 0):.1f}%",
                )

    console.print(table)


@app.command("deep-research")
def deep_research_cmd(
    question: str = typer.Argument(..., help="Question to research (multi-hop)"),
    timeout: int = typer.Option(1740, "--timeout", "-t", help="Wall-clock timeout in seconds"),
    max_turns: int = typer.Option(50, "--max-turns", help="Maximum tool call turns"),
    mock: bool = typer.Option(False, "--mock", help="Use mock provider"),
):
    """Run agentic deep research on a single question (BrowseComp-style).

    Uses Claude Opus with direct tool access for multi-hop reasoning.
    Default timeout: 29 minutes (for 30m budget with margin).
    """
    from .deepresearch.agentic import agentic_research

    provider = MockNimbleProvider() if mock else _get_provider()

    console.print(Panel(
        f"[bold]{question[:100]}[/bold]\n\n"
        f"Model: Claude Opus 4.6 | Max turns: {max_turns} | Timeout: {timeout}s",
        title="Deep Research (Agentic)",
        border_style="magenta",
    ))

    async def _run():
        return await agentic_research(
            question=question,
            provider=provider,
            timeout_seconds=timeout,
            max_turns=max_turns,
        )

    session = asyncio.run(_run())

    # Display results
    if session.final_answer:
        console.print(f"\n[bold green]Answer: {session.final_answer}[/bold green]")
        console.print(f"Confidence: {session.final_confidence:.0%}")
    else:
        console.print("\n[bold red]No answer found[/bold red]")

    console.print(f"\nHops: {len(session.hops)} | Searches: {session.total_searches} | "
                  f"Extracts: {session.total_extracts} | LLM calls: {session.total_llm_calls} | "
                  f"Candidates: {len(session.candidates)} | Elapsed: {session.elapsed_seconds:.0f}s")

    if session.candidates:
        console.print("\n[bold]Candidates:[/bold]")
        for c in sorted(session.candidates, key=lambda x: x.confidence, reverse=True)[:5]:
            met = len(c.constraints_met)
            console.print(f"  [{c.confidence:.0%}] {c.answer} (met {met} constraints)")

    if session.constraints:
        console.print("\n[bold]Constraints:[/bold]")
        for c in session.constraints:
            icon = "[green]+[/green]" if c.is_met else "[red]-[/red]"
            console.print(f"  {icon} [{c.category}] {c.text}")


if __name__ == "__main__":
    app()
