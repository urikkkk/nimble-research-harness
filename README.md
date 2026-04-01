# Nimble Research Harness

A production-grade dynamic web research agent platform powered by [Nimble](https://nimbleway.com).

Unlike static research tools with hardcoded skills, this harness **dynamically generates a task-specific research skill** for every user request, deploys it into a runtime, and executes it with time-budgeted depth control.

## Key Features

- **Dynamic skill generation** — every research request gets a custom-built skill spec
- **Time-budgeted execution** — choose 30s to 1h; depth/breadth/verification scale accordingly
- **WSA-aware hybrid execution** — discovers and uses Nimble's 459+ pre-built Web Search Agents alongside raw tools
- **10-stage pipeline** — intake, discovery, skill gen, deployment, planning, research, extraction, analysis, verification, reporting
- **Evidence-first** — every claim cites sources; verification flags confidence levels
- **Persistent sessions** — checkpoint after each stage; resume interrupted runs
- **Structured output** — brief, full report, JSON, evidence table, or source pack formats

## Architecture

```
CLI / API
    │
    ▼
┌─────────────────────┐
│    Orchestrator      │ 10-stage pipeline with checkpoints
├─────────────────────┤
│  Claude SDK Agents   │ skill_builder, planner, analyst, verifier
│  Function Agents     │ intake, researcher, extractor, monitor
├─────────────────────┤
│    Tool Registry     │ Nimble Search/Extract/Map/Crawl/Agents
├─────────────────────┤
│  WSA Discovery       │ 459+ pre-built agents, scored & selected
├─────────────────────┤
│  JSON Persistence    │ .research_sessions/{id}/
└─────────────────────┘
```

## Quick Start

```bash
# Install
uv pip install -e ".[dev]"

# Set up environment
cp .env.example .env
# Edit .env with your NIMBLE_API_KEY and ANTHROPIC_API_KEY

# Run research (interactive — asks for time budget)
nrh research start "What is Nimble's competitive position in web scraping?"

# Run with explicit budget
nrh research start "Compare Bright Data vs Nimble" --budget 5m

# Run with mock provider (no API keys needed)
nrh research start "Test query" --mock --budget 30s

# List sessions
nrh session list

# Inspect a session
nrh research inspect <session_id>

# View generated skill spec
nrh skill inspect <session_id>

# Re-view report
nrh research report <session_id>
```

## Time Budgets

| Budget | Searches | Extracts | Crawl | Concurrency | Verification |
|--------|----------|----------|-------|-------------|-------------|
| 30s    | 3        | 1        | 0     | 3           | Skip        |
| 2m     | 8        | 5        | 0     | 5           | Spot-check  |
| 5m     | 20       | 15       | 50    | 8           | Key claims  |
| 10m    | 40       | 30       | 200   | 10          | Key claims  |
| 30m    | 100      | 80       | 1000  | 15          | All claims  |
| 1h     | 200      | 500      | 5000  | 20          | All claims  |

## How It Works

1. **Intake** — classify the query, suggest time budget
2. **WSA Discovery** — scan 459+ pre-built Nimble agents, score fit against task
3. **Skill Generation** — Claude builds a `DynamicSkillSpec` with subquestions, strategies, and policies
4. **Deployment** — register the skill and its tools into the runtime
5. **Planning** — Claude creates an execution plan with ordered tool calls
6. **Research** — parallel fan-out of searches, WSA runs, and extractions
7. **Extraction** — deep content extraction from priority URLs
8. **Analysis** — Claude synthesizes evidence into claims and report
9. **Verification** — Claude cross-checks claims against evidence
10. **Reporting** — format and persist the final output

## Optional API Server

```bash
# Install API dependencies
uv pip install -e ".[api]"

# Start server
uvicorn nimble_research_harness.serve:app --reload

# POST /research/start
curl -X POST http://localhost:8000/research/start \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Nimble?", "time_budget": "2m"}'
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nimble_research_harness

# Run specific test module
pytest tests/test_models/
```

## Project Structure

```
src/nimble_research_harness/
  orchestrator/    # 10-stage pipeline engine
  agents/          # Claude SDK + function agents
  skillgen/        # Dynamic skill generation & deployment
  tools/           # Tool registry + Nimble tool definitions
  nimble/          # Nimble API client, provider protocol, mock
  wsa/             # WSA catalog, scorer, hybrid strategy
  models/          # Pydantic v2 data models
  budget/          # Time budget presets
  storage/         # JSON file persistence
  reports/         # Report formatters
  infra/           # Errors, retry, logging, context
  cli.py           # Typer CLI
  serve.py         # FastAPI wrapper
```

## License

MIT
