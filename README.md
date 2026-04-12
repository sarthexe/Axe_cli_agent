# axe

A terminal-based autonomous coding agent that reads your codebase, runs shell commands, edits files, and tracks cost with automatic model routing.

## Why axe?

Most CLI coding tools send every step to an expensive model. `axe` starts cheap and escalates only when needed.

| Tier | Model | Cost (in/out per 1M) | Typical use |
|------|-------|----------------------|-------------|
| T1 | `gpt-4.1-mini` | $0.40 / $1.60 | Reads, search, simple edits, shell commands |
| T2 | `gpt-4.1` | $2.00 / $8.00 | Harder debugging, multi-file work |
| T3 | `o3-mini` | $1.10 / $4.40 | Escalated reasoning when lower tiers fail |

## Quick Start

```bash
# Clone and install
git clone https://github.com/sarthexe/cli_agent.git
cd cli_agent
pip install -e .

# Required environment variable
export OPENAI_API_KEY=sk-...

# Single-shot mode
axe "list all TODO comments in this project"

# REPL mode
axe
```

## Features Implemented Today

- Autonomous tool loop (`think -> tool call -> observe -> retry`) with iteration caps.
- Six built-in tools:
  - `shell`
  - `file_read`
  - `file_write`
  - `file_edit`
  - `glob_search`
  - `grep_search`
- Safety controls:
  - Blocked dangerous commands (`rm -rf /`, `mkfs`, `dd if=/dev/zero`, fork bomb patterns)
  - Destructive command confirmation prompt: `[y/n/edit]`
  - Truthfulness guard after blocked/skipped actions
- `--dry-run` mode (previews actions without writing files or executing shell).
- `/explain <task>` to preview a numbered execution plan with a cost estimate.
- Project memory persisted at `.agent/memory.json`.
- Session snapshots for `/rewind` at `.agent/snapshots/`.
- Cost tracking with per-model breakdown and budget visibility.
- Structured session logging at `.agent/logs/session.jsonl`.

## CLI Usage

```bash
axe [PROMPT] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-m, --model` | Override model for this run |
| `-c, --config` | Use custom TOML config file |
| `-v, --verbose` | Enable verbose logging |
| `--dry-run` | Preview changes without mutating files/running shell |
| `--version` | Print version and exit |

## REPL Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Show all available REPL commands | `/help` |
| `/exit` | Exit REPL and print cost summary | `/exit` |
| `/quit` | Alias for `/exit` | `/quit` |
| `/cost` | Show session cost table | `/cost` |
| `/version` | Show app and Python version | `/version` |
| `/memory` | Show project memory table | `/memory` |
| `/memory set` | Update memory key/value | `/memory set language=python` |
| `/memory detect` | Re-run memory auto-detection | `/memory detect` |
| `/rewind` | List available snapshot steps | `/rewind` |
| `/rewind <step>` | Restore files from a snapshot step | `/rewind 3` |
| `/clear` | Clear conversation history (keep memory) | `/clear` |
| `/explain <task>` | Preview execution plan + estimated cost | `/explain fix failing tests` |

## Configuration

Default project config file:

```text
config/config.toml
```

You can override it per run:

```bash
axe --config /path/to/config.toml "your prompt"
```

Environment variables supported:

- `OPENAI_API_KEY` (required)
- `OPENAI_BASE_URL` (optional OpenAI-compatible endpoint override)

Minimal config example:

```toml
[llm]
default_provider = "openai"
tier1_model = "gpt-4.1-mini"
tier2_model = "gpt-4.1"
tier3_model = "o3-mini"

[llm.openai]
base_url = "https://api.openai.com/v1"
temperature = 0.0

[agent]
max_iterations = 25
max_retries_per_tool = 3
confirmation_required = ["rm", "drop", "kill", "truncate"]
```

## Architecture

```text
axe/
├── main.py
├── agent/
│   ├── core.py
│   ├── context.py
│   ├── snapshots.py
│   └── tokens.py
├── tools/
│   ├── base.py
│   ├── registry.py
│   ├── shell.py
│   ├── file_read.py
│   ├── file_write.py
│   ├── file_edit.py
│   ├── glob_search.py
│   └── grep_search.py
├── llm/
│   ├── provider.py
│   ├── openai_provider.py
│   └── router.py
├── cost/
│   ├── pricing.py
│   └── tracker.py
├── project/
│   └── memory.py
├── sandbox/
│   └── permissions.py
└── utils/
    └── logger.py
```

## Demo

Terminal recording (GIF/asciinema): coming soon.

## Status

Current repo state is focused on shipping the core agent loop + safety/cost/memory workflow with OpenAI routing.

## License

MIT (declared in `pyproject.toml`; root `LICENSE` file can be added in a follow-up).

## Author

Sarthak Maurya - [github.com/sarthexe](https://github.com/sarthexe)
