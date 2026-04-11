# axe

A terminal-based autonomous coding agent that reads your codebase, writes code, runs commands, and fixes its own mistakes — while spending 100x less than Claude Code or Codex.

```
$ axe "find the bug in auth.py and fix it"

[gpt-4.1-mini] Searching codebase for auth.py...
[gpt-4.1-mini] Reading src/auth.py (lines 1-84)...
[gpt-4.1-mini] Found it — missing token expiry check on line 42.
[gpt-4.1-mini] Editing src/auth.py...
[gpt-4.1-mini] Running pytest tests/test_auth.py...
[gpt-4.1-mini] ✓ All 7 tests passing.

────────────────────────────────────────
  Session cost: $0.0028
  Calls: gpt-4.1-mini: 5
  Tokens: 3,420 in / 890 out
────────────────────────────────────────
```

## Why axe?

Every CLI agent today runs every single request through the most expensive model available. Ask it to `cat` a file? That's a $0.02 Claude Sonnet call. Run `pytest`? Another $0.02.

axe has a **three-tier model router** that picks the cheapest model that can handle each task:

| Tier | Model | Cost | Handles |
|------|-------|------|---------|
| 1 — Cheap | `gpt-4.1-mini` | $0.40/1M in | 70% of calls — reads, searches, simple edits |
| 2 — Mid | `gpt-4.1` | $2.00/1M in | 25% of calls — multi-file edits, complex debugging |
| 3 — Premium | `o3-mini` | $1.10/1M in | 5% of calls — architecture decisions, stuck loops |

The router doesn't predict difficulty. It starts cheap and **escalates only on observed failure** — 2 consecutive failures bumps to the next tier. One success resets to Tier 1.

**Typical session cost: $0.01–0.05.** Same session on Claude Code: $2–5.

## Quick start

```bash
# Clone and install
git clone https://github.com/sarthexe/axe.git
cd axe
pip install -e .

# Set your API key
export OPENAI_API_KEY=sk-...

# Run it
axe "list all TODO comments in this project"
```

## Features

**Autonomous agent loop** — Think → plan → act → observe → retry. The agent reads files before editing, runs code after changes, and self-corrects on errors. Capped at 10 iterations to prevent runaway loops.

**6 built-in tools:**
- `shell` — Execute commands with timeout and output capture
- `file_read` — Read files with line numbers and range support
- `file_write` — Create or overwrite files
- `file_edit` — Surgical find-and-replace (never rewrites whole files)
- `glob_search` — Find files by pattern, respects .gitignore
- `grep_search` — Search content across files

**Smart model routing** — One OpenAI API key, three model tiers. Auto-escalation on failure. Every action shows which model is handling it:
```
[gpt-4.1-mini] Reading src/main.py...
[gpt-4.1-mini] Editing src/main.py... FAILED
[gpt-4.1-mini] Retrying... FAILED
[↑ gpt-4.1] Escalating — 2 consecutive failures.
[gpt-4.1] Reading error context more carefully...
[gpt-4.1] Fixed — missing import on line 3.
```

**Cost tracking** — Every session shows total cost, calls per model, and token usage. Set a budget limit to prevent surprises. Check mid-session with `/cost`.

**Project memory** — The agent remembers your project across sessions. On first run, it detects your language, framework, and test runner. On subsequent runs, it already knows the codebase structure.
```
$ axe
[memory] Loaded project context: python, fastapi, pytest
>
```

**Safety first** — Destructive commands (`rm`, `drop`, `kill`) require confirmation. Known dangerous patterns (`rm -rf /`, `mkfs`) are blocked entirely. All file edits use surgical replacement, not full rewrites.

## Commands

| Command | What it does |
|---------|-------------|
| `/cost` | Show session cost breakdown |
| `/memory` | View/edit project memory |
| `/rewind <step>` | Restore files from a previous snapshot |
| `/explain <task>` | Preview a numbered execution plan with estimated cost |
| `/clear` | Clear conversation history |
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit (shows session summary) |

`--dry-run` mode previews actions without running shell commands or writing files.
Session events are logged to `.agent/logs/session.jsonl`.

## Cost comparison

Typical 30-minute coding session (reading files, editing code, running tests):

| Tool | Typical cost | Model |
|------|-------------|-------|
| Claude Code | $2.00–5.00 | Claude Sonnet (all calls) |
| Codex CLI | $1.00–3.00 | GPT-5 (all calls) |
| Aider | $0.50–2.00 | Depends on model choice |
| **axe** | **$0.01–0.05** | **Routed: mini/4.1/o3-mini** |

## Configuration

axe works with zero configuration — just set `OPENAI_API_KEY`. For customization, create `~/.axe/config.toml`:

```toml
[llm.tiers]
cheap = "gpt-4.1-mini"
mid = "gpt-4.1"
premium = "o3-mini"

[router]
escalate_after_failures = 2
show_model_in_output = true

[cost]
session_budget = 1.00

[agent]
max_iterations = 10
max_retries_per_tool = 3
```

Any OpenAI-compatible endpoint works — point `base_url` to Ollama, Groq, or OpenRouter:
```toml
[llm]
base_url = "http://localhost:11434/v1"
```

## Architecture

```
axe/
├── main.py              # CLI entry point (click)
├── agent/
│   ├── core.py          # Agent loop (think-plan-act-observe)
│   ├── planner.py       # Prompt construction + LLM calls
│   ├── context.py       # Token budget + sliding window
│   └── memory.py        # Conversation history
├── tools/
│   ├── registry.py      # Tool discovery + dispatch
│   ├── shell.py         # Command execution
│   ├── file_read.py     # Read with line numbers
│   ├── file_write.py    # Create/overwrite files
│   ├── file_edit.py     # Surgical str_replace
│   ├── glob_search.py   # Find files by pattern
│   └── grep_search.py   # Search content
├── llm/
│   ├── provider.py      # OpenAI SDK wrapper
│   └── router.py        # 3-tier model selection
├── cost/
│   ├── tracker.py       # Per-call cost logging
│   └── pricing.py       # Model pricing table
└── project/
    └── memory.py        # Persistent .axe/memory.json
```

## How the router works

```
User prompt
    │
    ▼
┌─────────────────────┐
│  Start: gpt-4.1-mini │ ◄── Always start cheap
└─────────┬───────────┘
          │
          ▼
     Tool call succeeds? ──Yes──► Reset to Tier 1
          │
          No (2 consecutive)
          │
          ▼
┌─────────────────────┐
│  Escalate: gpt-4.1   │ ◄── Try smarter model
└─────────┬───────────┘
          │
          ▼
     Still failing? (2 more)
          │
          ▼
┌─────────────────────┐
│  Escalate: o3-mini    │ ◄── Premium reasoning
└─────────────────────┘
```

## License

MIT

## Author

**Sarthak Maurya** — [github.com/sarthexe](https://github.com/sarthexe)
