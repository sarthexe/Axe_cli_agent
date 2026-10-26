# axe

A terminal-based autonomous coding agent that reads your codebase, runs shell commands, edits files, and tracks cost with automatic model routing.

## Why axe?

Most CLI coding tools send every step to an expensive model. `axe` routes each task to an **open-weight tier** and falls back across providers only when needed. **Jev** evaluates each turn in one fast pass; deterministic Python policy owns the actual routing, including intra-loop downshifting and failure escalation.

| Tier | Default model | Typical use |
|------|---------------|-------------|
| `fast` | `qwen/qwen3-coder-next` | Reads, search, simple edits, shell commands |
| `coder` | `deepseek/deepseek-v4` | Default workhorse for multi-file changes |
| `architect` | `deepseek/deepseek-v4-pro` | Hard debugging and large refactors |
| `security` | `zhipu/glm-5.3` | Security review and remediation |

Each tier has an ordered fallback list in `src/axe/config/default_config.toml`; a failed, rate-limited, keyless, or unreachable candidate is skipped automatically.

## Quick Start

```bash
# Clone and install (editable)
git clone https://github.com/sarthexe/cli_agent.git
cd cli_agent
uv sync

# Pick at least one provider
export OPENROUTER_API_KEY="your-openrouter-api-key"
# or DEEPSEEK_API_KEY / ZHIPU_API_KEY, or run fully local with Ollama

uv run axe --help

# Single-shot mode
uv run axe "list all TODO comments in this project"

# REPL mode
uv run axe
```

The installed console script is `axe`, so after `uv sync` you can also call `axe ...` directly inside the project environment.

## Features Implemented Today

- Autonomous tool loop (`think -> tool call -> observe -> retry`) with iteration caps.
- Six core tools:
  - `shell`
  - `file_read`
  - `file_write`
  - `file_edit`
  - `glob_search`
  - `grep_search`
- `airlock_scan` is always available through the external MCP server or the built-in cascade fallback.
- Safety controls:
  - Blocked dangerous commands (`rm -rf /`, `mkfs`, `dd if=/dev/zero`, fork bomb patterns)
  - Destructive command confirmation prompt: `[y/n/edit]`
  - Truthfulness guard after blocked/skipped actions
- `--dry-run` mode (previews actions without writing files or executing shell).
- `/explain <task>` to preview a numbered execution plan with a cost estimate.
- `/tier` to inspect routing mode, Jev intent/confidence/risk, and the last decision; `/tier auto` or `/tier <name>` switches modes.
- Jev decision engine:
  - One `/evaluate` pass with five typed questions: task intent, reasoning depth, thinking need, destructive risk, and AST sufficiency.
  - 500ms routing budget, TypeSafe/OpenRouter response normalization, and `typesafe/jev-latest` cost accounting.
  - Deterministic heuristic fallback when Jev is disabled, keyless, slow, or returns an error.
  - Risk-gated security routing, confidence-gated fallback, and intent/depth tier mapping.
  - Intra-loop auto-downshifting to `fast` on routine verification turns and escalation to `coder`/`architect` after failures or tracebacks.
- Multi-provider engine:
  - OpenRouter, DeepSeek, Zhipu/GLM, Moonshot/Kimi, Fireworks, vLLM, llama.cpp, and native Ollama.
  - Cross-provider fallback within a tier; Ollama candidates are skipped when the daemon is down or the model is not pulled.
  - Thinking-mode mapping (`off` / `low` / `medium` / `high`) across hybrid open-weight models.
  - Native `reasoning_content` and inline `<think>...</think>` normalization.
- KV-cache-aware cost tracking:
  - Cache-hit vs cache-miss input pricing, cache savings (USD), and cache hit ratio.
  - Reasoning-token accounting and per-tier/provider call breakdown.
  - Local throughput (`tok/s`), latency, and TTFT metrics.
- Project memory persisted at `.axe/memory.json`.
- Session snapshots for `/rewind` at `.axe/snapshots/`.
- Structured session logging at `.axe/logs/session.jsonl`.
- Layered TOML configuration with `AXE_*` environment overrides.
- Code-Airlock security:
  - Bidirectional outbound redaction and inbound local restoration through a process-local ephemeral vault.
  - Deterministic placeholders (`<AIRLOCK_SECRET_001>`, `<AIRLOCK_PII_001>`) shared across files.
  - Built-in high-precision cascade scanner for provider keys, private keys, JWTs, bearer tokens, DB URLs, `.env` assignments, and developer PII.
  - Native MCP JSON-RPC clients over stdio and SSE/HTTP; `code-airlock` boots as the default MCP server when configured.
  - Graceful built-in `airlock_scan` fallback when the external binary is missing or fails its handshake.
  - `/airlock` status and on-demand path/text scanning without exposing raw values.
- Guided security remediation:
  - Every `file_write`, `file_edit`, and `shell` call is scanned before it reaches disk or a process.
  - Deterministic fixes: hardcoded credentials become `os.environ.get(...)` / `process.env.*` lookups, `.env` / `.env.example` / `.gitignore` are updated, and `yaml.load()` becomes `yaml.safe_load()`.
  - Interactive `[a]uto` / `[e]xtract` / `[i]gnore once` / `[b]lock` flow, or `strict_block` mode that feeds a remediation hint back to the agent.
- Turn snapshots and instant rollback:
  - `/snapshots` lists recent file-modifying turns; `/undo` restores the pre-turn state (including deleting newly created files).
  - `SnapshotManager.run_speculative_check()` runs a verification command and automatically rolls back on failure.
- AST/symbol context compression:
  - Python files are parsed with `ast`; imports, classes, signatures, annotations, decorators, docstrings, and line spans are kept while bodies are replaced by `...`.
  - JS/TS/Go/Rust/C/C++ files use structural signature extraction.
  - Jev's `use_ast_compression` signal (or a near-budget context) triggers compression; `/cost` and `/tier` report the saved tokens.
- Scoped project memory:
  - `.axe/memory.json` stores glob/tag-scoped rules plus a 10-entry failed-attempt ring buffer.
  - Matching failures inject a compact `<AvoidRepeatingFailures>` block so the agent does not retry the same broken fix.
- Pre-commit Airlock gate:
  - `axe --install-hook` writes `.git/hooks/pre-commit` and preserves any existing hook as `pre-commit.axe.bak`.
  - `axe --run-pre-commit` scans staged added lines and blocks secrets, PII, and critical vulnerabilities with a raw-free report.

## CLI Usage

```bash
axe [PROMPT] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--tier [auto\|fast\|coder\|architect\|security]` | Override the Jev auto-router (default: `auto`) |
| `--no-airlock` | Bypass outbound Code-Airlock redaction (requires confirmation) |
| `-y, --yes` | Confirm `--no-airlock` non-interactively |
| `--install-hook` | Install the Code-Airlock git pre-commit hook |
| `--run-pre-commit` | Scan staged git diffs with Code-Airlock (hook entrypoint) |
| `-m, --model` | Override the starting model for this run |
| `-c, --config` | Add an explicit TOML config file (highest file priority) |
| `-v, --verbose` | Enable verbose logging |
| `--dry-run` | Preview changes without mutating files/running shell |
| `--version` | Print version and exit |

`--no-airlock` only disables redaction after you type `yes` at the prompt (or pass `--yes` in a non-interactive script). There is no API key requirement just to start the CLI or read `--help`.

## REPL Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Show all available REPL commands | `/help` |
| `/exit` | Exit REPL and print cost summary | `/exit` |
| `/quit` | Alias for `/exit` | `/quit` |
| `/clear` | Clear conversation history (keep memory) | `/clear` |
| `/cost` | Show cost, cache hit ratio, savings, reasoning tokens, and throughput | `/cost` |
| `/tier` | Show routing mode, Jev signals, and the last decision | `/tier` |
| `/tier auto` | Re-enable Jev auto-routing | `/tier auto` |
| `/tier <name>` | Lock a specific tier | `/tier architect` |
| `/airlock` | Show AirLock mode, vault statistics, and MCP status | `/airlock` |
| `/airlock on\|off` | Enable or disable AirLock for this session | `/airlock off` |
| `/airlock scan <path_or_text>` | Scan a file, directory, or inline text and show a raw-free summary | `/airlock scan .env` |
| `/version` | Show app and Python version | `/version` |
| `/memory` | Show project memory table | `/memory` |
| `/memory set` | Update memory key/value | `/memory set language=python` |
| `/memory detect` | Re-run memory auto-detection | `/memory detect` |
| `/rewind` | List available snapshot steps | `/rewind` |
| `/rewind <step>` | Restore files from a snapshot step | `/rewind 3` |
| `/snapshots` | List turn snapshots and the files they touched | `/snapshots` |
| `/undo` | Restore files to before the last file-modifying turn | `/undo` |
| `/explain <task>` | Preview execution plan + estimated cost | `/explain fix failing tests` |

## Configuration

Axe merges configuration in this order (lowest to highest priority):

1. Bundled defaults: `src/axe/config/default_config.toml` (shipped inside the wheel)
2. User global: `~/.config/axe/config.toml`
3. Workspace local: `<repo_root>/.axe/config.toml` (repo root is the nearest `.git`)
4. Explicit file: `--config <path>` or the `AXE_CONFIG` environment variable
5. Environment variables: `.env`, provider key variables, and `AXE_*` overrides

Nested `AXE_*` overrides use a double underscore:

```bash
AXE_GENERAL__MAX_ITERATIONS=40
AXE_ROUTER__JEV__ENABLED=false
AXE_PROVIDERS__OPENROUTER__BASE_URL=https://openrouter.ai/api/v1
```

Supported provider variables:

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | OpenRouter key |
| `DEEPSEEK_API_KEY` | DeepSeek key (enables direct `deepseek/*` routing) |
| `ZHIPU_API_KEY` | Zhipu key (enables direct `zhipu/*` routing) |
| `MOONSHOT_API_KEY` | Moonshot/Kimi key |
| `FIREWORKS_API_KEY` | Fireworks key |
| `JEV_API_KEY` | Jev decision model key |
| `OLLAMA_HOST` | Ollama base URL (default `http://localhost:11434`) |
| `VLLM_API_KEY` / `VLLM_BASE_URL` | Local vLLM |
| `LLAMACPP_API_KEY` / `LLAMACPP_BASE_URL` | Local llama.cpp |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | Optional legacy OpenAI-compatible endpoint |

A minimal workspace override (`.axe/config.toml`):

```toml
[general]
max_iterations = 40
snapshot_before_edit = true

[providers]
default = "openrouter"

[router.jev]
enabled = true
endpoint = "https://openrouter.ai/api/v1/evaluate"
model = "typesafe/jev-latest"
confidence_threshold = 0.75
risk_gate_threshold = 0.65
ast_compression_threshold = 0.80
timeout_ms = 500

[router.tiers.coder]
models = ["deepseek/deepseek-v4", "qwen/qwen3.8-27b"]
thinking_mode = "low"
max_tokens = 8192

[security.airlock]
enabled = true
default_mcp_enabled = true
mcp_command = "code-airlock"
mcp_args = ["mcp", "serve"]
redact_outbound_context = true
restore_on_file_write = true
guided_remediation_mode = "interactive"
pre_commit_exclude_globs = []  # e.g. ["docs/*", "**/fixtures/*"]
```

Notes:

- Model labels are OpenRouter-style `vendor/model` IDs by default. `deepseek/*`, `zhipu/*`, and `moonshot/*` route directly when the matching key is configured; otherwise they use the default OpenRouter slug. `vllm/*` maps to `local_vllm`, `llamacpp/*` maps to `llamacpp`, and `provider:model` targets any configured provider explicitly.
- `ollama/*` candidates are health-checked against `/api/tags`; unreachable daemons or missing pulled models are skipped with a warning.
- Unlisted cloud models use a conservative fallback rate (never a fabricated $0); local `ollama/*` and `vllm/*` inference is $0.
- Jev consumes `JEV_API_KEY` or the configured provider key (for example `OPENROUTER_API_KEY` when `router.jev.provider = "openrouter"`). If no key is available, or the call exceeds `timeout_ms`, routing falls back to a deterministic heuristic and marks `JevEvaluation.used_fallback = True`.
- Explicit `--tier`/`/tier` locks bypass Jev entirely. Auto mode evaluates the task and then applies the policy matrix; tier decisions expose `jev_eval`, `risk_flagged`, and `use_ast_compression` for Phase 4/5 consumers.
- Airlock placeholders and raw values live only in process memory. They are never written to logs, snapshots, or cost files. Outbound prompts, LLM messages, tool schemas, file reads, grep matches, shell output, and Jev payloads are sanitized before network calls; inbound `file_write`, `file_edit`, and `shell` arguments are restored locally just before execution.
- `security.airlock.guided_remediation_mode` accepts `interactive` (default), `strict_block`, or `warn`. `strict_block` denies unsafe calls and returns a remediation hint to the agent; `warn` allows the action but records the finding.
- The pre-commit gate honors `# airlock: ignore`, `# pragma: allowlist secret`, and `<!-- airlock: ignore -->` on intentional fixture lines. `security.airlock.pre_commit_exclude_globs` skips matching paths (for example `tests/fixtures/*`) while `tests/` stays scanned by default.
- Obvious documentation placeholders (`your-api-key`, `changeme`, `sk-...`) are not reported as secrets, and reserved test emails (`@example.com`, `@example.org`, `@example.net`, `@test.local`, `@localhost`) are ignored by proposed-write/pre-commit checks while still being redacted from outbound model context.
- The external `code-airlock` MCP server is optional. If it is missing or fails to initialize, Axe registers its built-in `airlock_scan` tool and continues with the local cascade scanner.

## Architecture

```text
src/axe/
├── __init__.py              # __version__ = "0.2.0"
├── cli/
│   ├── app.py               # console entrypoint, argument parsing
│   ├── repl.py              # REPL loop, slash commands, one-shot runner
│   ├── security_prompt.py   # AirLock bypass + Guided Security Issue panel
│   └── git_hooks.py         # pre-commit hook installer + staged-diff gate
├── config/
│   ├── default_config.toml  # packaged defaults
│   └── settings.py          # layered loader + Pydantic schema
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
│   ├── provider.py           # LLMResponse, TokenUsage, errors, think parsing
│   ├── jev_decision.py       # single-pass /evaluate client + heuristic fallback
│   ├── router.py             # Jev policy matrix, tier queue, intra-loop policy
│   └── providers/
│       ├── open_compat.py    # OpenRouter/DeepSeek/Zhipu/Fireworks/vLLM/llama.cpp
│       ├── ollama.py         # native Ollama, health, discovery, throughput
│       └── registry.py       # model -> provider resolution and keys
├── security/
│   ├── airlock_middleware.py # MCP-primary + built-in cascade, sanitize/restore
│   ├── guided_fixer.py       # deterministic remediation + tool-call guard
│   └── secret_vault.py       # process-local placeholder <-> raw-value mappings
├── mcp/
│   ├── client.py             # JSON-RPC stdio + SSE/HTTP clients and manager
│   ├── bridge.py             # MCP tools -> native Axe tool adapters
│   └── default_servers.py    # code-airlock boot + built-in airlock_scan fallback
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

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
uv build
```

## Status

Phase 1 is complete: package migration under `src/axe`, `axe` console script, layered config, and Jev/tier/AirLock schemas.

Phase 2 is complete: multi-provider open-weight inference (OpenRouter, DeepSeek, Zhipu, Moonshot, Fireworks, vLLM, llama.cpp, Ollama), cross-provider tier fallback, reasoning/thinking normalization, and KV-cache/throughput cost tracking.

Phase 3 is complete: Jev single-pass decisioning, deterministic risk/confidence policy, intra-loop downshifting and failure escalation, and `/tier` decision inspection.

Phase 4 is complete: bidirectional Code-Airlock sanitization, ephemeral secret vault, native MCP stdio/SSE/HTTP bridge, default `code-airlock` boot, built-in cascade fallback, and `/airlock` inspection/scanning.

Phase 5 is complete: guided security remediation with deterministic env extraction, speculative turn snapshots with `/undo`, AST/symbol context compression, scoped rules and failed-attempt memory, and the `axe --install-hook` / `axe --run-pre-commit` Airlock gate.

## License

MIT (declared in `pyproject.toml`; root `LICENSE` file can be added in a follow-up).

## Author

Sarthak Maurya - [github.com/sarthexe](https://github.com/sarthexe)
