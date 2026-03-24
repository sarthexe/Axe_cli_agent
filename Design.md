# Autonomous CLI Agent — Design Document

## 1. What This Is

A terminal-resident AI agent that operates in a **think → plan → act → observe → retry** loop. It reads your codebase, runs shell commands, writes and edits files, and self-corrects when things break — all from a single terminal session.

**Key differentiator:** Intelligent multi-model routing. The agent automatically picks the cheapest model that can handle each task, escalating to premium models only when needed. A typical coding session costs $0.01–0.05 instead of $2–5 on Claude Code or Codex.

---

## 2. Core Loop (The Brain)

The entire agent is one recursive control loop:

```
┌─────────────────────────────────────────────────────┐
│                   USER PROMPT                        │
└──────────────────────┬───────────────────────────────┘
                       ▼
              ┌────────────────┐
              │     THINK      │  LLM decides intent + plan
              │  (reasoning)   │  "I need to read X, then fix Y"
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │      PLAN      │  Emit a structured action
              │  (tool call)   │  {tool: "shell", cmd: "pytest"}
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │      ACT       │  Execute the action in sandbox
              │  (execution)   │  Run command, write file, etc.
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │    OBSERVE     │  Capture stdout/stderr/exit code
              │  (feedback)    │  Feed result back into context
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │   EVALUATE     │  Did it work? Goal met?
              │  (check)       │  If no → back to THINK
              └───────┬────────┘
                      ▼
                 ┌──────────┐
                 │   DONE   │  Present result to user
                 └──────────┘
```

**Key constraint:** Cap the loop at N iterations (default: 10). If the agent can't solve it in N tries, surface what it tried and ask the user. Infinite loops are the #1 failure mode of naive agents.

---

## 3. Architecture

```
cli-agent/
├── main.py                  # Entry point, REPL / single-shot mode
├── agent/
│   ├── core.py              # The main agent loop (think-plan-act-observe)
│   ├── planner.py           # Prompt construction + LLM call
│   ├── context.py           # Context window manager (token budget)
│   ├── memory.py            # Conversation history + summarization
│   └── snapshots.py         # Session replay / time-travel checkpoints
├── tools/
│   ├── registry.py          # Tool registry (discover, validate, dispatch)
│   ├── shell.py             # Execute shell commands
│   ├── file_read.py         # Read files (with line ranges)
│   ├── file_write.py        # Write / create files
│   ├── file_edit.py         # Surgical str_replace style edits
│   ├── glob_search.py       # Find files by pattern
│   └── grep_search.py       # Search content across files
├── sandbox/
│   ├── executor.py          # Subprocess runner with timeout + resource limits
│   └── permissions.py       # Allowlist / blocklist for dangerous commands
├── llm/
│   ├── provider.py          # Abstract LLM interface
│   ├── router.py            # Smart model router (cheap → expensive escalation)
│   ├── gemini_provider.py   # Google Gemini API (default — free tier)
│   ├── deepseek_provider.py # DeepSeek API (cheap production workhorse)
│   ├── anthropic_provider.py# Claude API (premium fallback)
│   └── openai_compat.py     # OpenAI-compatible wrapper (Ollama, OpenRouter, etc.)
├── cost/
│   ├── tracker.py           # Per-session and cumulative cost tracking
│   ├── pricing.py           # Model pricing table (auto-updated)
│   └── budget.py            # Budget limits and alerts
├── project/
│   ├── memory.py            # Persistent project memory (.agent/memory.json)
│   └── patterns.py          # Error pattern learning DB
├── config/
│   ├── settings.py          # Pydantic settings (env vars, defaults)
│   └── default.toml         # Default config file
└── utils/
    ├── tokens.py            # Token counting (tiktoken)
    ├── diff.py              # Unified diff generation for edits
    └── logger.py            # Structured logging (JSON lines)
```

---

## 4. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **Language** | Python 3.11+ | Async support, rich ecosystem, you live here already. |
| **LLM (default)** | Gemini 2.0 Flash | Free tier: 15 RPM, 1000 req/day. Good tool calling. Zero cost during development. |
| **LLM (production)** | DeepSeek V3.2 | $0.28/$0.42 per 1M tokens. 90% cache discount. Best price-to-quality for code tasks. |
| **LLM (premium)** | Claude Sonnet 4.6 | $3/$15 per 1M tokens. Escalation-only for complex multi-file reasoning. |
| **CLI framework** | `click` + `rich` | Click for arg parsing, Rich for pretty terminal output (spinners, syntax highlighting, panels). |
| **Config** | Pydantic Settings + TOML | Type-safe config, env var overrides, sensible defaults. |
| **Subprocess** | `asyncio.create_subprocess_exec` | Non-blocking command execution with timeout control. |
| **Token counting** | `tiktoken` | Accurate counts for cost tracking and context budget. |
| **HTTP client** | `httpx` (async) | Non-blocking API calls. Connection pooling. Retry with backoff. |
| **Logging** | `structlog` | JSON-lines structured logs. Debug an agent's decisions after the fact. |

**No database. No server. No Docker.** It's a CLI tool. State lives in memory for the session, persists project memory to `.agent/`, and dumps debug logs to `.jsonl`.

---

## 5. Multi-Model Router (The Cost Engine)

This is the core differentiator. Every other CLI agent uses one model for everything. Yours routes intelligently.

### 5.1 Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MODEL ROUTER                            │
│                                                             │
│  Tier 1: FREE (Gemini Flash)         ← 70% of calls        │
│  Simple reads, grep, glob, single-file edits, shell cmds   │
│  Cost: $0.00                                                │
│                                                             │
│  Tier 2: CHEAP (DeepSeek V3.2)       ← 25% of calls        │
│  Multi-file edits, complex debugging, code generation       │
│  Cost: ~$0.001 per call                                     │
│                                                             │
│  Tier 3: PREMIUM (Claude Sonnet)     ← 5% of calls         │
│  Architecture decisions, cross-module refactors, stuck loops│
│  Cost: ~$0.02 per call                                      │
│                                                             │
│  Typical session (30 tool calls): $0.01 – $0.05             │
│  Same session on Claude Code:        $2.00 – $5.00          │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Routing Logic

```python
class ModelRouter:
    def select_model(self, task_context: dict) -> Provider:
        """
        Route to the cheapest model that can handle the task.
        Escalation is based on OBSERVED failure, not predicted difficulty.
        """
        # Always start at Tier 1
        if self.consecutive_failures == 0:
            return self.tier1  # Gemini Flash

        # Escalate to Tier 2 after 2 consecutive failures
        # (malformed tool calls, wrong tool choice, bad edits)
        if self.consecutive_failures >= 2:
            return self.tier2  # DeepSeek

        # Escalate to Tier 3 after 2 more failures on Tier 2
        if self.tier2_failures >= 2:
            return self.tier3  # Claude Sonnet

    def on_success(self):
        """Reset failure counter on any successful tool execution."""
        self.consecutive_failures = 0
        self.tier2_failures = 0

    def on_failure(self, tier: int):
        """Track failures per tier."""
        if tier == 1:
            self.consecutive_failures += 1
        elif tier == 2:
            self.tier2_failures += 1
```

**Transparency:** The agent ALWAYS shows which model it's using:
```
[gemini-flash] Reading src/main.py...
[gemini-flash] Editing src/main.py — replacing DB connection logic...
[gemini-flash] Running pytest... FAILED
[gemini-flash] Retrying fix... FAILED
[↑ deepseek] Escalating — 2 consecutive failures.
[deepseek] Reading error trace more carefully...
[deepseek] The issue is a missing import in src/db.py...
```

### 5.3 Cost Tracking

```python
class CostTracker:
    def __init__(self):
        self.session_cost = 0.0
        self.session_calls = {"gemini": 0, "deepseek": 0, "claude": 0}
        self.cumulative_file = Path.home() / ".cli-agent" / "costs.jsonl"

    def record(self, provider: str, input_tokens: int, output_tokens: int):
        cost = self.calculate(provider, input_tokens, output_tokens)
        self.session_cost += cost
        self.session_calls[provider] += 1

        # Check budget limit
        if self.session_cost > self.budget_limit:
            raise BudgetExceeded(f"Session cost ${self.session_cost:.4f} exceeds limit")

    def summary(self) -> str:
        """Show at end of session or on /cost command."""
        return (
            f"Session cost: ${self.session_cost:.4f}\n"
            f"Calls: {self.session_calls}\n"
            f"Avg cost/call: ${self.session_cost / sum(self.session_calls.values()):.6f}"
        )
```

Displayed at end of every session:
```
────────────────────────────────────────
Session complete.
  Cost:   $0.0034
  Calls:  gemini: 8, deepseek: 2, claude: 0
  Tokens: 12,340 in / 4,210 out
────────────────────────────────────────
```

---

## 6. Tool System Design

Every tool is a Python class that implements one interface:

```python
from dataclasses import dataclass

@dataclass
class ToolResult:
    output: str          # What the LLM sees
    success: bool        # Did the tool succeed?
    metadata: dict       # Extra info (exit_code, file_path, etc.)

class BaseTool:
    name: str            # "shell", "file_read", etc.
    description: str     # For the LLM's system prompt
    parameters: dict     # JSON Schema for the tool's arguments

    def validate(self, args: dict) -> bool:
        """Validate args before execution."""
        ...

    async def execute(self, args: dict) -> ToolResult:
        """Run the tool, return structured result."""
        ...
```

### 6.1 Tool Definitions

**shell** — Execute a command.
```
args: {command: str, timeout_seconds: int = 30, working_dir: str = "."}
returns: {stdout, stderr, exit_code}
```
- Runs via `asyncio.create_subprocess_exec` inside a subprocess.
- Hard timeout (kill after N seconds).
- Blocked commands: `rm -rf /`, `:(){ :|:& };:`, `mkfs`, `dd if=/dev/zero`, etc.
- Captures combined stdout+stderr up to 50KB. Truncates with `[...truncated, showing last 200 lines]`.

**file_read** — Read a file or directory listing.
```
args: {path: str, line_start: int = None, line_end: int = None}
returns: {content (with line numbers), total_lines, file_size}
```
- Returns numbered lines (so the LLM can reference them in edits).
- Caps at ~500 lines per read. If file is larger, force the LLM to use line ranges.

**file_write** — Create or overwrite a file.
```
args: {path: str, content: str, create_dirs: bool = True}
returns: {path, bytes_written}
```

**file_edit** — Surgical find-and-replace (like `str_replace`).
```
args: {path: str, old_str: str, new_str: str}
returns: {path, diff}
```
- `old_str` must appear exactly once in the file. If ambiguous, fail and tell the LLM.
- Returns a unified diff so the LLM can verify the change.

**glob_search** — Find files by pattern.
```
args: {pattern: str, root: str = "."}
returns: {matches: list[str]}
```
- Respects `.gitignore` by default. Skips `node_modules`, `.git`, `__pycache__`, `venv`.

**grep_search** — Search content across files.
```
args: {query: str, path: str = ".", include: str = None}
returns: {matches: list[{file, line_number, content}]}
```

### 6.2 Tool Registry

```python
class ToolRegistry:
    _tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool): ...
    def get(self, name: str) -> BaseTool: ...
    def schema_for_llm(self) -> list[dict]:
        """Return all tool schemas in OpenAI function-calling format."""
        ...
```

The registry auto-generates the tools section of the system prompt. When you add a new tool, the agent automatically knows about it.

---

## 7. Context Window Management

```
┌──────────────────── CONTEXT BUDGET ────────────────────┐
│                                                        │
│  [SYSTEM PROMPT]           ~800 tokens (fixed)         │
│  [TOOL DEFINITIONS]        ~600 tokens (fixed)         │
│  [PROJECT MEMORY]          ~500 tokens (loaded once)   │
│  [CONVERSATION HISTORY]    remaining budget (sliding)   │
│  [CURRENT OBSERVATION]     last tool result             │
│                                                        │
│  BUDGET: model_context_window - max_output_tokens      │
│  Gemini Flash: 1M (but target 32K effective)           │
│  DeepSeek: 128K (target 32K effective)                 │
│  Claude Sonnet: 200K (target 64K effective)            │
└────────────────────────────────────────────────────────┘
```

### Strategy: Sliding Window with Summarization

```python
class ContextManager:
    def __init__(self, max_tokens: int, reserve_output: int = 4096):
        self.budget = max_tokens - reserve_output
        self.system_prompt: str
        self.tool_schemas: str
        self.project_memory: str         # Loaded from .agent/memory.json
        self.history: list[Message]
        self.summaries: list[str]

    def build_messages(self) -> list[dict]:
        """Assemble messages that fit within budget."""
        fixed_cost = count_tokens(self.system_prompt + self.tool_schemas + self.project_memory)
        remaining = self.budget - fixed_cost

        # Take messages from the end (most recent first)
        # When history exceeds budget: summarize oldest N messages into one block
        ...
```

**Rules:**
1. Never send raw files over 500 lines. Always read with line ranges.
2. Truncate tool output to 50KB. Let the LLM re-read specific sections.
3. When history grows past 60% of budget, summarize the oldest third into a single "so far" block.
4. Always keep the last 2 user messages + last 2 tool results intact (no summarization).
5. Use the *effective* context target (32K), not the model's max. Cheaper on tokens, better quality.

---

## 8. LLM Provider Abstraction

All providers implement the same interface. The router sits on top.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str | None              # Text response
    tool_calls: list[ToolCall] | None # Structured tool calls
    usage: dict                       # {prompt_tokens, completion_tokens}
    stop_reason: str                  # "end_turn", "tool_use", etc.
    model: str                        # Which model actually ran
    cost: float                       # Calculated cost for this call

class LLMProvider(ABC):
    name: str                         # "gemini", "deepseek", "claude"
    tier: int                         # 1, 2, or 3

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...

class GeminiProvider(LLMProvider):
    """
    Google Generative AI API.
    - Free tier: 15 RPM, 1000 requests/day
    - Tool calling via function_declarations
    - 1M context window
    """
    name = "gemini"
    tier = 1

class DeepSeekProvider(LLMProvider):
    """
    OpenAI-compatible API at https://api.deepseek.com
    - $0.28/1M input, $0.42/1M output (90% cache discount)
    - Native tool calling
    - 128K context window
    """
    name = "deepseek"
    tier = 2

class AnthropicProvider(LLMProvider):
    """
    Anthropic Messages API.
    - $3/1M input, $15/1M output
    - Best tool calling reliability
    - 200K context window
    """
    name = "claude"
    tier = 3

class OpenAICompatProvider(LLMProvider):
    """
    Generic OpenAI-compatible wrapper.
    Works with: Ollama (local), OpenRouter, Together, Groq, etc.
    Configure via base_url + api_key.
    """
    name = "custom"
    tier = 1  # configurable
```

---

## 9. The System Prompt

```
You are an autonomous coding agent running in a terminal.
You have access to tools to interact with the filesystem and shell.

## Rules
- ALWAYS read a file before editing it. Never guess at contents.
- ALWAYS run the code after making changes to verify it works.
- When a command fails, read the error carefully. Fix the root cause, not symptoms.
- If you're unsure, search the codebase first (grep/glob) before making assumptions.
- Never run destructive commands (rm -rf, drop database, etc.) without user confirmation.
- Explain what you're about to do before doing it.
- After 3 failed attempts at the same fix, stop and ask the user for help.

## Working Style
1. Understand the request fully.
2. Explore the relevant code (read, grep, glob).
3. Plan the changes.
4. Make the changes (edit/write).
5. Verify (run tests, execute code).
6. If broken, read error, fix, re-verify.
7. Report the result.

## Project Context
{loaded from .agent/memory.json if it exists}

## Available Tools
{auto-generated from tool registry}
```

---

## 10. Error Recovery (The Self-Fix Loop)

```python
async def act_and_recover(self, action: ToolCall, max_retries: int = 3) -> ToolResult:
    for attempt in range(max_retries):
        result = await self.execute_tool(action)

        if result.success:
            self.router.on_success()
            # Learn the pattern if this was a fix for a previous error
            if attempt > 0:
                self.patterns.learn(previous_error, action, result)
            return result

        # Feed failure back into context
        self.context.add_observation(
            f"[ATTEMPT {attempt + 1}/{max_retries}] Tool '{action.name}' failed:\n"
            f"{result.output}"
        )

        # Check pattern DB first — skip LLM if we've seen this before
        known_fix = self.patterns.match(result.output)
        if known_fix:
            action = known_fix.to_tool_call()
            continue

        # Notify router of failure (may trigger model escalation)
        self.router.on_failure(self.current_provider.tier)

        # Ask the (potentially upgraded) LLM to diagnose
        provider = self.router.select_model(self.context)
        recovery = await provider.complete(self.context.build_messages())

        if recovery.content and not recovery.tool_calls:
            return ToolResult(output=recovery.content, success=False, metadata={})

        previous_error = result.output
        action = recovery.tool_calls[0]

    return ToolResult(
        output=f"Failed after {max_retries} attempts. Last error:\n{result.output}",
        success=False,
        metadata={"exhausted_retries": True}
    )
```

**What makes this different from other agents:** Two things. First, the error pattern DB catches known errors without burning an API call. Second, when a retry does need the LLM, it doesn't re-ask the same model — it escalates to a smarter one that gets a fresh look at the full context.

---

## 11. Differentiating Features

### 11.1 Session Replay & Time Travel

No agent lets you rewind. When the agent makes 5 changes and something breaks at step 3, you're stuck.

```python
class SnapshotManager:
    """Checkpoint file state before every edit. Enable /rewind N."""

    def __init__(self, snapshot_dir: Path = Path(".agent/snapshots")):
        self.snapshot_dir = snapshot_dir
        self.snapshots: list[Snapshot] = []

    def checkpoint(self, step: int, files_affected: list[str]):
        """Copy affected files to .agent/snapshots/{step}/ before editing."""
        step_dir = self.snapshot_dir / str(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        for f in files_affected:
            shutil.copy2(f, step_dir / Path(f).name)
        self.snapshots.append(Snapshot(step=step, files=files_affected, timestamp=now()))

    def rewind(self, to_step: int):
        """Restore all files to their state at step N."""
        snapshot = self.snapshots[to_step]
        for f in snapshot.files:
            src = self.snapshot_dir / str(to_step) / Path(f).name
            shutil.copy2(src, f)
```

User experience:
```
> /rewind 3
Restored 2 files to state at step 3 (before "fix auth middleware").
Steps 4, 5 undone.
```

### 11.2 Project Memory (Persistent Across Sessions)

Other agents start fresh every time. Yours remembers the project.

```json
// .agent/memory.json — auto-maintained, user-editable
{
    "project": {
        "name": "neosutra",
        "language": "python",
        "framework": "fastapi",
        "test_runner": "pytest",
        "key_files": {
            "config": "config/database.py",
            "entry": "src/main.py",
            "models": "src/models/"
        }
    },
    "learned": [
        "Uses pgvector for embeddings",
        "DB migrations via alembic",
        "Auth tokens are in src/auth/jwt.py"
    ],
    "last_session": {
        "date": "2026-03-23",
        "summary": "Fixed RAG pipeline chunking. Tests passing.",
        "files_touched": ["src/pipeline/chunker.py", "tests/test_chunker.py"]
    }
}
```

On startup, this gets loaded into the system prompt's project context section. The agent knows what it worked on last time without you re-explaining.

**Auto-learn:** After each session, the agent appends new discoveries (detected test runner, key config paths, patterns found) to memory. User can review/edit with `/memory` command.

### 11.3 Error Pattern DB

When the agent fixes an error, log the pattern. Next time, check locally before calling the LLM.

```python
class PatternDB:
    """Local pattern matching — skip LLM calls for known fixes."""

    def __init__(self, db_path: Path = Path(".agent/patterns.json")):
        self.patterns: list[ErrorPattern] = self.load(db_path)

    def match(self, error_output: str) -> ErrorPattern | None:
        """Check if this error matches a previously solved pattern."""
        for pattern in self.patterns:
            if pattern.signature in error_output:
                return pattern
        return None

    def learn(self, error_output: str, fix_applied: str, file_context: str):
        """Record a successful error → fix mapping."""
        signature = self.extract_signature(error_output)
        self.patterns.append(ErrorPattern(
            signature=signature,
            fix=fix_applied,
            context=file_context,
            times_used=0
        ))
```

Common errors (missing imports, typos, config issues) get fixed instantly without an API call. Saves cost AND time.

### 11.4 Explain Mode (Plan Preview)

Every agent just acts. Yours can show the full plan before executing.

```
> /explain refactor the auth module to use JWT

[PLAN — no changes will be made]
1. grep_search: Find all files referencing auth/session
   → Expect: routes using session-based auth
2. file_read: Read src/auth/session.py
   → Understand current implementation
3. file_write: Create src/auth/jwt.py
   → New JWT implementation
4. file_edit: Update src/auth/session.py imports in 3 route files
5. shell: Run pytest tests/test_auth.py
   → Verify nothing broke

Estimated cost: ~$0.003 (8 calls on Gemini Flash)
Proceed? [y/n/edit]
```

---

## 12. Safety & Sandboxing

| Risk | Mitigation |
|---|---|
| `rm -rf /` or destructive commands | Blocklist regex on commands. Require user confirmation for anything matching `rm`, `drop`, `truncate`, `kill`, `mkfs`. |
| Infinite loops | Hard cap on agent iterations (default 10). Hard timeout on subprocesses (default 30s). |
| Token blowout / cost runaway | Cost tracker with per-session budget limit (default: $0.50). Alert at 80%. Hard stop at limit. |
| Writes to wrong files | Log all file writes with full diffs. Support `--dry-run` mode. Session snapshots enable `/rewind`. |
| Runaway resource usage | `ulimit` on subprocess (max memory, max file size, max CPU time). |
| Prompt injection from file contents | Keep user instructions in system prompt. Treat file contents as untrusted data in user messages. |
| API key leakage | Keys only in env vars or `~/.cli-agent/config.toml` (chmod 600). Never logged, never in context. |

**Confirmation mode** (default for sensitive ops):
```
Agent wants to run: rm -rf ./build/
[y/n/edit]>
```

---

## 13. Config Schema

```toml
# ~/.cli-agent/config.toml

[llm]
default_provider = "gemini"          # "gemini" | "deepseek" | "anthropic" | "custom"
fallback_provider = "deepseek"       # Tier 2 escalation
premium_provider = "anthropic"       # Tier 3 escalation

[llm.gemini]
model = "gemini-2.0-flash"
# api_key via GEMINI_API_KEY env var
temperature = 0.0
context_window = 1000000

[llm.deepseek]
model = "deepseek-chat"
base_url = "https://api.deepseek.com"
# api_key via DEEPSEEK_API_KEY env var
temperature = 0.0
context_window = 131072

[llm.anthropic]
model = "claude-sonnet-4-20250514"
# api_key via ANTHROPIC_API_KEY env var
temperature = 0.0
context_window = 200000

[llm.custom]
# OpenAI-compatible endpoint (Ollama, OpenRouter, Groq, Together, etc.)
model = "qwen3.5:4b"
base_url = "http://localhost:11434/v1"
api_key = "ollama"
temperature = 0.0
context_window = 262144

[router]
escalate_after_failures = 2          # Consecutive failures before tier upgrade
reset_on_success = true              # Reset failure counter on any success
show_model_in_output = true          # [gemini-flash] prefix on every action

[cost]
session_budget = 0.50                # Hard stop at $0.50 per session
alert_at_percent = 80                # Warn at 80% of budget
log_file = "~/.cli-agent/costs.jsonl"

[agent]
max_iterations = 10
max_retries_per_tool = 3
confirmation_required = ["rm", "drop", "kill", "truncate"]

[sandbox]
command_timeout_seconds = 30
max_output_bytes = 51200             # 50KB
blocked_commands = ["rm -rf /", "mkfs", "dd if=/dev/zero"]

[context]
max_file_lines = 500
summarize_threshold = 0.6
keep_recent_messages = 4
effective_context_target = 32768     # Target 32K regardless of model max

[features]
project_memory = true                # Persistent .agent/memory.json
error_patterns = true                # Error pattern learning DB
session_snapshots = true             # Time-travel checkpoints
explain_mode = false                 # Default off, enable with /explain
```

---

## 14. Build Phases

### Phase 1 — Skeleton + Gemini (Day 1-2)
- [ ] Project scaffolding, config loading, CLI entry point (`click`)
- [ ] LLM provider abstraction + Gemini provider (free tier)
- [ ] Single tool: `shell` (execute commands, capture output)
- [ ] Basic agent loop: prompt → LLM → tool call → execute → show result
- [ ] Cost tracker (counting tokens + displaying summary)
- [ ] **Milestone:** `agent "list all python files"` works on free Gemini API

### Phase 2 — File Operations (Day 3-4)
- [ ] `file_read`, `file_write`, `file_edit` tools
- [ ] `glob_search`, `grep_search` tools
- [ ] Tool registry with auto-schema generation
- [ ] **Milestone:** `agent "read main.py and add error handling to the DB connection"` works

### Phase 3 — Multi-Model Router (Day 5-6)
- [ ] DeepSeek provider
- [ ] Anthropic provider
- [ ] Router with failure-based escalation logic
- [ ] Model name in terminal output `[gemini-flash]` prefix
- [ ] **Milestone:** Agent auto-escalates to DeepSeek when Gemini fails a task

### Phase 4 — Context & Memory (Day 7-8)
- [ ] Token counting with tiktoken
- [ ] Context window manager with sliding window + summarization
- [ ] Project memory (`.agent/memory.json`) — load on startup, auto-learn on exit
- [ ] Multi-turn REPL mode (persistent session)
- [ ] **Milestone:** Agent remembers project structure across sessions

### Phase 5 — Self-Repair + Snapshots (Day 9-10)
- [ ] Error recovery loop (retry with escalation)
- [ ] Session snapshots before every edit
- [ ] `/rewind N` command
- [ ] Auto-run after edits (detect test runner, run it)
- [ ] **Milestone:** `agent "fix the failing tests"` works, and `/rewind 2` restores old state

### Phase 6 — Polish & Ship (Day 11-12)
- [ ] Error pattern DB (learn from fixes, skip LLM for known patterns)
- [ ] Explain mode (`/explain` shows plan without executing)
- [ ] `--dry-run` mode
- [ ] Command blocklist + confirmation prompts
- [ ] Rich terminal output (syntax-highlighted code, diff panels, spinners, cost display)
- [ ] OpenAI-compatible provider (plug in Ollama, Groq, OpenRouter)
- [ ] **Milestone:** Ship it. README headline: "costs 100x less than Claude Code."

---

## 15. Critical Design Decisions

**1. Tool calls, not free-form parsing.**
Don't regex-parse the LLM's prose for commands. Use native function-calling format. Gemini, DeepSeek, and Claude all support it. Structured JSON in, structured JSON out. This eliminates 80% of "agent did something weird" bugs.

**2. Edit by replacement, not by rewrite.**
The `file_edit` tool does `str_replace` (find exact string → replace with new string), NOT "rewrite the whole file." This prevents the LLM from accidentally dropping code, losing imports, or hallucinating functions.

**3. Observe before act.**
Hard-code into the system prompt: "ALWAYS read before edit." The #1 agent failure is editing a file the LLM has never seen.

**4. Cheap-first, smart-escalation.**
Start every request on the free/cheapest model. Only escalate when the model demonstrably fails — not when the task "looks hard." This is a core philosophical difference from tools that always use the most expensive model.

**5. Log everything.**
Every LLM call (with model name, tokens, cost), every tool execution, every retry — write it to a `.jsonl` file. When the agent does something stupid, you need the full trace.

**6. Cost is a first-class feature.**
Display cost per session. Let users set budgets. Show which model handled each step. Make cost transparency the product's identity.

**7. Provider-agnostic from day one.**
All providers implement the same interface. Adding a new one (Groq, Together, Mistral, local Ollama) is one file. Users can bring whatever API keys they have. No vendor lock-in.
