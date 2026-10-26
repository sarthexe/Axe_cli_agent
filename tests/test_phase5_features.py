"""Phase 5 tests: guided remediation, snapshots, AST context, memory, hooks."""

from __future__ import annotations

import json
import io
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from axe.agent.context import ContextManager, compress_code_to_ast_skeleton
from axe.agent.core import Agent
from axe.agent.snapshots import SnapshotManager
from axe.agent.tokens import count_tokens
from axe.cli.git_hooks import (
    StagedFinding,
    _print_block_report,
    install_pre_commit_hook,
    run_pre_commit_check,
)
from axe.cli.repl import handle_repl_command
from axe.cli.security_prompt import prompt_guided_security_issue
from axe.config.settings import Settings, load_settings
from axe.llm.provider import LLMProvider, LLMResponse, ToolCall
from axe.project.memory import ProjectMemory
from axe.security.airlock_middleware import AirlockMiddleware
from axe.security.guided_fixer import GuidedSecurityFixer
from axe.tools.file_read import FileReadTool
from axe.tools.file_write import FileWriteTool
from axe.tools.registry import ToolRegistry

FAKE_KEY = "sk-" + "proj-" + "abcdefghijklmnopqrstuvwxyz0123456789"
BODY_SENTINEL = "MAGIC_BODY_SENTINEL_42"


def _settings() -> Settings:
    return load_settings(env={}, include_user=False, include_workspace=False)


class ScriptedProvider(LLMProvider):
    name = "fake"
    model = "fake/model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.last_decision: Any = None

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        return self.responses.pop(0)


def _source_with_large_bodies() -> str:
    body = "\n".join(
        f"        total += value * self.factor + {index}  # {BODY_SENTINEL}"
        for index in range(40)
    )
    return (
        '"""Calculator module used by the AST compression tests."""\n'
        "import os\n"
        "from typing import Iterable\n"
        "\n"
        "\n"
        "class Calculator:\n"
        '    """Accumulates weighted values."""\n'
        "\n"
        "    factor: int = 2\n"
        "\n"
        "    def __init__(self, factor: int = 2) -> None:\n"
        "        self.factor = factor\n"
        "\n"
        "    def compute(self, values: Iterable[int]) -> int:\n"
        '        """Return the weighted sum."""\n'
        "        total = 0\n"
        f"{body}\n"
        "        return total\n"
        "\n"
        "\n"
        "async def fetch(url: str, timeout: float = 5.0) -> str:\n"
        '    """Fetch a URL."""\n'
        "    cleaned = url.strip()\n"
        f"    marker = '{BODY_SENTINEL}_fetch'\n"
        "    return cleaned + marker\n"
    )


def _rendered_read(path: str, source: str) -> str:
    lines = source.splitlines()
    body = "\n".join(f"{index:4d} | {line}" for index, line in enumerate(lines, 1))
    return f"--- {path} (lines 1-{len(lines)} of {len(lines)}) ---\n{body}"


# ---------------------------------------------------------------------------
# Guided security remediation
# ---------------------------------------------------------------------------


def test_guided_fixer_detects_secrets_and_unsafe_patterns(tmp_path: Path) -> None:
    settings = _settings()
    fixer = GuidedSecurityFixer(
        settings,
        AirlockMiddleware(settings),
        repo_root=tmp_path,
        console=Console(quiet=True),
    )
    content = (
        f'API_KEY = "{FAKE_KEY}"\n'
        "import yaml\n"
        "config = yaml.load(stream)\n"
        'query = f"SELECT * FROM users WHERE id={user_id}"\n'
        "subprocess.run(cmd, shell=True)\n"
    )

    findings = fixer.scan_proposed_write("service.py", content)
    categories = {finding.category for finding in findings}
    patterns = {finding.unsafe_pattern for finding in findings}

    assert "api_key" in categories
    assert "unsafe_yaml_load" in patterns
    assert "sql_interpolation" in patterns
    assert "unsafe_shell_true" in patterns
    assert all(FAKE_KEY not in json.dumps(f.public_dict()) for f in findings)


def test_extract_to_env_rewrites_code_and_updates_env_files(tmp_path: Path) -> None:
    settings = _settings()
    fixer = GuidedSecurityFixer(
        settings,
        AirlockMiddleware(settings),
        repo_root=tmp_path,
        console=Console(quiet=True),
    )
    source = f'API_KEY = "{FAKE_KEY}"\n\nprint(API_KEY)\n'

    findings = fixer.scan_proposed_write("app.py", source)
    plan = fixer.generate_remediation("app.py", source, findings)
    result = plan.apply_auxiliary()

    assert 'API_KEY = os.environ.get("API_KEY")' in plan.fixed_content
    assert "import os" in plan.fixed_content
    assert FAKE_KEY not in plan.fixed_content
    assert "API_KEY" in result.env_keys

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    example_text = (tmp_path / ".env.example").read_text(encoding="utf-8")
    gitignore_text = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert f"API_KEY={FAKE_KEY}" in env_text
    assert FAKE_KEY not in example_text
    assert "API_KEY=" in example_text
    assert ".env" in gitignore_text.splitlines()


def test_strict_block_mode_blocks_write_and_returns_remediation(tmp_path: Path) -> None:
    settings = _settings()
    settings.security.airlock.guided_remediation_mode = "strict_block"
    fixer = GuidedSecurityFixer(
        settings,
        AirlockMiddleware(settings),
        repo_root=tmp_path,
        console=Console(quiet=True),
    )

    guard = fixer.guard_tool_call(
        "file_write",
        {"path": str(tmp_path / "app.py"), "content": f'KEY = "{FAKE_KEY}"\n'},
    )

    assert guard.allowed is False
    assert "BLOCKED" in guard.message
    assert "environment variable" in guard.message
    assert FAKE_KEY not in guard.message
    assert not (tmp_path / "app.py").exists()

    # Retrying the same unsafe content must not be treated as pre-existing.
    retry = fixer.guard_tool_call(
        "file_write",
        {"path": str(tmp_path / "app.py"), "content": f'KEY = "{FAKE_KEY}"\n'},
    )
    assert retry.allowed is False


def test_interactive_auto_remediation_updates_arguments(tmp_path: Path) -> None:
    settings = _settings()
    fixer = GuidedSecurityFixer(
        settings,
        AirlockMiddleware(settings),
        repo_root=tmp_path,
        console=Console(quiet=True),
        assume_yes=True,
    )

    guard = fixer.guard_tool_call(
        "file_write",
        {"path": str(tmp_path / "app.py"), "content": f'KEY = "{FAKE_KEY}"\n'},
    )

    assert guard.allowed is True
    assert guard.choice == "auto"
    assert FAKE_KEY not in guard.arguments["content"]
    assert "os.environ.get" in guard.arguments["content"]
    assert (tmp_path / ".env").exists()


def test_documentation_placeholders_and_test_emails_are_not_flagged(
    tmp_path: Path,
) -> None:
    settings = _settings()
    fixer = GuidedSecurityFixer(
        settings,
        AirlockMiddleware(settings),
        repo_root=tmp_path,
        console=Console(quiet=True),
    )
    content = (
        'export OPENROUTER_API_KEY="your-openrouter-api-key"\n'
        "DEEPSEEK_API_KEY=changeme\n"
        "TOKEN=sk-...\n"
        'contact = "dev@example.com"\n'
    )

    findings = fixer.scan_proposed_write("README.md", content)

    assert not [
        finding
        for finding in findings
        if finding.category in {"env_secret", "email"}
    ]
    # Outbound context still redacts reserved test emails.
    result = fixer.airlock.sanitize_outbound(
        "dev@example.com",
        source="outbound_test",
    )
    assert "dev@example.com" not in result.sanitized_text


def test_inline_ignore_markers_suppress_proposed_write_findings(
    tmp_path: Path,
) -> None:
    settings = _settings()
    fixer = GuidedSecurityFixer(
        settings,
        AirlockMiddleware(settings),
        repo_root=tmp_path,
        console=Console(quiet=True),
    )

    for marker in (
        "# airlock: ignore",
        "# pragma: allowlist secret",
        "<!-- airlock: ignore -->",
    ):
        content = f'API_KEY = "{FAKE_KEY}"  {marker}\n'
        assert fixer.scan_proposed_write("fixture.py", content) == []


def test_guided_prompt_renders_literal_option_brackets() -> None:
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        width=120,
        force_terminal=False,
        color_system=None,
    )
    finding = SimpleNamespace(
        category="api_key",
        severity="high",
        line=1,
        message="Hardcoded key.",
        masked_snippet="KEY=<AIRLOCK_SECRET_001>",
    )

    decision = prompt_guided_security_issue(
        console,
        [finding],
        path="app.py",
        input_fn=lambda _prompt: "b",
    )
    output = buffer.getvalue()

    assert decision.choice == "block"
    for option in ("[a]", "[e]", "[i]", "[b]"):
        assert option in output


def test_pre_commit_report_renders_literal_extract_option() -> None:
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        width=120,
        force_terminal=False,
        color_system=None,
    )

    _print_block_report(
        console,
        [
            StagedFinding(
                path="app.py",
                line=3,
                category="api_key",
                detector="builtin-airlock-cascade",
                severity="high",
                masked_snippet="KEY=<AIRLOCK_SECRET_001>",
            )
        ],
    )

    assert "[e]" in buffer.getvalue()


# ---------------------------------------------------------------------------
# Speculative snapshots and /undo
# ---------------------------------------------------------------------------


def test_snapshot_rollback_restores_original_and_removes_new_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "app.py"
    target.write_text("original\n", encoding="utf-8")
    created = tmp_path / "new.py"
    manager = SnapshotManager(snapshot_dir=tmp_path / ".snapshots", cwd=tmp_path)

    turn = manager.begin_turn()
    assert manager.checkpoint(turn, str(target))
    target.write_text("modified\n", encoding="utf-8")
    turn_two = manager.begin_turn()
    assert manager.checkpoint(turn_two, str(created))
    created.write_text("created\n", encoding="utf-8")

    assert [snapshot.turn_id for snapshot in manager.list_snapshots()] == [1, 2]

    report = manager.rollback_last()
    assert report.success is True
    assert not created.exists()

    report = manager.rollback_last()
    assert report.success is True
    assert target.read_text(encoding="utf-8") == "original\n"
    assert report.turn_id == 1


def test_agent_file_write_creates_snapshot_and_undo_restores(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("before\n", encoding="utf-8")
    settings = _settings()
    snapshots = SnapshotManager(
        snapshot_dir=tmp_path / ".snapshots",
        console=Console(quiet=True),
        cwd=tmp_path,
    )
    provider = ScriptedProvider(
        [
            LLMResponse(
                content="",
                model="fake/model",
                provider="fake",
                tool_calls=[
                    ToolCall(
                        name="file_write",
                        arguments={"path": str(target), "content": "after\n"},
                        id="call_1",
                    )
                ],
            ),
            LLMResponse(content="done", model="fake/model", provider="fake"),
        ]
    )
    registry = ToolRegistry()
    registry.register(FileWriteTool())
    agent = Agent(
        provider,
        registry,
        settings,
        console=Console(quiet=True),
        snapshot_manager=snapshots,
    )

    assert agent.run("update the file") == "done"
    assert target.read_text(encoding="utf-8") == "after\n"
    assert len(snapshots.list_snapshots()) == 1

    session = SimpleNamespace(snapshots=snapshots, settings=settings)
    assert handle_repl_command("/undo", session) is True
    assert target.read_text(encoding="utf-8") == "before\n"


def test_speculative_check_rolls_back_on_failed_verification(
    tmp_path: Path,
) -> None:
    target = tmp_path / "app.py"
    target.write_text("stable\n", encoding="utf-8")
    manager = SnapshotManager(
        snapshot_dir=tmp_path / ".snapshots",
        console=Console(quiet=True),
        cwd=tmp_path,
    )
    turn = manager.begin_turn()
    assert manager.checkpoint(turn, str(target))
    target.write_text("broken\n", encoding="utf-8")

    command = f'"{sys.executable}" -c "import sys; sys.exit(3)"'
    result = manager.run_speculative_check(
        command,
        rollback_on_failure=True,
        cwd=tmp_path,
    )

    assert result.returncode == 3
    assert result.rolled_back is True
    assert target.read_text(encoding="utf-8") == "stable\n"


# ---------------------------------------------------------------------------
# AST context compression
# ---------------------------------------------------------------------------


def test_ast_skeleton_preserves_symbols_and_cuts_tokens() -> None:
    source = _source_with_large_bodies()
    skeleton = compress_code_to_ast_skeleton("calculator.py", source)

    assert "import os" in skeleton
    assert "from typing import Iterable" in skeleton
    assert "class Calculator:" in skeleton
    assert "factor: int = 2" in skeleton
    assert "def __init__(self, factor: int = 2) -> None:" in skeleton
    assert "def compute(self, values: Iterable[int]) -> int:" in skeleton
    assert "async def fetch(url: str, timeout: float = 5.0) -> str:" in skeleton
    assert "# lines" in skeleton
    assert BODY_SENTINEL not in skeleton
    assert "total += value" not in skeleton
    assert "return cleaned + marker" not in skeleton

    before = count_tokens(source)
    after = count_tokens(skeleton)
    assert after < before * 0.5


def test_context_prepare_compresses_background_files_when_jev_requests_it() -> None:
    manager = ContextManager(_settings().context, console=Console(quiet=True))
    source = _source_with_large_bodies()
    path = "/repo/calculator.py"
    rendered = _rendered_read(path, source)
    manager.register_read_output(path, rendered)
    messages = [
        {"role": "system", "content": "You are Axe."},
        {"role": "tool", "tool_call_id": "1", "content": rendered},
    ]

    prepared = manager.prepare(messages, use_ast_compression=True)

    assert "symbol skeleton" in prepared[1]["content"]
    assert BODY_SENTINEL not in prepared[1]["content"]
    assert manager.tokens_saved_by_ast > 0
    assert manager.ast_compressions == 1
    # The original history is not mutated.
    assert rendered == messages[1]["content"]


def test_context_keeps_actively_edited_files_in_full_fidelity() -> None:
    manager = ContextManager(_settings().context, console=Console(quiet=True))
    source = _source_with_large_bodies()
    path = "/repo/calculator.py"
    rendered = _rendered_read(path, source)
    manager.register_read_output(path, rendered)
    manager.set_active_files({path})
    messages = [
        {"role": "system", "content": "You are Axe."},
        {"role": "tool", "tool_call_id": "1", "content": rendered},
    ]

    prepared = manager.prepare(messages, use_ast_compression=True)

    assert prepared[1]["content"] == rendered
    assert manager.tokens_saved_by_ast == 0


def test_jev_compact_decision_compresses_read_results_in_agent(tmp_path: Path) -> None:
    source = _source_with_large_bodies()
    target = tmp_path / "calculator.py"
    target.write_text(source, encoding="utf-8")
    settings = _settings()

    provider = ScriptedProvider(
        [
            LLMResponse(
                content="",
                model="fake/model",
                provider="fake",
                tool_calls=[
                    ToolCall(
                        name="file_read",
                        arguments={"path": str(target)},
                        id="call_1",
                    )
                ],
            ),
            LLMResponse(content="done", model="fake/model", provider="fake"),
        ]
    )
    provider.last_decision = SimpleNamespace(
        use_ast_compression=True,
        risk_flagged=False,
    )
    manager = ContextManager(settings.context, console=Console(quiet=True))
    registry = ToolRegistry()
    registry.register(FileReadTool(settings.context))
    agent = Agent(
        provider,
        registry,
        settings,
        console=Console(quiet=True),
        context_manager=manager,
    )

    assert agent.run("inspect the calculator") == "done"

    followup = json.dumps(provider.calls[1]["messages"])
    assert "symbol skeleton" in followup
    assert BODY_SENTINEL not in followup
    assert manager.tokens_saved_by_ast > 0


# ---------------------------------------------------------------------------
# Scoped failure memory
# ---------------------------------------------------------------------------


def test_memory_injects_scoped_rules_and_failure_avoidance(tmp_path: Path) -> None:
    memory = ProjectMemory(memory_dir=tmp_path / ".axe", console=Console(quiet=True))
    memory.add_rule("*.py", "Use type annotations on public functions.")
    memory.add_rule("security", "Never log credentials.")
    memory.record_failed_attempt(
        tool_name="shell",
        target_file="src/app.py",
        error_summary="AssertionError: expected 2",
        avoid_hint="Inspect the failing assertion instead of rerunning pytest blindly.",
    )

    context = memory.to_turn_context(paths=["src/app.py"])

    assert "<ScopedRules>" in context
    assert "Use type annotations" in context
    assert "Never log credentials" not in context
    assert "<AvoidRepeatingFailures>" in context
    assert "AssertionError" in context
    assert "Avoid:" in context

    unrelated = memory.to_turn_context(paths=["docs/readme.md"])
    assert "<AvoidRepeatingFailures>" not in unrelated


def test_memory_is_backward_compatible_and_ring_buffer_is_bounded(
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / ".axe"
    memory_dir.mkdir()
    (memory_dir / "memory.json").write_text(
        json.dumps({"language": "python", "key_files": ["main.py"]}),
        encoding="utf-8",
    )
    memory = ProjectMemory(memory_dir=memory_dir, console=Console(quiet=True))
    memory.load()

    assert memory.data["language"] == "python"
    for index in range(15):
        memory.record_failed_attempt(
            tool_name="shell",
            target_file=f"src/file_{index}.py",
            error_summary=f"failure {index}",
            avoid_hint="change approach",
        )

    assert len(memory.data["failed_attempts"]) == 10
    assert memory.data["failed_attempts"][-1]["target_file"] == "src/file_14.py"


def test_agent_injects_failure_memory_on_next_turn(tmp_path: Path) -> None:
    settings = _settings()
    memory = ProjectMemory(memory_dir=tmp_path / ".axe", console=Console(quiet=True))
    provider = ScriptedProvider([])
    agent = Agent(
        provider,
        ToolRegistry(),
        settings,
        console=Console(quiet=True),
        project_memory=memory,
    )
    target = tmp_path / "app.py"

    agent._record_tool_outcome(
        "file_write",
        {"path": str(target)},
        "Error: disk is read-only",
        "error",
    )
    prepared = agent._inject_failure_memory(
        [{"role": "system", "content": "system"}]
    )

    injected = "\n".join(str(message.get("content", "")) for message in prepared)
    assert "<AvoidRepeatingFailures>" in injected
    assert "disk is read-only" in injected


# ---------------------------------------------------------------------------
# Pre-commit installer and staged-diff gate
# ---------------------------------------------------------------------------


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Axe Test"],
        check=True,
        capture_output=True,
    )


def test_install_pre_commit_hook_preserves_existing_hook(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    existing = repo / ".git" / "hooks" / "pre-commit"
    existing.write_text("#!/bin/sh\necho legacy\n", encoding="utf-8")

    hook = install_pre_commit_hook(repo, console=Console(quiet=True))

    assert hook == repo / ".git" / "hooks" / "pre-commit"
    script = hook.read_text(encoding="utf-8")
    assert "axe --run-pre-commit" in script
    assert "Installed by Axe" in script
    backup = repo / ".git" / "hooks" / "pre-commit.axe.bak"
    assert backup.read_text(encoding="utf-8") == "#!/bin/sh\necho legacy\n"
    if os.name != "nt":
        assert hook.stat().st_mode & stat.S_IXUSR


def test_pre_commit_blocks_staged_secret_and_allows_clean_diff(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    clean_buffer = io.StringIO()
    console = Console(file=clean_buffer, record=True, width=120)

    clean = repo / "clean.py"
    clean.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "clean.py"], check=True)
    assert run_pre_commit_check(repo, _settings(), console=console) == 0

    # Replace the clean staged file with a secret-bearing one.
    dirty = repo / "config.py"
    dirty.write_text(f'API_KEY = "{FAKE_KEY}"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "config.py"], check=True)
    dirty_console = Console(file=io.StringIO(), record=True, width=120)
    code = run_pre_commit_check(repo, _settings(), console=dirty_console)

    output = dirty_console.file.getvalue()
    assert code == 1
    assert "config.py" in output
    assert "api_key" in output
    assert FAKE_KEY not in output
    assert "Commit blocked by Code-Airlock" in output


def test_pre_commit_honors_inline_ignore_and_exclude_globs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    _init_repo(repo)
    console = Console(quiet=True)

    ignored = repo / "tests" / "test_fixture.py"
    ignored.write_text(
        f'TOKEN = "{FAKE_KEY}"  # airlock: ignore\n',
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "tests/test_fixture.py"],
        check=True,
    )
    assert run_pre_commit_check(repo, _settings(), console=console) == 0

    excluded = repo / "tests" / "test_other_fixture.py"
    excluded.write_text(f'TOKEN = "{FAKE_KEY}"\n', encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "add", "tests/test_other_fixture.py"],
        check=True,
    )

    settings = _settings()
    settings.security.airlock.pre_commit_exclude_globs = ["tests/*"]
    assert run_pre_commit_check(repo, settings, console=console) == 0

    # The same unannotated fixture is blocked without the exclude glob.
    assert run_pre_commit_check(repo, _settings(), console=console) == 1

