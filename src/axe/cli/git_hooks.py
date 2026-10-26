"""Git pre-commit hook installation and staged-diff security gate.

The hook runs ``axe --run-pre-commit`` before a commit leaves the machine. The
gate inspects only staged added lines, scans them with Code-Airlock and the
guided fixer, and prints a raw-value-free report when it blocks the commit.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from axe.config.settings import Settings
from axe.security.airlock_middleware import AirlockMiddleware, SecurityFinding
from axe.security.guided_fixer import GuidedSecurityFixer

_HOOK_MARKER = "# Installed by Axe"
_SECRET_CATEGORIES = {
    "api_key",
    "jwt",
    "private_key",
    "aws_key",
    "db_url",
    "env_secret",
    "custom",
}
_PII_CATEGORIES = {"email", "phone", "ip_address"}
_BLOCKING_SEVERITIES = {"critical"}
_BLOCKING_CATEGORIES = _SECRET_CATEGORIES | _PII_CATEGORIES

_HOOK_SCRIPT = """#!/bin/sh
# Installed by Axe - staged-diff security gate (Code-Airlock).
# The previous hook, if any, is preserved as pre-commit.axe.bak and chained.
HOOK_DIR=$(dirname "$0")
if [ -x "$HOOK_DIR/pre-commit.axe.bak" ]; then
    "$HOOK_DIR/pre-commit.axe.bak" "$@" || exit $?
fi
if command -v axe >/dev/null 2>&1; then
    exec axe --run-pre-commit
fi
exec python -m axe.cli.app --run-pre-commit
"""


@dataclass(slots=True)
class StagedFinding:
    """A display-safe finding attached to a staged file/line."""

    path: str
    line: int
    category: str
    detector: str
    severity: str
    message: str = ""
    masked_snippet: str = ""


@dataclass(slots=True)
class StagedScanReport:
    """Aggregate result of scanning the staged diff."""

    findings: list[StagedFinding] = field(default_factory=list)
    scanned_files: int = 0

    @property
    def blocking(self) -> list[StagedFinding]:
        return [finding for finding in self.findings if _is_blocking(finding)]


def _resolve_repo_root(repo_root: Path | None) -> Path:
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    root = root.expanduser()
    try:
        root = root.resolve()
    except OSError:
        root = Path.cwd()
    if (root / ".git").exists():
        return root
    for candidate in root.parents:
        if (candidate / ".git").exists():
            return candidate
    return root


def install_pre_commit_hook(
    repo_root: Path | None = None,
    console: Console | None = None,
) -> Path:
    """Install ``.git/hooks/pre-commit`` and return its path.

    An existing hook is preserved as ``pre-commit.axe.bak`` and chained before
    the Axe gate so repository-specific checks keep running.
    """
    root = _resolve_repo_root(repo_root)
    git_dir = root / ".git"
    if not git_dir.exists():
        raise FileNotFoundError(f"No .git directory found at {root}")

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "pre-commit"
    backup_path = hooks_dir / "pre-commit.axe.bak"

    if hook_path.exists():
        existing = ""
        try:
            existing = hook_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            existing = ""
        if _HOOK_MARKER not in existing and not backup_path.exists():
            try:
                shutil.copy2(hook_path, backup_path)
            except OSError:
                pass

    hook_path.write_text(_HOOK_SCRIPT, encoding="utf-8", newline="\n")
    try:
        os.chmod(hook_path, 0o755)
    except OSError:  # pragma: no cover - Windows filesystems may ignore mode
        pass

    if console is not None:
        console.print(
            Panel(
                f"Installed pre-commit gate at [cyan]{escape(str(hook_path))}[/cyan].\n"
                "It scans staged added lines with Code-Airlock and blocks "
                "secret/PII/critical findings.\n"
                "Run [bold]axe /airlock scan <path>[/bold] for a manual audit.",
                title="axe --install-hook",
                border_style="cyan",
            )
        )
    return hook_path


def run_pre_commit_check(
    repo_root: Path | None = None,
    settings: Settings | None = None,
    console: Console | None = None,
) -> int:
    """Scan staged added lines and return a process exit code."""
    root = _resolve_repo_root(repo_root)
    output = console or Console()
    resolved_settings = settings or Settings()

    airlock = AirlockMiddleware(resolved_settings, console=output)
    fixer = GuidedSecurityFixer(
        resolved_settings,
        airlock,
        repo_root=root,
        console=output,
        assume_yes=True,
    )

    added_lines = _staged_added_lines(root, console=output)
    if added_lines is None:
        return 0
    if not added_lines:
        output.print("[dim]Axe pre-commit: no staged changes to scan.[/dim]")
        return 0

    findings: list[StagedFinding] = []
    exclude_globs = list(
        resolved_settings.security.airlock.pre_commit_exclude_globs
    )
    for path, lines in added_lines.items():
        if _path_excluded(path, exclude_globs):
            continue
        text = "\n".join(line for _, line in lines)
        if not text.strip():
            continue
        mapped = [line_number for line_number, _ in lines]
        for finding in fixer.scan_proposed_write(path, text):
            line = _map_line(finding, mapped)
            findings.append(
                StagedFinding(
                    path=path,
                    line=line,
                    category=finding.category,
                    detector=finding.detector_source,
                    severity=finding.severity,
                    message=finding.message,
                    masked_snippet=_safe_snippet(finding, airlock),
                )
            )

    blocking = [finding for finding in findings if _is_blocking(finding)]
    if not blocking:
        output.print(
            f"[success]Axe pre-commit: {len(added_lines)} staged file(s) clean.[/success]"
        )
        return 0

    _print_block_report(output, blocking)
    return 1


def _path_excluded(path: str, exclude_globs: list[str]) -> bool:
    """Return ``True`` when a staged path matches an Airlock exclude glob."""
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    for pattern in exclude_globs:
        glob = str(pattern).replace("\\", "/").strip()
        if not glob:
            continue
        if fnmatch.fnmatch(normalized, glob) or fnmatch.fnmatch(name, glob):
            return True
    return False


def _staged_added_lines(
    root: Path,
    *,
    console: Console | None = None,
) -> dict[str, list[tuple[int, str]]] | None:
    """Return ``{path: [(line_number, added_line), ...]}`` for staged files."""
    try:
        names = subprocess.run(
            ["git", "-C", str(root), "diff", "--cached", "--name-only", "-z"],
            capture_output=True,
            timeout=15,
            check=False,
        )
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--cached", "-U0", "--no-color"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        if console is not None:
            console.print("[error]Axe pre-commit: unable to run git.[/error]")
        return None

    if names.returncode != 0 or diff.returncode != 0:
        if console is not None:
            console.print(
                "[error]Axe pre-commit: git diff failed (not a repository?).[/error]"
            )
        return None

    staged = [
        item.decode("utf-8", errors="replace").replace("\\", "/")
        for item in names.stdout.split(b"\0")
        if item
    ]
    if not staged:
        return {}

    files: dict[str, list[tuple[int, str]]] = {}
    current: str | None = None
    new_line = 0
    for raw_line in (diff.stdout or "").splitlines():
        if raw_line.startswith("+++ b/"):
            current = raw_line[6:]
            if current not in files:
                files[current] = []
            continue
        if raw_line.startswith("+++ /dev/null"):
            current = None
            continue
        if raw_line.startswith("@@ "):
            match = re.search(r"\+(\d+)(?:,\d+)?", raw_line)
            new_line = int(match.group(1)) if match else 0
            continue
        if current is None:
            continue
        if raw_line.startswith("+"):
            files[current].append((new_line, raw_line[1:]))
            new_line += 1
        elif raw_line.startswith("-"):
            continue
        else:
            new_line += 1

    return {path: lines for path, lines in files.items() if lines}


def _map_line(finding: SecurityFinding, mapped: list[int]) -> int:
    if finding.line is None:
        return mapped[0] if mapped else 1
    index = max(0, min(finding.line - 1, len(mapped) - 1))
    return mapped[index] if mapped else 1


def _safe_snippet(finding: SecurityFinding, airlock: AirlockMiddleware) -> str:
    snippet = finding.masked_snippet or ""
    if not snippet:
        return ""
    try:
        return airlock.sanitize_outbound(
            snippet,
            source=f"pre_commit:{finding.category}",
        ).sanitized_text.strip()[:240]
    except Exception:  # noqa: BLE001 - reporting must never leak or crash
        return "[snippet withheld]"


def _is_blocking(finding: StagedFinding) -> bool:
    return (
        finding.category in _BLOCKING_CATEGORIES
        or finding.severity.lower() in _BLOCKING_SEVERITIES
    )


def _print_block_report(
    console: Console,
    findings: list[StagedFinding],
) -> None:
    table = Table(
        title="Axe Pre-Commit Airlock",
        show_header=True,
        header_style="bold red",
        border_style="red",
        expand=False,
    )
    table.add_column("File")
    table.add_column("Line", justify="right")
    table.add_column("Category", style="bold")
    table.add_column("Severity")
    table.add_column("Detector")
    table.add_column("Masked snippet")

    categories: set[str] = set()
    for finding in findings[:100]:
        categories.add(finding.category)
        table.add_row(
            escape(finding.path),
            str(finding.line),
            escape(finding.category),
            escape(finding.severity),
            escape(finding.detector),
            escape(finding.masked_snippet or "-")[:160],
        )

    console.print()
    console.print(table)
    hints = [
        "Remove the staged secret or replace it with an environment lookup.",
        "Run `axe /airlock scan <path>` for a manual audit.",
        "Then re-stage the fixed files: `git add <path>`.",
    ]
    if any(category in _SECRET_CATEGORIES for category in categories):
        hints.insert(0, "Use \\[e] Extract to .env in the guided remediation flow.")
    console.print(
        Panel(
            "\n".join(f"• {hint}" for hint in hints),
            title="Commit blocked by Code-Airlock",
            border_style="red",
        )
    )
    console.print(
        "[dim]Raw secret values are withheld; only masked snippets are shown.[/dim]\n"
    )
