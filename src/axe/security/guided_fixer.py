"""Guided security remediation for proposed writes and shell commands.

The fixer is deterministic: it detects actionable findings, produces a
:class:`RemediationPlan` (env extraction, safe YAML load, etc.), and can guard
an agent tool call before it reaches disk or a shell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from axe.cli.security_prompt import (
    GuidedSecurityDecision,
    prompt_guided_security_issue,
)
from axe.config.settings import Settings
from axe.security.airlock_middleware import AirlockMiddleware, SecurityFinding

_GUARDED_TOOLS = {"file_write", "file_edit", "shell"}
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
_DEFAULT_ENV_NAMES = {
    "api_key": "API_KEY",
    "jwt": "JWT_TOKEN",
    "private_key": "PRIVATE_KEY",
    "aws_key": "AWS_ACCESS_KEY_ID",
    "db_url": "DATABASE_URL",
    "env_secret": "SECRET_VALUE",
    "custom": "SECRET_VALUE",
}
_CODE_SUFFIXES_PY = {".py"}
_CODE_SUFFIXES_JS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}


@dataclass(slots=True)
class RemediationResult:
    """Outcome of applying a remediation plan."""

    path: str
    files_changed: list[str] = field(default_factory=list)
    env_keys: list[str] = field(default_factory=list)
    target_written: bool = False


@dataclass(slots=True)
class RemediationPlan:
    """Deterministic changes proposed for one security finding set."""

    path: str
    original_content: str
    fixed_content: str
    findings: list[SecurityFinding] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    env_entries: dict[str, str] = field(default_factory=dict, repr=False)
    env_example_entries: dict[str, str] = field(default_factory=dict)
    repo_root: Path | None = None
    target_is_full_file: bool = True

    @property
    def has_changes(self) -> bool:
        return self.fixed_content != self.original_content or bool(self.env_entries)

    @property
    def environment_keys(self) -> list[str]:
        return list(self.env_entries)

    def apply(self, *, write_target: bool = True) -> RemediationResult:
        """Apply the plan. Raw env values are written only to local ``.env``."""
        files_changed: list[str] = []
        target_written = False

        if (
            write_target
            and self.target_is_full_file
            and self.fixed_content != self.original_content
        ):
            target = Path(self.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.fixed_content, encoding="utf-8")
            files_changed.append(str(target))
            target_written = True

        root = self.repo_root or _repo_root_for(self.path)
        if self.env_entries:
            env_path = root / ".env"
            gitignore_path = root / ".gitignore"
            if _upsert_env(env_path, self.env_entries):
                files_changed.append(str(env_path))
            if _ensure_gitignore(gitignore_path, ".env"):
                files_changed.append(str(gitignore_path))

        if self.env_example_entries:
            example_path = root / ".env.example"
            if _upsert_env(example_path, self.env_example_entries):
                files_changed.append(str(example_path))

        return RemediationResult(
            path=self.path,
            files_changed=files_changed,
            env_keys=list(self.env_entries),
            target_written=target_written,
        )

    def apply_auxiliary(self) -> RemediationResult:
        """Apply env/gitignore changes without writing the target file."""
        return self.apply(write_target=False)

    def safe_summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "changes": list(self.changes),
            "environment_keys": list(self.env_entries),
            "has_changes": self.has_changes,
        }


@dataclass(slots=True)
class GuardResult:
    """Result of reviewing a tool call before local execution."""

    allowed: bool
    arguments: dict[str, Any]
    message: str = ""
    findings: list[SecurityFinding] = field(default_factory=list)
    plan: RemediationPlan | None = None
    choice: str = "allow"
    audit_note: str = ""


class GuidedSecurityFixer:
    """Detect, explain, and deterministically remediate tool-call risks."""

    def __init__(
        self,
        settings: Settings,
        airlock: AirlockMiddleware | None = None,
        *,
        repo_root: Path | None = None,
        console: Console | None = None,
        assume_yes: bool = False,
    ) -> None:
        self.settings = settings
        self.airlock = airlock or AirlockMiddleware(settings, console=console)
        self.repo_root = Path(repo_root) if repo_root is not None else None
        self.console = console or Console()
        self.assume_yes = assume_yes

    @property
    def mode(self) -> str:
        return self.settings.security.airlock.guided_remediation_mode

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def scan_proposed_write(self, path: str, content: str) -> list[SecurityFinding]:
        return self.airlock.scan_proposed_write(path, content)

    def scan_proposed_command(self, command: str) -> list[SecurityFinding]:
        return self.airlock.scan_proposed_command(command)

    # ------------------------------------------------------------------
    # Remediation
    # ------------------------------------------------------------------

    def generate_remediation(
        self,
        path: str,
        content: str,
        findings: list[SecurityFinding],
        *,
        target_is_full_file: bool = True,
    ) -> RemediationPlan:
        """Create a deterministic remediation plan without touching disk."""
        fixed = content
        changes: list[str] = []
        env_entries: dict[str, str] = {}
        env_example_entries: dict[str, str] = {}
        processed_secrets: set[str] = set()
        root = self.repo_root or _repo_root_for(path)
        suffix = Path(path).suffix.lower()

        for finding in findings:
            category = finding.category
            if category in _SECRET_CATEGORIES and finding.raw_value:
                raw = finding.raw_value
                if raw in processed_secrets:
                    continue
                processed_secrets.add(raw)
                key = _unique_env_key(
                    _extract_key_name(fixed, raw, category),
                    env_entries,
                    raw,
                )
                replacement = _env_lookup(suffix, key)
                if replacement is None:
                    continue
                quoted = re.compile(
                    r"(?P<quote>[\"'])" + re.escape(raw) + r"(?P=quote)"
                )
                if not quoted.search(fixed):
                    continue
                fixed = quoted.sub(replacement, fixed)
                if suffix in _CODE_SUFFIXES_PY:
                    fixed = _ensure_os_import(fixed)
                env_entries[key] = raw
                env_example_entries[key] = finding.placeholder or "your_value_here"
                changes.append(
                    f"Replaced hardcoded {category} with environment lookup {key}."
                )

            if finding.unsafe_pattern == "unsafe_yaml_load" or (
                category == "unsafe_pattern" and "yaml.load" in finding.message
            ):
                updated = re.sub(r"\byaml\.load\s*\(", "yaml.safe_load(", fixed)
                if updated != fixed:
                    fixed = updated
                    changes.append("Replaced yaml.load() with yaml.safe_load().")

        if not changes and not env_entries:
            changes.append("No deterministic auto-fix available; manual review required.")

        return RemediationPlan(
            path=path,
            original_content=content,
            fixed_content=fixed,
            findings=list(findings),
            changes=changes,
            env_entries=env_entries,
            env_example_entries=env_example_entries,
            repo_root=root,
            target_is_full_file=target_is_full_file,
        )

    def remediation_hint(self, findings: list[SecurityFinding]) -> str:
        hints: list[str] = []
        for finding in findings:
            category = finding.category
            if category in _SECRET_CATEGORIES:
                hints.append(
                    "Extract the credential to an environment variable and reference "
                    "it with os.environ.get()/process.env."
                )
            elif category in _PII_CATEGORIES:
                hints.append("Remove or replace developer PII before writing.")
            elif category == "risk_gate":
                hints.append("Review the high-risk action before proceeding.")
            elif finding.unsafe_pattern == "unsafe_yaml_load":
                hints.append("Use yaml.safe_load() instead of yaml.load().")
            elif finding.unsafe_pattern == "sql_interpolation":
                hints.append("Use parameterized SQL queries.")
            elif finding.unsafe_pattern == "unsafe_shell_true":
                hints.append("Pass an argument list to subprocess and avoid shell=True.")
            elif finding.unsafe_pattern in {"unsafe_eval", "unsafe_exec"}:
                hints.append("Avoid dynamic eval()/exec() code execution.")
            elif finding.unsafe_pattern in {
                "destructive_rm_rf",
                "pipe_to_shell",
                "world_writable",
            }:
                hints.append("Replace the destructive command with a scoped safe alternative.")
            elif finding.message:
                hints.append(finding.message)
        # Preserve order while removing duplicates.
        return " ".join(dict.fromkeys(hints))

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def guard_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        risk_flagged: bool = False,
    ) -> GuardResult:
        """Review a file-write/edit/shell call before local execution."""
        if tool_name not in _GUARDED_TOOLS:
            return GuardResult(allowed=True, arguments=dict(arguments))
        if not self.settings.security.airlock.enabled:
            return GuardResult(allowed=True, arguments=dict(arguments))

        known_raw = {
            entry.raw_value
            for entry in self.airlock.vault.entries()
            if entry.raw_value
        }
        path = str(arguments.get("path") or "")
        if tool_name == "file_write":
            content = str(arguments.get("content") or "")
            findings = self.scan_proposed_write(path, content)
        elif tool_name == "file_edit":
            replacement = str(arguments.get("replacement") or "")
            findings = self.scan_proposed_write(path, replacement)
        else:
            command = str(arguments.get("command") or "")
            findings = self.scan_proposed_command(command)
            content = ""

        # Secrets already mapped to a vault placeholder were not introduced by
        # this tool call; re-writing an existing credential is not a new leak.
        findings = [
            finding
            for finding in findings
            if not (finding.raw_value and finding.raw_value in known_raw)
        ]

        if risk_flagged:
            findings.insert(
                0,
                SecurityFinding(
                    category="risk_gate",
                    detector_source="jev",
                    severity="high",
                    message="Jev flagged this action as high risk.",
                    source_hint=path or "shell_command",
                ),
            )

        if not findings:
            return GuardResult(allowed=True, arguments=dict(arguments))

        hint = self.remediation_hint(findings)

        def block(reason: str) -> GuardResult:
            # Do not let a blocked secret remain "known"; otherwise a retry of
            # the same unsafe content would be treated as pre-existing.
            for finding in findings:
                raw = finding.raw_value
                if raw and raw not in known_raw:
                    self.airlock.vault.remove_raw(raw)
            return self._blocked(
                tool_name,
                arguments,
                findings,
                hint,
                reason=reason,
            )

        mode = self.mode

        if mode == "strict_block":
            return block("strict_block mode")

        decision: GuidedSecurityDecision
        if mode == "warn":
            self.console.print(
                "[warning]Airlock warning: security findings detected; "
                "warn mode allows the action.[/warning]"
            )
            decision = GuidedSecurityDecision("ignore", "warn mode")
        else:
            decision = prompt_guided_security_issue(
                self.console,
                findings,
                path=path or "shell_command",
                mode=mode,
                assume_yes=self.assume_yes,
                remediation_hint=hint,
            )

        if decision.choice == "block":
            return block(decision.reason or "user blocked")

        if decision.choice == "ignore":
            return GuardResult(
                allowed=True,
                arguments=dict(arguments),
                findings=findings,
                choice="ignore",
                audit_note=(
                    f"Ignored once: {tool_name} risk accepted "
                    f"({decision.reason or 'user choice'})."
                ),
            )

        plan = self.generate_remediation(
            path or tool_name,
            content if tool_name == "file_write" else str(arguments.get("replacement") or ""),
            findings,
            target_is_full_file=tool_name == "file_write",
        )
        if tool_name == "shell":
            return block("shell auto-remediation requires manual rewrite")
        if decision.choice == "extract" and not plan.env_entries:
            return block("no extractable secret found")
        if plan.fixed_content == plan.original_content:
            return block("no deterministic auto-fix available")

        result = plan.apply_auxiliary()
        updated = dict(arguments)
        if tool_name == "file_write":
            updated["content"] = plan.fixed_content
        else:
            updated["replacement"] = plan.fixed_content

        return GuardResult(
            allowed=True,
            arguments=updated,
            findings=findings,
            plan=plan,
            choice=decision.choice,
            audit_note=(
                f"Auto-remediated {tool_name}: {len(plan.changes)} change(s); "
                f"env keys: {', '.join(result.env_keys) or 'none'}."
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _blocked(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        findings: list[SecurityFinding],
        hint: str,
        *,
        reason: str,
    ) -> GuardResult:
        path = str(arguments.get("path") or "shell_command")
        details = ", ".join(
            (
                f"{finding.category}"
                + (f" at line {finding.line}" if finding.line else "")
            )
            for finding in findings[:6]
        )
        message = (
            f"[BLOCKED] Airlock security issue in {tool_name} ({path}): "
            f"{details}. {hint} Rewrite the tool call safely; do not retry the "
            f"same unsafe content. ({reason})"
        )
        return GuardResult(
            allowed=False,
            arguments=dict(arguments),
            message=message,
            findings=findings,
            choice="block",
            audit_note=f"Blocked {tool_name}: {details}. Reason: {reason}.",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _repo_root_for(path: str) -> Path:
    candidate = Path(path).expanduser()
    start = candidate.parent if candidate.suffix else candidate
    try:
        start = start.resolve()
    except OSError:
        start = Path.cwd()
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return start


def _extract_key_name(content: str, raw: str, category: str) -> str:
    for line in content.splitlines():
        if raw not in line:
            continue
        match = re.search(
            r"""(?:export\s+)?(?:const|let|var|final|private|public|static)?\s*
                ["']?([A-Za-z_][A-Za-z0-9_]*)["']?\s*
                (?::[^=\n]+)?\s*[:=]""",
            line,
            re.VERBOSE,
        )
        if match:
            return _normalize_env_key(match.group(1))
    return _DEFAULT_ENV_NAMES.get(category, "SECRET_VALUE")


def _normalize_env_key(key: str) -> str:
    key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    key = re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_").upper()
    if not key:
        return "SECRET_VALUE"
    if key[0].isdigit():
        key = f"SECRET_{key}"
    return key


def _unique_env_key(key: str, existing: dict[str, str], raw: str) -> str:
    if key not in existing or existing[key] == raw:
        return key
    index = 2
    while f"{key}_{index}" in existing:
        index += 1
    return f"{key}_{index}"


def _env_lookup(suffix: str, key: str) -> str | None:
    if suffix in _CODE_SUFFIXES_PY:
        return f'os.environ.get("{key}")'
    if suffix in _CODE_SUFFIXES_JS:
        return f"process.env.{key}"
    return None


def _ensure_os_import(content: str) -> str:
    if re.search(r"(?m)^\s*import\s+os\b", content) or re.search(
        r"(?m)^\s*from\s+os\s+import\b",
        content,
    ):
        return content
    lines = content.splitlines()
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if insert_at < len(lines) and "coding" in lines[insert_at] and lines[insert_at].startswith("#"):
        insert_at += 1
    lines.insert(insert_at, "import os")
    suffix = "\n" if content.endswith("\n") else ""
    return "\n".join(lines) + suffix


def _upsert_env(path: Path, entries: dict[str, str]) -> bool:
    if not entries:
        return False
    existing_text = ""
    if path.exists():
        try:
            existing_text = path.read_text(encoding="utf-8")
        except OSError:
            existing_text = ""
    lines = existing_text.splitlines()
    changed = False
    for key, value in entries.items():
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=.*$")
        replaced = False
        for index, line in enumerate(lines):
            if pattern.match(line):
                new_line = f"{key}={value}"
                if line != new_line:
                    lines[index] = new_line
                    changed = True
                replaced = True
                break
        if not replaced:
            lines.append(f"{key}={value}")
            changed = True
    if changed or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True
    return False


def _ensure_gitignore(path: Path, entry: str) -> bool:
    lines: list[str] = []
    if path.exists():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
    if entry in {line.strip() for line in lines}:
        return False
    lines.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True
