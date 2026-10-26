"""Bidirectional Airlock sanitization middleware.

Outbound text is scanned by the external ``code-airlock`` MCP server first when
available, then by a zero-dependency built-in cascade. Secrets are held only in
the process-local :class:`SecretVault`; inbound tool arguments can restore exact
raw values without ever sending them to the model.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from axe.config.settings import Settings
from axe.security.secret_vault import (
    PLACEHOLDER_RE,
    RedactionMatch,
    SecretVault,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import Console

    from axe.mcp.client import MCPClientManager

AIRLOCK_SCAN_TOOL_NAMES = (
    "airlock_scan",
    "scan_text",
    "sanitize_payload",
    "audit_diff",
)

_RESTORABLE_TOOLS = {"file_edit", "file_write", "shell"}

# ---------------------------------------------------------------------------
# Built-in cascade scanner
# ---------------------------------------------------------------------------

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?"
    r"-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_JWT_RE = re.compile(
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
)
_OPENROUTER_RE = re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{16,}\b")
_ANTHROPIC_RE = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")
_OPENAI_RE = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b")
_HUGGINGFACE_RE = re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")
_AWS_RE = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_GITHUB_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"
)
_SLACK_RE = re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")
_STRIPE_RE = re.compile(r"\b[sr]k_live_[A-Za-z0-9]{16,}\b")
_ZHIPU_RE = re.compile(r"\b[0-9a-f]{32}\.[A-Za-z0-9]{16,}\b")
_BEARER_RE = re.compile(
    r"(?i)\bBearer\s+([A-Za-z0-9\-._~+/]{20,}=*)"
)
_DB_URL_RE = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|rediss|amqp|amqps|"
    r"mssql)://[^\s\"'`]+:[^\s\"'`@]+@[^\s\"'`]+"
)
_ENV_ASSIGN_RE = re.compile(
    r"""
    (?:^|[\s;,(])
    (?:export\s+)?
    [A-Z0-9_]*(?:API[_-]?KEY|SECRET(?:_KEY)?|TOKEN|PASSWORD|PASSWD|
    PRIVATE[_-]?KEY|ACCESS[_-]?KEY|DATABASE_URL|DB_URL)
    [ \t]*[:=][ \t]*
    (?P<quote>["']?)
    (?P<value>[^\s"'#,\r\n]+)
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}"
    r"[\s.-]\d{4}(?!\w)"
)
_INTL_PHONE_RE = re.compile(r"(?<!\w)\+\d{10,15}(?!\w)")
_IPV4_RE = re.compile(
    r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\w.])"
)

_CATEGORY_PRIORITY = {
    "private_key": 100,
    "jwt": 90,
    "aws_key": 85,
    "db_url": 80,
    "api_key": 70,
    "env_secret": 50,
    "email": 40,
    "phone": 40,
    "ip_address": 40,
    "custom": 10,
}
_IGNORED_ENV_VALUES = {
    "",
    "null",
    "none",
    "true",
    "false",
    "0",
    "1",
    "changeme",
    "change-me",
    "your_key_here",
    "your-key-here",
    "your_api_key",
    "your-api-key",
    "your_token_here",
    "placeholder",
    "example",
    "dummy",
    "secret",
    "token",
    "password",
    "api_key",
    "api-key",
    "apikey",
    "secret_key",
    "access_key",
    "sk-...",
    "...",
}
_IGNORE_MARKERS = (
    "# airlock: ignore",
    "# pragma: allowlist secret",
    "<!-- airlock: ignore -->",
)
_RESERVED_TEST_EMAIL_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "test.local",
    "localhost",
}
_DOC_PLACEHOLDER_RE = re.compile(
    r"(?:your|dummy|example|fake|test)[-_ ]?"
    r"(?:api[-_ ]?key|key|token|secret|password|credential)",
    re.IGNORECASE,
)
_UNSAFE_PATTERNS: list[tuple[str, str, str, str]] = [
    (r"\beval\s*\(", "unsafe_eval", "Dynamic eval() is unsafe.", "high"),
    (r"\bexec\s*\(", "unsafe_exec", "Dynamic exec() is unsafe.", "high"),
    (r"\bos\.system\s*\(", "unsafe_os_system", "os.system() is unsafe.", "high"),
    (
        r"subprocess\.(?:Popen|run|call)\s*\([^)]*shell\s*=\s*True",
        "unsafe_shell_true",
        "subprocess with shell=True is unsafe.",
        "high",
    ),
    (
        r"\byaml\.load\s*\(",
        "unsafe_yaml_load",
        "yaml.load() is unsafe; use yaml.safe_load().",
        "high",
    ),
    (
        r"(?is)\b(?:select|insert|update|delete)\b[^\n]*(?:\{[^}\n]+\}|%s|"
        r"%\([^)]+\)s|[\"']\s*\+\s*[A-Za-z_])",
        "sql_interpolation",
        "SQL query appears to use string interpolation; use parameterized queries.",
        "high",
    ),
    (
        r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sh|bash|zsh)\b",
        "pipe_to_shell",
        "Downloading and piping directly to a shell is unsafe.",
        "critical",
    ),
    (
        r"\brm\s+-rf\b",
        "destructive_rm_rf",
        "Recursive forced deletion is destructive.",
        "high",
    ),
    (
        r"\bpickle\.loads?\s*\(",
        "unsafe_pickle",
        "Untrusted pickle loading is unsafe.",
        "high",
    ),
    (
        r"\bverify\s*=\s*False\b",
        "tls_verification_disabled",
        "TLS verification disabled.",
        "high",
    ),
    (
        r"\bchmod\s+777\b",
        "world_writable",
        "World-writable permissions are unsafe.",
        "high",
    ),
]


def _is_placeholder(value: str) -> bool:
    return PLACEHOLDER_RE.fullmatch(value) is not None or "<AIRLOCK_" in value


def _is_obvious_placeholder_value(value: str) -> bool:
    """Return ``True`` for documentation-style dummy env values."""
    raw = value.strip().strip("\"'")
    if not raw:
        return True
    lowered = raw.lower()
    if lowered in _IGNORED_ENV_VALUES:
        return True
    if lowered.endswith("...") or lowered in {"sk-...", "sk-or-...", "sk-proj-..."}:
        return True
    if lowered.startswith(("your-", "your_", "your ", "<your", "<your-")):
        return True
    if re.fullmatch(r"<[^>]+>", raw):
        return True
    if re.fullmatch(r"x{3,}", lowered):
        return True
    if _DOC_PLACEHOLDER_RE.search(lowered):
        return True
    return False


def _is_reserved_test_email(value: str) -> bool:
    _local, separator, domain = value.rpartition("@")
    if not separator:
        return False
    lowered = domain.lower()
    return any(
        lowered == reserved or lowered.endswith(f".{reserved}")
        for reserved in _RESERVED_TEST_EMAIL_DOMAINS
    )


def _is_suppressed_at(text: str, start: int, end: int | None = None) -> bool:
    """Check the containing line(s) for an inline Airlock ignore marker."""
    line_start = text.rfind("\n", 0, max(start, 0)) + 1
    search_end = max(start, end or start)
    line_end = text.find("\n", search_end)
    if line_end == -1:
        line_end = len(text)
    snippet = text[line_start:line_end].lower()
    return any(marker in snippet for marker in _IGNORE_MARKERS)


@dataclass(slots=True)
class SanitizeResult:
    """Result of one outbound sanitization pass."""

    sanitized_text: str
    redaction_count: int
    findings: list[RedactionMatch] = field(default_factory=list)
    detector_source: str = "builtin-airlock-cascade"

    @property
    def changed(self) -> bool:
        return self.redaction_count > 0


@dataclass(slots=True)
class SecurityFinding:
    """Display-safe security finding for proposed writes and audits."""

    category: str
    detector_source: str = "builtin-airlock-cascade"
    severity: str = "high"
    message: str = ""
    placeholder: str = ""
    source_hint: str | None = None
    unsafe_pattern: str | None = None
    line: int | None = None
    masked_snippet: str | None = None
    raw_value: str = field(default="", repr=False)

    def public_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "detector_source": self.detector_source,
            "severity": self.severity,
            "message": self.message,
            "placeholder": self.placeholder,
            "source_hint": self.source_hint,
            "unsafe_pattern": self.unsafe_pattern,
            "line": self.line,
            "masked_snippet": self.masked_snippet,
        }


class BuiltinAirlockScanner:
    """High-precision, zero-dependency local secret/PII scanner."""

    def scan(
        self,
        text: str,
        *,
        source_hint: str | None = None,
        include_pii: bool = True,
    ) -> list[RedactionMatch]:
        if not text:
            return []

        findings: list[RedactionMatch] = []

        def add(
            raw: str,
            category: str,
            start: int,
            end: int,
        ) -> None:
            if not raw or not raw.strip() or _is_placeholder(raw):
                return
            findings.append(
                RedactionMatch(
                    raw_value=raw,
                    category=category,
                    detector_source="builtin-airlock-cascade",
                    start=start,
                    end=end,
                    source_hint=source_hint,
                )
            )

        for match in _PRIVATE_KEY_RE.finditer(text):
            add(match.group(0), "private_key", match.start(), match.end())
        for match in _JWT_RE.finditer(text):
            add(match.group(0), "jwt", match.start(), match.end())
        for pattern, category in (
            (_OPENROUTER_RE, "api_key"),
            (_ANTHROPIC_RE, "api_key"),
            (_OPENAI_RE, "api_key"),
            (_HUGGINGFACE_RE, "api_key"),
            (_AWS_RE, "aws_key"),
            (_GITHUB_RE, "api_key"),
            (_SLACK_RE, "api_key"),
            (_STRIPE_RE, "api_key"),
            (_ZHIPU_RE, "api_key"),
        ):
            for match in pattern.finditer(text):
                add(match.group(0), category, match.start(), match.end())
        for match in _BEARER_RE.finditer(text):
            add(match.group(1), "api_key", match.start(1), match.end(1))
        for match in _DB_URL_RE.finditer(text):
            add(match.group(0), "db_url", match.start(), match.end())
        for match in _ENV_ASSIGN_RE.finditer(text):
            value = match.group("value")
            start, end = match.span("value")
            stripped = value.rstrip(",;)")
            end -= len(value) - len(stripped)
            value = stripped
            lowered = value.lower()
            if _is_obvious_placeholder_value(value):
                continue
            if lowered in _IGNORED_ENV_VALUES or len(value) < 6:
                continue
            if any(char in value for char in "(){}[]") or value.startswith("$"):
                continue
            if re.match(
                r"(?:os|settings|config|self|env|process|system|ctx|context|app)"
                r"\.[A-Za-z0-9_.]+$",
                value,
            ):
                continue
            add(value, "env_secret", start, end)

        if include_pii:
            for match in _EMAIL_RE.finditer(text):
                add(match.group(0), "email", match.start(), match.end())
            for pattern in (_PHONE_RE, _INTL_PHONE_RE):
                for match in pattern.finditer(text):
                    add(match.group(0), "phone", match.start(), match.end())
            for match in _IPV4_RE.finditer(text):
                ip = match.group(0)
                if ip in {"127.0.0.1", "0.0.0.0"}:
                    continue
                add(ip, "ip_address", match.start(), match.end())

        return self._dedupe(findings)

    @staticmethod
    def _dedupe(findings: list[RedactionMatch]) -> list[RedactionMatch]:
        ordered = sorted(
            findings,
            key=lambda finding: (
                finding.start,
                -(finding.end - finding.start),
                -_CATEGORY_PRIORITY.get(finding.category, 0),
            ),
        )
        accepted: list[RedactionMatch] = []
        for finding in ordered:
            overlaps = any(
                not (
                    finding.end <= existing.start
                    or finding.start >= existing.end
                )
                for existing in accepted
            )
            if overlaps:
                continue
            accepted.append(finding)
        accepted.sort(key=lambda finding: finding.start)
        return accepted


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class AirlockMiddleware:
    """Bidirectional outbound redaction + inbound local restoration."""

    def __init__(
        self,
        settings: Settings,
        *,
        vault: SecretVault | None = None,
        console: "Console | None" = None,
        scanner: BuiltinAirlockScanner | None = None,
        mcp_manager: "MCPClientManager | None" = None,
    ) -> None:
        self.settings = settings
        self.airlock = settings.security.airlock
        self.vault = vault or SecretVault()
        self.console = console
        self.scanner = scanner or BuiltinAirlockScanner()
        self._mcp_manager = mcp_manager
        self._scan_count = 0
        self._redaction_count = 0
        self._restored_count = 0
        self._last_detector_source = "builtin-airlock-cascade"
        self._last_mcp_error: str | None = None

    # ------------------------------------------------------------------
    # Status / wiring
    # ------------------------------------------------------------------

    def set_mcp_manager(self, manager: "MCPClientManager | None") -> None:
        self._mcp_manager = manager

    @property
    def enabled(self) -> bool:
        return bool(self.airlock.enabled)

    @property
    def outbound_redaction(self) -> bool:
        return bool(self.airlock.enabled and self.airlock.redact_outbound_context)

    @property
    def mode(self) -> str:
        if not self.enabled:
            return "disabled"
        try:
            if (
                self._mcp_manager is not None
                and self._mcp_manager.has_scan_tool(AIRLOCK_SCAN_TOOL_NAMES)
            ):
                return "mcp_connected"
        except Exception:  # noqa: BLE001 - status inspection is best-effort
            return "builtin_cascade"
        return "builtin_cascade"

    def stats(self) -> dict[str, Any]:
        vault_stats = self.vault.stats()
        mcp_status = "not_configured"
        if self._mcp_manager is not None:
            mcp_status = str(
                getattr(self._mcp_manager, "status", "unknown")
            )
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "outbound_redaction": self.outbound_redaction,
            "restore_on_file_write": bool(self.airlock.restore_on_file_write),
            "mcp_status": mcp_status,
            "mcp_error": self._last_mcp_error,
            "scan_count": self._scan_count,
            "redaction_count": self._redaction_count,
            "restored_placeholders": self._restored_count,
            "last_detector_source": self._last_detector_source,
            **vault_stats,
        }

    # ------------------------------------------------------------------
    # Outbound sanitization
    # ------------------------------------------------------------------

    def sanitize_outbound(
        self,
        text: str,
        source: str = "context",
    ) -> SanitizeResult:
        """Scan and redact outbound text, registering findings in the vault."""
        if not self.outbound_redaction or not text:
            return SanitizeResult(
                sanitized_text=text,
                redaction_count=0,
                findings=[],
                detector_source="disabled",
            )

        findings = self._scan(text, source_hint=source)
        if not findings:
            self._scan_count += 1
            return SanitizeResult(
                sanitized_text=text,
                redaction_count=0,
                findings=[],
                detector_source=self._last_detector_source,
            )

        sanitized = self.vault.redact_text(text, findings)
        count = sum(
            max(finding.redacted_count, 1)
            for finding in findings
            if finding.placeholder
        )
        self._scan_count += 1
        self._redaction_count += count
        return SanitizeResult(
            sanitized_text=sanitized,
            redaction_count=count,
            findings=findings,
            detector_source=findings[0].detector_source,
        )

    def sanitize_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return a sanitized deep copy of outbound LLM messages."""
        return self.sanitize_payload(messages, source="llm_messages")

    def sanitize_payload(
        self,
        value: Any,
        *,
        source: str = "payload",
    ) -> Any:
        """Recursively sanitize any JSON-like payload."""
        if not self.outbound_redaction:
            return copy.deepcopy(value)

        def _sanitize(item: Any) -> Any:
            if isinstance(item, str):
                return self.sanitize_outbound(item, source=source).sanitized_text
            if isinstance(item, dict):
                return {key: _sanitize(child) for key, child in item.items()}
            if isinstance(item, list):
                return [_sanitize(child) for child in item]
            if isinstance(item, tuple):
                return tuple(_sanitize(child) for child in item)
            return item

        return _sanitize(value)

    # ------------------------------------------------------------------
    # Inbound restoration
    # ------------------------------------------------------------------

    def restore_inbound_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> dict[str, Any]:
        """Restore placeholders in local write/shell tool arguments."""
        if not self.enabled or not self.airlock.restore_on_file_write:
            return dict(tool_args)
        if tool_name not in _RESTORABLE_TOOLS:
            return dict(tool_args)
        restored, count = self.vault.restore_tool_args(tool_args, enabled=True)
        self._restored_count += count
        return restored

    # ------------------------------------------------------------------
    # Scanning / reporting
    # ------------------------------------------------------------------

    def scan_text(
        self,
        text: str,
        *,
        source_hint: str | None = None,
    ) -> list[RedactionMatch]:
        """Scan text and register discovered values for placeholder display."""
        findings = self._scan(text, source_hint=source_hint, include_pii=True)
        for finding in findings:
            finding.placeholder = self.vault.register(
                finding.raw_value,
                finding.category,
                finding.detector_source,
                finding.source_hint,
                count=max(finding.redacted_count, 1),
            )
        self._scan_count += 1
        return findings

    def scan_file(self, path: str | Path) -> list[RedactionMatch]:
        file_path = Path(path).expanduser()
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return self.scan_text(text, source_hint=str(file_path))

    def scan_path(self, path_or_text: str) -> list[RedactionMatch]:
        """Scan a file, directory tree, or raw text string."""
        try:
            candidate = Path(path_or_text).expanduser()
            is_file = candidate.is_file()
            is_dir = candidate.is_dir()
        except (OSError, ValueError):
            is_file = False
            is_dir = False
            candidate = Path(".")
        if is_file:
            return self.scan_file(candidate)
        if not is_dir:
            return self.scan_text(path_or_text, source_hint="inline_text")

        findings: list[RedactionMatch] = []
        skip_dirs = {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".axe",
            "dist",
            "build",
        }
        text_suffixes = {
            ".py", ".toml", ".json", ".md", ".txt", ".yaml", ".yml",
            ".ini", ".cfg", ".sh", ".ps1", ".js", ".ts", ".tsx", ".jsx",
            ".rs", ".go", ".java", ".rb", ".php", ".sql", ".xml", ".html",
            ".css", ".lock", ".example", ".env",
        }
        scanned = 0
        for file_path in sorted(candidate.rglob("*")):
            if scanned >= 250:
                break
            if not file_path.is_file():
                continue
            if any(part in skip_dirs for part in file_path.parts):
                continue
            if file_path.suffix.lower() not in text_suffixes and not file_path.name.startswith(".env"):
                continue
            try:
                if file_path.stat().st_size > 1_500_000:
                    continue
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            scanned += 1
            findings.extend(self.scan_text(text, source_hint=str(file_path)))
        return findings

    def scan_proposed_write(
        self,
        path: str,
        content_or_diff: str,
    ) -> list[SecurityFinding]:
        """Inspect LLM-generated content for new secrets and unsafe patterns."""
        if not self.enabled:
            return []

        findings = self._scan(
            content_or_diff,
            source_hint=path or "proposed_write",
            include_pii=True,
        )
        findings = [
            finding
            for finding in findings
            if not _is_suppressed_at(
                content_or_diff,
                finding.start,
                finding.end,
            )
            and not (
                finding.category == "email"
                and _is_reserved_test_email(finding.raw_value)
            )
        ]
        results: list[SecurityFinding] = []
        lines = content_or_diff.splitlines()
        for finding in findings:
            placeholder = self.vault.register(
                finding.raw_value,
                finding.category,
                finding.detector_source,
                path,
                count=max(finding.redacted_count, 1),
            )
            line_number = content_or_diff.count("\n", 0, finding.start) + 1
            masked_snippet: str | None = None
            if 1 <= line_number <= len(lines):
                line_text = lines[line_number - 1]
                if finding.raw_value and finding.raw_value in line_text:
                    masked_snippet = line_text.replace(
                        finding.raw_value,
                        placeholder,
                    )
                elif "\n" in finding.raw_value:
                    masked_snippet = f"[multiline {finding.category} -> {placeholder}]"
            results.append(
                SecurityFinding(
                    category=finding.category,
                    detector_source=finding.detector_source,
                    severity="high",
                    message=f"Proposed write contains a {finding.category}.",
                    placeholder=placeholder,
                    source_hint=path,
                    line=line_number,
                    masked_snippet=masked_snippet,
                    raw_value=finding.raw_value,
                )
            )

        results.extend(self._unsafe_findings(content_or_diff, path))
        return results

    def scan_proposed_command(self, command: str) -> list[SecurityFinding]:
        """Inspect a proposed shell command for secrets and unsafe patterns."""
        if not self.enabled or not command:
            return []

        findings = self._scan(
            command,
            source_hint="shell_command",
            include_pii=True,
        )
        findings = [
            finding
            for finding in findings
            if not _is_suppressed_at(command, finding.start, finding.end)
        ]
        results: list[SecurityFinding] = []
        for finding in findings:
            placeholder = self.vault.register(
                finding.raw_value,
                finding.category,
                finding.detector_source,
                "shell_command",
                count=max(finding.redacted_count, 1),
            )
            masked = command.replace(finding.raw_value, placeholder)
            results.append(
                SecurityFinding(
                    category=finding.category,
                    detector_source=finding.detector_source,
                    severity="high",
                    message=f"Shell command contains a {finding.category}.",
                    placeholder=placeholder,
                    source_hint="shell_command",
                    line=1,
                    masked_snippet=masked[:300],
                    raw_value=finding.raw_value,
                )
            )
        results.extend(self._unsafe_findings(command, "shell_command"))
        return results

    @staticmethod
    def _unsafe_findings(text: str, source_hint: str | None) -> list[SecurityFinding]:
        results: list[SecurityFinding] = []
        for pattern, category, message, severity in _UNSAFE_PATTERNS:
            match = re.search(pattern, text)
            if match is None:
                continue
            if _is_suppressed_at(text, match.start(), match.end()):
                continue
            line = text.count("\n", 0, match.start()) + 1
            results.append(
                SecurityFinding(
                    category="unsafe_pattern",
                    detector_source="builtin-airlock-cascade",
                    severity=severity,
                    message=message,
                    source_hint=source_hint,
                    unsafe_pattern=category,
                    line=line,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Detection internals
    # ------------------------------------------------------------------

    def _scan(
        self,
        text: str,
        *,
        source_hint: str | None,
        include_pii: bool | None = None,
    ) -> list[RedactionMatch]:
        try:
            mcp_findings = self._scan_with_mcp(text, source_hint)
        except Exception as exc:  # noqa: BLE001 - MCP scanning is advisory
            self._last_mcp_error = str(exc)
            mcp_findings = []
        builtin_findings = self.scanner.scan(
            text,
            source_hint=source_hint,
            include_pii=(
                self.outbound_redaction if include_pii is None else include_pii
            ),
        )
        if not mcp_findings:
            self._last_detector_source = "builtin-airlock-cascade"
            self._last_mcp_error = None
            return builtin_findings

        self._last_detector_source = "code-airlock-mcp"
        self._last_mcp_error = None
        mcp_keys = {
            (finding.raw_value, finding.start, finding.end)
            for finding in mcp_findings
        }
        merged = [
            *mcp_findings,
            *[
                finding
                for finding in builtin_findings
                if (finding.raw_value, finding.start, finding.end) not in mcp_keys
            ],
        ]
        return self.scanner._dedupe(merged)

    def _scan_with_mcp(
        self,
        text: str,
        source_hint: str | None,
    ) -> list[RedactionMatch]:
        manager = self._mcp_manager
        if manager is None or not manager.has_scan_tool(AIRLOCK_SCAN_TOOL_NAMES):
            return []

        tool = manager.find_scan_tool(AIRLOCK_SCAN_TOOL_NAMES)
        if tool is None:
            return []
        server_name, tool_name = tool
        result: Any = None
        last_error: Exception | None = None
        for args in ({"text": text}, {"content": text}, {"payload": text}):
            try:
                result = manager.call_tool(tool_name, args, server=server_name)
                break
            except Exception as exc:  # noqa: BLE001 - MCP scan is advisory
                last_error = exc
        if result is None:
            self._last_mcp_error = str(last_error) if last_error else "empty scan result"
            return []

        items: Any = result
        if isinstance(result, str):
            try:
                items = json.loads(result)
            except json.JSONDecodeError:
                items = None
        if isinstance(items, Mapping):
            for key in ("findings", "matches", "secrets", "results"):
                candidate = items.get(key)
                if isinstance(candidate, list):
                    items = candidate
                    break

        if not isinstance(items, list):
            self._last_mcp_error = "MCP scan returned no usable findings"
            return []

        findings: list[RedactionMatch] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            raw = (
                item.get("value")
                or item.get("match")
                or item.get("raw")
                or item.get("secret")
                or item.get("text")
            )
            if not isinstance(raw, str) or not raw or _is_placeholder(raw):
                continue
            category = str(item.get("category") or item.get("type") or "custom")
            start = text.find(raw)
            if start < 0:
                continue
            findings.append(
                RedactionMatch(
                    raw_value=raw,
                    category=category,
                    detector_source="code-airlock-mcp",
                    start=start,
                    end=start + len(raw),
                    source_hint=source_hint or "mcp_scan",
                )
            )
        if not findings:
            self._last_mcp_error = "MCP scan returned no raw findings; using builtin scanner"
        return findings
