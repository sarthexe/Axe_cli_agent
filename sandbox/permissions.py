"""Command safety checks for shell execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from config.settings import AgentSettings, SandboxSettings


@dataclass(slots=True)
class SafetyDecision:
    """Outcome of analyzing a shell command."""

    blocked: bool = False
    needs_confirmation: bool = False
    reason: str = ""
    rule: str = ""


class CommandSafety:
    """Analyze shell commands against blocklist and confirmation rules."""

    _DEFAULT_BLOCK_PATTERNS: list[tuple[str, str]] = [
        ("rm_root", r"(^|[;&|])\s*rm\s+-rf\s+/\s*($|[;&|])"),
        ("mkfs", r"\bmkfs(\.[a-z0-9_+-]+)?\b"),
        ("dd_zero", r"\bdd\b[^;\n]*\bif=/dev/zero\b"),
        ("fork_bomb", r":\(\)\s*\{\s*:\|:&\s*\};:"),
    ]
    _DEFAULT_CONFIRM_PATTERNS: list[tuple[str, str]] = [
        ("find_delete", r"\bfind\b[^;\n]*\s-delete\b"),
        ("python_remove", r"\b(os\.remove|os\.unlink|shutil\.rmtree|pathlib\.path\([^)]*\)\.unlink)\b"),
        ("rm_file", r"(^|[;&|])\s*rm\s+(-[a-zA-Z]+\s+)*[^/\n]+"),
    ]

    def __init__(
        self,
        sandbox_settings: SandboxSettings | None = None,
        agent_settings: AgentSettings | None = None,
    ) -> None:
        self._block_patterns: list[tuple[str, Pattern[str]]] = [
            (name, re.compile(pattern, re.IGNORECASE))
            for name, pattern in self._DEFAULT_BLOCK_PATTERNS
        ]

        self._literal_blocked: list[str] = []
        if sandbox_settings is not None:
            self._literal_blocked = [
                item.strip() for item in sandbox_settings.blocked_commands if item.strip()
            ]

        confirmation_tokens = (
            agent_settings.confirmation_required
            if agent_settings is not None
            else ["rm", "drop", "kill", "truncate"]
        )
        self._confirm_patterns: list[Pattern[str]] = [
            re.compile(rf"(^|[;&|])\s*.*\b{re.escape(token)}\b", re.IGNORECASE)
            for token in confirmation_tokens
            if token.strip()
        ]
        self._confirm_patterns.extend(
            re.compile(pattern, re.IGNORECASE)
            for _, pattern in self._DEFAULT_CONFIRM_PATTERNS
        )

    def analyze(self, command: str) -> SafetyDecision:
        """Return safety decision for the provided shell command."""
        raw = command.strip()
        if not raw:
            return SafetyDecision()

        for name, pattern in self._block_patterns:
            if pattern.search(raw):
                reason = self._reason_for_rule(name)
                return SafetyDecision(blocked=True, reason=reason, rule=name)

        lowered = raw.lower()
        for literal in self._literal_blocked:
            if literal.lower() in lowered:
                return SafetyDecision(
                    blocked=True,
                    reason=f"Matches blocked command pattern '{literal}'.",
                    rule="literal_block",
                )

        for pattern in self._confirm_patterns:
            if pattern.search(raw):
                return SafetyDecision(
                    needs_confirmation=True,
                    reason="Potentially destructive command.",
                    rule="confirm_required",
                )

        return SafetyDecision()

    @staticmethod
    def blocked_message(command: str, decision: SafetyDecision) -> str:
        """Human-readable blocked message for tool output."""
        return (
            f"[BLOCKED] The command '{command}' was blocked for safety. "
            f"{decision.reason or 'This command matches a forbidden pattern.'}"
        )

    @staticmethod
    def _reason_for_rule(rule: str) -> str:
        reasons = {
            "rm_root": "Refusing recursive delete at filesystem root.",
            "mkfs": "Formatting disks/filesystems is not allowed.",
            "dd_zero": "Overwriting devices with /dev/zero is not allowed.",
            "fork_bomb": "Fork bomb pattern detected.",
        }
        return reasons.get(rule, "Command matched a blocked rule.")
