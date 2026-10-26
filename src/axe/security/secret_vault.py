"""Process-local ephemeral vault for bidirectional Airlock redaction.

Raw secret values live only in this process's memory. Placeholders are
deterministic and stable for the lifetime of the process, so the same secret
seen in ``.env`` and later in ``config.py`` always maps to the same token.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

PLACEHOLDER_RE = re.compile(r"<AIRLOCK_(?:SECRET|PII)_\d{3,}>")
_PLACEHOLDER_SCAN_RE = re.compile(r"<AIRLOCK_(?:SECRET|PII)_\d{3,}>")

PII_CATEGORIES = frozenset({"email", "phone", "ip_address"})


def is_pii_category(category: str) -> bool:
    return category in PII_CATEGORIES


@dataclass(slots=True)
class RedactionMatch:
    """One detected span that should be redacted."""

    placeholder: str = ""
    category: str = "custom"
    detector_source: str = "builtin-airlock-cascade"
    redacted_count: int = 1
    source_hint: str | None = None
    raw_value: str = field(default="", repr=False)
    start: int = 0
    end: int = 0

    def public_dict(self) -> dict[str, Any]:
        """Return a display-safe summary (never includes raw_value)."""
        return {
            "placeholder": self.placeholder,
            "category": self.category,
            "detector_source": self.detector_source,
            "redacted_count": self.redacted_count,
            "source_hint": self.source_hint,
        }


@dataclass(slots=True)
class VaultEntry:
    """Bidirectional mapping for one unique raw value."""

    placeholder: str = ""
    category: str = "custom"
    detector_source: str = "builtin-airlock-cascade"
    redacted_count: int = 1
    source_hint: str | None = None
    raw_value: str = field(default="", repr=False)

    @property
    def is_pii(self) -> bool:
        return is_pii_category(self.category)


class SecretVault:
    """In-memory placeholder <-> raw-value mapping."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_raw: dict[str, VaultEntry] = {}
        self._by_placeholder: dict[str, VaultEntry] = {}
        self._secret_counter = 0
        self._pii_counter = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        raw_value: str,
        category: str = "custom",
        detector_source: str = "builtin-airlock-cascade",
        source_hint: str | None = None,
        *,
        count: int = 1,
    ) -> str:
        """Register a raw value and return its deterministic placeholder.

        Registering the same raw value again returns the exact same placeholder
        and increments that entry's occurrence count.
        """
        if not raw_value:
            return raw_value
        if PLACEHOLDER_RE.fullmatch(raw_value):
            return raw_value
        increment = max(int(count), 1)

        with self._lock:
            existing = self._by_raw.get(raw_value)
            if existing is not None:
                existing.redacted_count += increment
                if existing.source_hint is None and source_hint:
                    existing.source_hint = source_hint
                return existing.placeholder

            if is_pii_category(category):
                self._pii_counter += 1
                placeholder = f"<AIRLOCK_PII_{self._pii_counter:03d}>"
            else:
                self._secret_counter += 1
                placeholder = f"<AIRLOCK_SECRET_{self._secret_counter:03d}>"

            entry = VaultEntry(
                placeholder=placeholder,
                raw_value=raw_value,
                category=category,
                detector_source=detector_source,
                redacted_count=increment,
                source_hint=source_hint,
            )
            self._by_raw[raw_value] = entry
            self._by_placeholder[placeholder] = entry
            return placeholder

    # ------------------------------------------------------------------
    # Redaction / restoration
    # ------------------------------------------------------------------

    def redact_text(
        self,
        text: str,
        findings: list[RedactionMatch] | None = None,
    ) -> str:
        """Replace raw values in ``text`` with deterministic placeholders.

        Replacements are applied longest-match first so a shorter token cannot
        corrupt a longer credential that contains it.
        """
        if not text:
            return text

        candidates: list[RedactionMatch] = []
        if findings:
            candidates = [
                finding
                for finding in findings
                if finding.raw_value
                and not PLACEHOLDER_RE.fullmatch(finding.raw_value)
            ]
        else:
            with self._lock:
                entries = list(self._by_raw.values())
            candidates = [
                RedactionMatch(
                    raw_value=entry.raw_value,
                    category=entry.category,
                    detector_source=entry.detector_source,
                    source_hint=entry.source_hint,
                )
                for entry in entries
            ]

        # Longest raw value first; ties keep their original order.
        candidates.sort(key=lambda finding: len(finding.raw_value), reverse=True)

        redacted = text
        for finding in candidates:
            raw = finding.raw_value
            if not raw or raw not in redacted:
                continue
            occurrences = redacted.count(raw)
            if occurrences <= 0:
                continue
            placeholder = self.register(
                raw,
                finding.category,
                finding.detector_source,
                finding.source_hint,
                count=occurrences,
            )
            finding.placeholder = placeholder
            finding.redacted_count = occurrences
            redacted = redacted.replace(raw, placeholder)

        return redacted

    def restore_text(self, text: str) -> tuple[str, int]:
        """Restore all known placeholders in ``text``."""
        if not text:
            return text, 0

        with self._lock:
            entries = dict(self._by_placeholder)

        restored_count = 0

        def _replace(match: re.Match[str]) -> str:
            nonlocal restored_count
            entry = entries.get(match.group(0))
            if entry is None:
                return match.group(0)
            restored_count += 1
            return entry.raw_value

        restored = _PLACEHOLDER_SCAN_RE.sub(_replace, text)
        return restored, restored_count

    def restore_tool_args(
        self,
        args: dict[str, Any],
        enabled: bool = True,
    ) -> tuple[dict[str, Any], int]:
        """Recursively restore placeholders in tool arguments."""
        if not enabled:
            return dict(args), 0

        def _restore(value: Any) -> tuple[Any, int]:
            if isinstance(value, str):
                return self.restore_text(value)
            if isinstance(value, dict):
                restored: dict[str, Any] = {}
                total = 0
                for key, item in value.items():
                    new_item, count = _restore(item)
                    restored[key] = new_item
                    total += count
                return restored, total
            if isinstance(value, list):
                restored_list: list[Any] = []
                total = 0
                for item in value:
                    new_item, count = _restore(item)
                    restored_list.append(new_item)
                    total += count
                return restored_list, total
            if isinstance(value, tuple):
                restored_tuple: list[Any] = []
                total = 0
                for item in value:
                    new_item, count = _restore(item)
                    restored_tuple.append(new_item)
                    total += count
                return tuple(restored_tuple), total
            return value, 0

        restored_args, count = _restore(dict(args))
        return restored_args, count

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def has_placeholders(self, text: str) -> bool:
        return PLACEHOLDER_RE.search(text) is not None

    def entries(self) -> list[VaultEntry]:
        with self._lock:
            return list(self._by_raw.values())

    def has_raw(self, raw_value: str) -> bool:
        """Return ``True`` when this raw value is already mapped."""
        with self._lock:
            return raw_value in self._by_raw

    def remove_raw(self, raw_value: str) -> bool:
        """Remove a raw mapping (used when a guarded write is blocked)."""
        with self._lock:
            entry = self._by_raw.pop(raw_value, None)
            if entry is None:
                return False
            self._by_placeholder.pop(entry.placeholder, None)
            return True

    def get_by_placeholder(self, placeholder: str) -> VaultEntry | None:
        with self._lock:
            return self._by_placeholder.get(placeholder)

    def stats(self) -> dict[str, Any]:
        """Return counts by category without exposing raw values."""
        with self._lock:
            entries = list(self._by_raw.values())

        by_category: dict[str, int] = {}
        redactions_by_category: dict[str, int] = {}
        secret_placeholders = 0
        pii_placeholders = 0
        total_secrets_redacted = 0
        total_pii_redacted = 0

        for entry in entries:
            by_category[entry.category] = by_category.get(entry.category, 0) + 1
            redactions_by_category[entry.category] = (
                redactions_by_category.get(entry.category, 0) + entry.redacted_count
            )
            if entry.is_pii:
                pii_placeholders += 1
                total_pii_redacted += entry.redacted_count
            else:
                secret_placeholders += 1
                total_secrets_redacted += entry.redacted_count

        return {
            "active_placeholders": len(entries),
            "secret_placeholders": secret_placeholders,
            "pii_placeholders": pii_placeholders,
            "total_secrets_redacted": total_secrets_redacted,
            "total_pii_redacted": total_pii_redacted,
            "by_category": by_category,
            "redactions_by_category": redactions_by_category,
            "entries": [
                {
                    "placeholder": entry.placeholder,
                    "category": entry.category,
                    "detector_source": entry.detector_source,
                    "redacted_count": entry.redacted_count,
                    "source_hint": entry.source_hint,
                }
                for entry in entries
            ],
        }

    def clear(self) -> None:
        """Remove all mappings (used by tests and session reset)."""
        with self._lock:
            self._by_raw.clear()
            self._by_placeholder.clear()
            self._secret_counter = 0
            self._pii_counter = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_raw)
