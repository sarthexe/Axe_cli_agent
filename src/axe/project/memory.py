"""Project memory — persists key project facts across sessions.

Auto-detects the project's language, framework, test runner, and key
files on first run, then stores them in ``.axe/memory.json``.  The
agent injects this context into its system prompt at startup so it
knows the project from the start.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

_MAX_FAILED_ATTEMPTS = 10
_MAX_TURN_RULES = 12


# File patterns used for auto-detection
_DETECT_RULES: list[dict[str, Any]] = [
    # Python
    {"file": "pyproject.toml", "language": "python", "key_file": True},
    {"file": "setup.py",       "language": "python", "key_file": True},
    {"file": "requirements.txt", "language": "python"},
    {"file": "Pipfile",        "language": "python"},
    {"file": "pytest.ini",     "test_runner": "pytest"},
    {"file": "setup.cfg",      "test_runner": "pytest"},  # often has [tool:pytest]

    # JavaScript / TypeScript
    {"file": "package.json",   "language": "javascript", "key_file": True},
    {"file": "tsconfig.json",  "language": "typescript"},
    {"file": "jest.config.js", "test_runner": "jest"},
    {"file": "vitest.config.ts", "test_runner": "vitest"},

    # Rust
    {"file": "Cargo.toml",    "language": "rust", "key_file": True},

    # Go
    {"file": "go.mod",        "language": "go", "key_file": True},

    # Frameworks
    {"file": "next.config.js",    "framework": "Next.js"},
    {"file": "next.config.mjs",   "framework": "Next.js"},
    {"file": "vite.config.ts",    "framework": "Vite"},
    {"file": "django/settings.py","framework": "Django"},
    {"file": "manage.py",         "framework": "Django"},
    {"file": "app.py",            "framework": "Flask"},
]

# Common entry points to look for
_ENTRY_POINTS = [
    "main.py", "app.py", "index.ts", "index.js",
    "src/main.py", "src/index.ts", "src/main.ts",
    "src/index.js", "src/app.py", "manage.py",
]


class ProjectMemory:
    """Persist and retrieve project-level facts across sessions.

    Usage::

        mem = ProjectMemory()
        mem.load()
        if not mem.data:
            mem.auto_detect()
        system_context = mem.to_system_context()
    """

    def __init__(
        self,
        memory_dir: Path | None = None,
        console: Console | None = None,
    ) -> None:
        self._dir = memory_dir or Path(".axe")
        self._file = self._dir / "memory.json"
        self._console = console or Console()
        self.data: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        """Load memory from disk.  Returns empty dict if not found."""
        if self._file.exists():
            try:
                self.data = json.loads(self._file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.data = {}
        return self.data

    def save(self) -> None:
        """Persist current memory to disk."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def auto_detect(self, root: Path | None = None) -> None:
        """Scan the project root and populate memory with detected facts."""
        root = root or Path.cwd()

        detected: dict[str, Any] = {
            "language": None,
            "framework": None,
            "test_runner": None,
            "entry_point": None,
            "key_files": [],
            "notes": [],
        }

        # Scan for known files
        for rule in _DETECT_RULES:
            target = root / rule["file"]
            if target.exists():
                if "language" in rule and not detected["language"]:
                    detected["language"] = rule["language"]
                if "framework" in rule and not detected["framework"]:
                    detected["framework"] = rule["framework"]
                if "test_runner" in rule and not detected["test_runner"]:
                    detected["test_runner"] = rule["test_runner"]
                if rule.get("key_file"):
                    detected["key_files"].append(rule["file"])

        # Detect test runner from pyproject.toml if present
        pyproject = root / "pyproject.toml"
        if pyproject.exists() and not detected["test_runner"]:
            try:
                content = pyproject.read_text(encoding="utf-8")
                if "pytest" in content:
                    detected["test_runner"] = "pytest"
            except OSError:
                pass

        # Detect entry point
        for ep in _ENTRY_POINTS:
            if (root / ep).exists():
                detected["entry_point"] = ep
                break

        # Find other important files (up to 10)
        skip_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", ".axe", ".agent"}
        found_files: list[str] = []
        for p in sorted(root.rglob("*")):
            if any(skip in p.parts for skip in skip_dirs):
                continue
            if p.is_file() and p.suffix in (".py", ".ts", ".js", ".rs", ".go"):
                rel = str(p.relative_to(root))
                if rel not in detected["key_files"] and rel != detected["entry_point"]:
                    found_files.append(rel)
            if len(found_files) >= 10:
                break
        detected["key_files"].extend(found_files[:10])

        # Merge into existing data (don't overwrite user-set values)
        for key, value in detected.items():
            if value and key not in self.data:
                self.data[key] = value
            elif key == "key_files" and value:
                existing = set(self.data.get("key_files", []))
                existing.update(value)
                self.data["key_files"] = sorted(existing)

        self.save()
        self._console.print("[dim]Project memory auto-detected and saved.[/dim]")

    def to_system_context(self) -> str:
        """Return a formatted string for injection into the system prompt."""
        if not self.data:
            return ""

        parts: list[str] = ["PROJECT CONTEXT (from memory):"]

        if self.data.get("language"):
            parts.append(f"  Language: {self.data['language']}")
        if self.data.get("framework"):
            parts.append(f"  Framework: {self.data['framework']}")
        if self.data.get("test_runner"):
            parts.append(f"  Test runner: {self.data['test_runner']}")
        if self.data.get("entry_point"):
            parts.append(f"  Entry point: {self.data['entry_point']}")
        if self.data.get("key_files"):
            files_str = ", ".join(self.data["key_files"][:10])
            parts.append(f"  Key files: {files_str}")
        if self.data.get("notes"):
            for note in self.data["notes"]:
                parts.append(f"  Note: {note}")

        return "\n".join(parts)

    def update(self, key: str, value: str) -> None:
        """Set a key in memory and persist."""
        if key == "notes":
            # Append to notes list
            notes = self.data.get("notes", [])
            notes.append(value)
            self.data["notes"] = notes
        elif key == "key_files":
            # Append to key_files list
            files = self.data.get("key_files", [])
            if value not in files:
                files.append(value)
            self.data["key_files"] = files
        else:
            self.data[key] = value
        self.save()

    # ------------------------------------------------------------------
    # Scoped rules
    # ------------------------------------------------------------------

    def add_rule(self, pattern: str, rule: str) -> None:
        """Add a rule scoped to a glob pattern, extension, or tag.

        ``pattern`` may be a glob (``tests/*``, ``*.py``) or a tag such as
        ``security``. Rules only surface when a turn touches a matching file
        or command, so the system prompt never accumulates every rule.
        """
        text = rule.strip()
        if not text:
            return
        rules = self.data.setdefault("scoped_rules", [])
        entry = {"pattern": pattern.strip() or "*", "rule": text}
        if entry not in rules:
            rules.append(entry)
        self.save()

    def scoped_rules_for(
        self,
        paths: list[str] | None = None,
        command: str | None = None,
    ) -> list[str]:
        """Return rules relevant to the supplied paths/command."""
        raw_rules = self.data.get("scoped_rules", [])
        if not isinstance(raw_rules, list):
            return []
        results: list[str] = []
        haystacks = [*paths] if paths else []
        if command:
            haystacks.append(command)
        for entry in raw_rules:
            if isinstance(entry, str):
                pattern, rule = "*", entry
            elif isinstance(entry, dict):
                pattern = str(entry.get("pattern", "*"))
                rule = str(entry.get("rule", ""))
            else:
                continue
            if not rule or rule in results:
                continue
            if self._rule_matches(pattern, haystacks):
                results.append(rule)
            if len(results) >= _MAX_TURN_RULES:
                break
        return results

    @staticmethod
    def _rule_matches(pattern: str, haystacks: list[str]) -> bool:
        if pattern in {"", "*", "general"} or not haystacks:
            return pattern in {"", "*", "general"}
        lowered = pattern.lower()
        for raw in haystacks:
            candidate = str(raw).replace("\\", "/")
            name = candidate.rsplit("/", 1)[-1]
            if fnmatch.fnmatch(candidate.lower(), lowered) or fnmatch.fnmatch(
                name.lower(), lowered
            ):
                return True
            if lowered.startswith("*.") and name.lower().endswith(lowered[1:]):
                return True
            if lowered in candidate.lower():
                return True
        return False

    # ------------------------------------------------------------------
    # Failed-attempt memory (scoped ring buffer)
    # ------------------------------------------------------------------

    def record_failed_attempt(
        self,
        *,
        tool_name: str,
        target_file: str,
        error_summary: str,
        avoid_hint: str,
        command: str | None = None,
        files_touched: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record a failed tool/test attempt so it is not repeated blindly."""
        touched = {str(path) for path in (files_touched or []) if str(path)}
        if target_file:
            touched.add(str(target_file))
        entry: dict[str, Any] = {
            "tool_name": tool_name,
            "target_file": target_file,
            "error_summary": str(error_summary).strip()[:300],
            "avoid_hint": str(avoid_hint).strip()[:300],
        }
        if touched:
            entry["files_touched"] = sorted(touched)
        if command:
            entry["command"] = str(command).strip()[:300]
        attempts = self.data.setdefault("failed_attempts", [])
        if not isinstance(attempts, list):
            attempts = []
            self.data["failed_attempts"] = attempts
        # Drop an older duplicate for the same target/tool before appending.
        attempts[:] = [
            item
            for item in attempts
            if not (
                isinstance(item, dict)
                and item.get("tool_name") == tool_name
                and item.get("target_file") == target_file
                and item.get("error_summary") == entry["error_summary"]
            )
        ]
        attempts.append(entry)
        del attempts[:-_MAX_FAILED_ATTEMPTS]
        self.save()
        return entry

    def failed_attempts_for(
        self,
        paths: list[str] | None = None,
        command: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return prior failures that match the current files or command."""
        attempts = self.data.get("failed_attempts", [])
        if not isinstance(attempts, list):
            return []
        paths = paths or []
        results: list[dict[str, Any]] = []
        for entry in reversed(attempts):
            if not isinstance(entry, dict):
                continue
            target = str(entry.get("target_file", ""))
            stored_command = str(entry.get("command", ""))
            matched = self._target_matches(target, paths)
            if not matched:
                touched = entry.get("files_touched", [])
                if isinstance(touched, list):
                    matched = any(
                        self._target_matches(str(item), paths) for item in touched
                    )
            if not matched and command and stored_command:
                matched = (
                    command.strip() == stored_command.strip()
                    or stored_command.strip() in command
                )
            if matched:
                results.append(entry)
            if len(results) >= _MAX_FAILED_ATTEMPTS:
                break
        return results

    @staticmethod
    def _target_matches(target: str, paths: list[str]) -> bool:
        if not target:
            return False
        normalized = target.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        for raw in paths:
            candidate = str(raw).replace("\\", "/").lower()
            candidate_name = candidate.rsplit("/", 1)[-1]
            if normalized == candidate or name == candidate_name:
                return True
            if candidate.endswith(normalized) or normalized.endswith(candidate):
                return True
        return False

    def clear_failed_attempts(self) -> None:
        """Remove all stored failure traces."""
        self.data["failed_attempts"] = []
        self.save()

    def to_turn_context(
        self,
        paths: list[str] | None = None,
        command: str | None = None,
    ) -> str:
        """Build a compact per-turn context block (rules + failure hints)."""
        rules = self.scoped_rules_for(paths, command)
        failures = self.failed_attempts_for(paths, command)
        if not rules and not failures:
            return ""

        parts: list[str] = []
        if rules:
            parts.append("<ScopedRules>")
            parts.extend(f"- {rule}" for rule in rules)
            parts.append("</ScopedRules>")

        if failures:
            parts.append("<AvoidRepeatingFailures>")
            for entry in failures:
                target = entry.get("target_file", "unknown")
                tool = entry.get("tool_name", "tool")
                summary = entry.get("error_summary", "")
                hint = entry.get("avoid_hint", "")
                line = f"- {tool} failed on {target}"
                touched = entry.get("files_touched", [])
                if isinstance(touched, list) and len(touched) > 1:
                    line += f" (touched: {', '.join(str(p) for p in touched[:4])})"
                if summary:
                    line += f": {summary}"
                if hint:
                    line += f" Avoid: {hint}"
                parts.append(line)
            parts.append(
                "Do not repeat these failed attempts; change the approach first."
            )
            parts.append("</AvoidRepeatingFailures>")

        return "\n".join(parts)

    def remove(self, key: str) -> bool:
        """Remove a key from memory.  Returns True if the key existed."""
        if key in self.data:
            del self.data[key]
            self.save()
            return True
        return False

    def display_table(self) -> Table:
        """Build a Rich table showing current memory contents."""
        table = Table(
            title="Project Memory",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=False,
        )
        table.add_column("Key", style="bold")
        table.add_column("Value")

        if not self.data:
            table.add_row("[dim]empty[/dim]", "[dim]Run auto-detect or set values manually[/dim]")
        else:
            for key, value in sorted(self.data.items()):
                if isinstance(value, list):
                    display = ", ".join(str(v) for v in value[:10])
                    if len(value) > 10:
                        display += f" (+{len(value) - 10} more)"
                else:
                    display = str(value)
                table.add_row(key, display)

        return table
