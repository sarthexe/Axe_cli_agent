"""Project memory — persists key project facts across sessions.

Auto-detects the project's language, framework, test runner, and key
files on first run, then stores them in ``.agent/memory.json``.  The
agent injects this context into its system prompt at startup so it
knows the project from the start.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


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
        self._dir = memory_dir or Path(".agent")
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
        important_patterns = ["*.py", "*.ts", "*.js", "*.rs", "*.go"]
        skip_dirs = {".git", ".venv", "venv", "node_modules", "__pycache__", ".agent"}
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
        self._console.print("[dim]🧠 Project memory auto-detected and saved.[/dim]")

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
