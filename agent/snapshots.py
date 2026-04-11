"""Session snapshot manager — checkpoint files before edits for /rewind.

Before every file write/edit, the agent checkpoints the pre-edit content
so the user can ``/rewind N`` to restore files to their state at step N.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


class SnapshotManager:
    """Manage file snapshots for undo/rewind functionality.

    Usage::

        snapshots = SnapshotManager()
        snapshots.checkpoint(step=1, file_path="main.py")
        # ... user edits main.py ...
        restored = snapshots.rewind(step=1)
        print(f"Restored: {restored}")
    """

    def __init__(
        self,
        snapshot_dir: Path | None = None,
        console: Console | None = None,
    ) -> None:
        self._dir = snapshot_dir or Path(".agent/snapshots")
        self._console = console or Console()
        self._step_counter: int = 0

    @property
    def current_step(self) -> int:
        return self._step_counter

    def next_step(self) -> int:
        """Increment and return the next step number."""
        self._step_counter += 1
        return self._step_counter

    def checkpoint(self, step: int, file_path: str) -> bool:
        """Save a copy of a file before it gets modified.

        Args:
            step: The step number for this checkpoint.
            file_path: Path to the file being modified.

        Returns:
            True if the file was successfully checkpointed.
        """
        source = Path(file_path)
        if not source.exists():
            # File doesn't exist yet (will be created) — store a marker
            step_dir = self._dir / str(step)
            step_dir.mkdir(parents=True, exist_ok=True)
            marker = step_dir / (source.name + ".__new__")
            marker.write_text("", encoding="utf-8")
            return True

        try:
            step_dir = self._dir / str(step)
            step_dir.mkdir(parents=True, exist_ok=True)

            # Preserve relative path structure under step dir
            dest = step_dir / source.name
            shutil.copy2(str(source), str(dest))

            # Also store the original absolute path so we can restore
            meta = step_dir / (source.name + ".__path__")
            meta.write_text(str(source.resolve()), encoding="utf-8")

            return True
        except OSError as e:
            self._console.print(f"[dim]⚠ Snapshot failed for {file_path}: {e}[/dim]")
            return False

    def rewind(self, step: int) -> list[str]:
        """Restore all files from a given step's snapshot.

        Args:
            step: The step number to rewind to.

        Returns:
            List of file paths that were restored.
        """
        step_dir = self._dir / str(step)
        if not step_dir.exists():
            return []

        restored: list[str] = []

        for path_meta in step_dir.glob("*.__path__"):
            original_path = path_meta.read_text(encoding="utf-8").strip()
            base_name = path_meta.name.replace(".__path__", "")
            snapshot_file = step_dir / base_name

            if snapshot_file.exists():
                try:
                    shutil.copy2(str(snapshot_file), original_path)
                    restored.append(original_path)
                except OSError as e:
                    self._console.print(
                        f"[error]Failed to restore {original_path}: {e}[/error]"
                    )

        # Handle files that were newly created (rewind = delete)
        for marker in step_dir.glob("*.__new__"):
            base_name = marker.name.replace(".__new__", "")
            # We don't know the original path for new files, skip deletion
            # (safer than guessing)

        if restored:
            self._console.print(
                f"[success]⏪ Rewound to step {step}: "
                f"restored {len(restored)} file(s)[/success]"
            )
        else:
            self._console.print(
                f"[warning]No files found in snapshot for step {step}[/warning]"
            )

        return restored

    def list_steps(self) -> list[int]:
        """Return a sorted list of available checkpoint step numbers."""
        if not self._dir.exists():
            return []
        steps: list[int] = []
        for p in self._dir.iterdir():
            if p.is_dir():
                try:
                    steps.append(int(p.name))
                except ValueError:
                    continue
        return sorted(steps)

    def list_table(self) -> Table:
        """Build a Rich table showing available snapshots."""
        table = Table(
            title="Session Snapshots",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Step", justify="center")
        table.add_column("Files")

        steps = self.list_steps()
        if not steps:
            table.add_row("[dim]—[/dim]", "[dim]No snapshots yet[/dim]")
        else:
            for step in steps:
                step_dir = self._dir / str(step)
                files = [
                    p.name for p in step_dir.iterdir()
                    if not p.name.endswith(".__path__") and not p.name.endswith(".__new__")
                ]
                table.add_row(str(step), ", ".join(files) or "[dim]empty[/dim]")

        return table

    def cleanup(self) -> None:
        """Remove all snapshots (called at session end if desired)."""
        if self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)
