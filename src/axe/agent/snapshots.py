"""Turn-level snapshots, speculative verification, and instant rollback."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

_MANIFEST = "manifest.json"


@dataclass(slots=True)
class SnapshotFile:
    """One file captured before a turn's modifications."""

    path: str
    existed: bool
    stored_name: str | None = None
    sha256: str = ""
    size: int = 0
    content: str | None = None


@dataclass(slots=True)
class TurnSnapshot:
    """A lightweight snapshot of one file-modifying turn."""

    turn_id: int
    timestamp: str
    files: list[SnapshotFile] = field(default_factory=list)
    git_diff_summary: str = ""
    rolled_back: bool = False

    @property
    def paths(self) -> list[str]:
        return [item.path for item in self.files]


@dataclass(slots=True)
class RollbackReport:
    """Result of restoring a turn snapshot."""

    turn_id: int | None = None
    restored_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    success: bool = False
    message: str = ""

    @property
    def changed_files(self) -> list[str]:
        return [*self.restored_files, *self.deleted_files]


@dataclass(slots=True)
class SpeculativeResult:
    """Outcome of a speculative verification command."""

    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""
    rolled_back: bool = False
    rollback: RollbackReport | None = None


class SnapshotManager:
    """Capture pre-edit file content per turn and restore it on demand."""

    def __init__(
        self,
        snapshot_dir: Path | None = None,
        console: Console | None = None,
        *,
        cwd: Path | None = None,
    ) -> None:
        self._dir = snapshot_dir or Path(".axe/snapshots")
        self._console = console or Console()
        self._cwd = Path(cwd).resolve() if cwd is not None else Path.cwd()
        self._step_counter: int = 0
        self._current_turn_id: int | None = None
        self._snapshots: dict[int, TurnSnapshot] = {}
        self._load_snapshots()

    # ------------------------------------------------------------------
    # Compatibility properties / creation
    # ------------------------------------------------------------------

    @property
    def current_step(self) -> int:
        return self._step_counter

    @property
    def current_turn_id(self) -> int | None:
        return self._current_turn_id

    def begin_turn(self) -> int:
        """Start a new file-modifying turn and return its ID."""
        self._step_counter += 1
        turn_id = self._step_counter
        snapshot = TurnSnapshot(
            turn_id=turn_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_diff_summary=self._git_diff_summary(),
        )
        self._snapshots[turn_id] = snapshot
        self._current_turn_id = turn_id
        self._save_manifest(snapshot)
        return turn_id

    def next_step(self) -> int:
        """Backward-compatible alias for :meth:`begin_turn`."""
        return self.begin_turn()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def checkpoint(self, step: int, file_path: str) -> bool:
        """Capture a file's pre-edit state for the given turn."""
        snapshot = self._snapshots.get(step)
        if snapshot is None:
            snapshot = TurnSnapshot(
                turn_id=step,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self._snapshots[step] = snapshot
            self._step_counter = max(self._step_counter, step)

        source = Path(file_path).expanduser()
        if not source.is_absolute():
            source = (self._cwd / source).resolve()
        else:
            source = source.resolve()
        path_key = str(source)

        if any(item.path == path_key for item in snapshot.files):
            return True

        turn_dir = self._turn_dir(step)
        turn_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"file_{len(snapshot.files):04d}"
        existed = source.exists() and source.is_file()
        sha256 = ""
        size = 0
        content: str | None = None

        if existed:
            try:
                data = source.read_bytes()
            except OSError as exc:
                self._console.print(
                    f"[dim]Snapshot failed for {file_path}: {exc}[/dim]"
                )
                return False
            sha256 = hashlib.sha256(data).hexdigest()
            size = len(data)
            try:
                content = data.decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - defensive
                content = None
            try:
                (turn_dir / stored_name).write_bytes(data)
            except OSError as exc:
                self._console.print(
                    f"[dim]Snapshot failed for {file_path}: {exc}[/dim]"
                )
                return False
        else:
            stored_name = None

        snapshot.files.append(
            SnapshotFile(
                path=path_key,
                existed=existed,
                stored_name=stored_name,
                sha256=sha256,
                size=size,
                content=content,
            )
        )
        self._save_manifest(snapshot)
        return True

    # ------------------------------------------------------------------
    # Rollback / listing
    # ------------------------------------------------------------------

    def rollback_last(self) -> RollbackReport:
        """Restore the most recent non-rolled-back turn."""
        candidates = [
            snapshot
            for snapshot in self._snapshots.values()
            if not snapshot.rolled_back
        ]
        if not candidates:
            return RollbackReport(success=False, message="No snapshots available.")
        snapshot = max(candidates, key=lambda item: item.turn_id)
        return self._restore_snapshot(snapshot)

    def rewind(self, step: int) -> list[str]:
        """Restore a specific turn (backward-compatible ``/rewind`` API)."""
        snapshot = self._snapshots.get(step)
        if snapshot is None:
            snapshot = self._load_snapshot(step)
        if snapshot is None:
            return self._rewind_legacy(step)
        report = self._restore_snapshot(snapshot)
        return report.changed_files

    def _restore_snapshot(self, snapshot: TurnSnapshot) -> RollbackReport:
        turn_dir = self._turn_dir(snapshot.turn_id)
        restored: list[str] = []
        deleted: list[str] = []

        for item in snapshot.files:
            target = Path(item.path)
            if item.existed:
                if not item.stored_name:
                    continue
                stored = turn_dir / item.stored_name
                if not stored.exists():
                    continue
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(stored, target)
                    restored.append(str(target))
                except OSError as exc:
                    self._console.print(
                        f"[error]Failed to restore {target}: {exc}[/error]"
                    )
            else:
                try:
                    if target.exists():
                        target.unlink()
                        deleted.append(str(target))
                except OSError as exc:
                    self._console.print(
                        f"[error]Failed to remove {target}: {exc}[/error]"
                    )

        snapshot.rolled_back = True
        self._save_manifest(snapshot)
        self._current_turn_id = self._latest_open_turn_id()
        report = RollbackReport(
            turn_id=snapshot.turn_id,
            restored_files=restored,
            deleted_files=deleted,
            success=True,
            message=(
                f"Restored {len(restored)} file(s)"
                + (f" and removed {len(deleted)} new file(s)" if deleted else "")
                + "."
            ),
        )
        if restored or deleted:
            self._console.print(f"[success]Undo turn {snapshot.turn_id}: {report.message}[/success]")
        else:
            self._console.print(
                f"[warning]Turn {snapshot.turn_id} had no changed files to restore.[/warning]"
            )
        return report

    def list_snapshots(self) -> list[TurnSnapshot]:
        self._load_snapshots()
        return sorted(self._snapshots.values(), key=lambda item: item.turn_id)

    def list_steps(self) -> list[int]:
        return [snapshot.turn_id for snapshot in self.list_snapshots()]

    def list_table(self) -> Table:
        table = Table(
            title="Turn Snapshots",
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
        )
        table.add_column("Turn", justify="center")
        table.add_column("Time")
        table.add_column("Files")
        table.add_column("Status", justify="center")

        snapshots = self.list_snapshots()
        if not snapshots:
            table.add_row("[dim]—[/dim]", "[dim]No snapshots yet[/dim]", "", "")
        else:
            for snapshot in snapshots[-15:]:
                files = ", ".join(Path(path).name for path in snapshot.paths) or "empty"
                status = "undone" if snapshot.rolled_back else "available"
                table.add_row(
                    str(snapshot.turn_id),
                    snapshot.timestamp.replace("T", " ")[:19],
                    files,
                    status,
                )
        return table

    # ------------------------------------------------------------------
    # Speculative verification
    # ------------------------------------------------------------------

    def run_speculative_check(
        self,
        command: str = "pytest -q",
        *,
        rollback_on_failure: bool = True,
        cwd: Path | None = None,
        timeout: float = 180.0,
    ) -> SpeculativeResult:
        """Run a verification command and roll back on failure if requested."""
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd or self._cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            returncode = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            returncode = 124
            stdout = exc.stdout or "" if isinstance(exc.stdout, str) else ""
            stderr = f"Speculative command timed out after {timeout}s."
        except OSError as exc:
            returncode = 1
            stdout = ""
            stderr = f"Speculative command failed to start: {exc}"

        report: RollbackReport | None = None
        rolled_back = False
        if returncode != 0 and rollback_on_failure:
            report = self.rollback_last()
            rolled_back = report.success

        return SpeculativeResult(
            command=command,
            returncode=returncode,
            stdout=stdout[-20_000:],
            stderr=stderr[-20_000:],
            rolled_back=rolled_back,
            rollback=report,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _turn_dir(self, turn_id: int) -> Path:
        return self._dir / str(turn_id)

    def _manifest_path(self, turn_id: int) -> Path:
        return self._turn_dir(turn_id) / _MANIFEST

    def _save_manifest(self, snapshot: TurnSnapshot) -> None:
        turn_dir = self._turn_dir(snapshot.turn_id)
        turn_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "turn_id": snapshot.turn_id,
            "timestamp": snapshot.timestamp,
            "git_diff_summary": snapshot.git_diff_summary,
            "rolled_back": snapshot.rolled_back,
            "files": [
                {
                    "path": item.path,
                    "existed": item.existed,
                    "stored_name": item.stored_name,
                    "sha256": item.sha256,
                    "size": item.size,
                }
                for item in snapshot.files
            ],
        }
        try:
            self._manifest_path(snapshot.turn_id).write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    def _load_snapshots(self) -> None:
        if not self._dir.exists():
            return
        for directory in sorted(self._dir.iterdir()):
            if not directory.is_dir():
                continue
            try:
                turn_id = int(directory.name)
            except ValueError:
                continue
            if turn_id in self._snapshots:
                continue
            snapshot = self._load_snapshot(turn_id)
            if snapshot is not None:
                self._snapshots[turn_id] = snapshot
                self._step_counter = max(self._step_counter, turn_id)

    def _load_snapshot(self, turn_id: int) -> TurnSnapshot | None:
        manifest = self._manifest_path(turn_id)
        if not manifest.exists():
            return None
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        files: list[SnapshotFile] = []
        for item in data.get("files", []):
            if not isinstance(item, dict):
                continue
            stored_name = item.get("stored_name")
            content: str | None = None
            if stored_name:
                stored = self._turn_dir(turn_id) / str(stored_name)
                if stored.exists():
                    try:
                        content = stored.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )
                    except OSError:
                        content = None
            files.append(
                SnapshotFile(
                    path=str(item.get("path", "")),
                    existed=bool(item.get("existed")),
                    stored_name=str(stored_name) if stored_name else None,
                    sha256=str(item.get("sha256", "")),
                    size=int(item.get("size", 0) or 0),
                    content=content,
                )
            )
        return TurnSnapshot(
            turn_id=int(data.get("turn_id", turn_id)),
            timestamp=str(data.get("timestamp", "")),
            files=files,
            git_diff_summary=str(data.get("git_diff_summary", "")),
            rolled_back=bool(data.get("rolled_back", False)),
        )

    def _latest_open_turn_id(self) -> int | None:
        open_turns = [
            snapshot.turn_id
            for snapshot in self._snapshots.values()
            if not snapshot.rolled_back
        ]
        return max(open_turns) if open_turns else None

    def _git_diff_summary(self) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self._cwd), "diff", "--stat"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return (completed.stdout or "").strip()[-2000:]

    def _rewind_legacy(self, step: int) -> list[str]:
        """Restore snapshots created by the pre-Phase-5 format."""
        step_dir = self._dir / str(step)
        if not step_dir.exists():
            return []
        restored: list[str] = []
        for path_meta in step_dir.glob("*.__path__"):
            try:
                original_path = path_meta.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            base_name = path_meta.name.replace(".__path__", "")
            snapshot_file = step_dir / base_name
            if snapshot_file.exists():
                try:
                    shutil.copy2(str(snapshot_file), original_path)
                    restored.append(original_path)
                except OSError as exc:
                    self._console.print(
                        f"[error]Failed to restore {original_path}: {exc}[/error]"
                    )
        return restored

    def cleanup(self) -> None:
        if self._dir.exists():
            shutil.rmtree(self._dir, ignore_errors=True)
        self._snapshots.clear()
        self._step_counter = 0
        self._current_turn_id = None
