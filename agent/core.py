"""Core agent loop logic."""

from __future__ import annotations

import difflib
import re
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from agent.tokens import count_tokens
from config.settings import Settings
from cost.pricing import cost_for
from llm.provider import LLMProvider
from sandbox.permissions import CommandSafety, SafetyDecision
from tools.registry import ToolRegistry

if TYPE_CHECKING:
    from cost.tracker import CostTracker
    from agent.context import ContextManager
    from agent.snapshots import SnapshotManager
    from utils.logger import SessionLogger


class Agent:
    """Autonomous agent that executes tools in a loop until finished."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        settings: Settings,
        console: Console | None = None,
        tracker: CostTracker | None = None,
        context_manager: ContextManager | None = None,
        snapshot_manager: SnapshotManager | None = None,
        command_safety: CommandSafety | None = None,
        dry_run: bool = False,
        session_logger: SessionLogger | None = None,
        project_context: str = "",
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.settings = settings
        self.console = console or Console()
        self.tracker = tracker
        self.context_manager = context_manager
        self.snapshot_manager = snapshot_manager
        self.command_safety = command_safety
        self.dry_run = dry_run
        self.session_logger = session_logger
        self.tool_failures: dict[str, int] = {}
        self._must_ack_non_execution: bool = False
        self._last_non_execution_note: str = ""
        self._destructive_declined_in_run: bool = False

        # Build system prompt with optional project context
        system_content = (
            "You are Axe, an autonomous CLI coding agent. You solve tasks by using tools — "
            "not by asking the user questions.\n\n"
            "RULES:\n"
            "1. ACT, don't ask. Never say 'should I?', 'would you like me to?', or 'shall I proceed?'. "
            "Just do it.\n"
            "2. If a dependency is missing, install it yourself and continue.\n"
            "3. If a command fails, read the error, fix the root cause, retry.\n"
            "4. ALWAYS read a file before editing it. Never guess contents.\n"
            "5. ALWAYS run code after changes to verify it works.\n"
            "6. Use grep_search and glob_search to find files — don't guess paths.\n"
            "7. After 3 failed attempts at the same fix, stop and explain what went wrong.\n"
            "8. When the task is done, give a short summary of what you did. No tool calls.\n"
            "9. When reading files, read at least 200 lines at a time (use start_line/end_line). "
            "Never read a file 40 lines at a time — it wastes iterations. "
            "If a file is under 500 lines, read the whole thing in one call.\n"
            "10. If the user declines a destructive command, do not try alternative destructive methods. "
            "Report non-execution honestly and stop destructive attempts.\n"
        )
        if project_context:
            system_content += f"\n{project_context}\n"

        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

    def run(self, prompt: str) -> str:
        """Run the agent core loop for a user prompt."""
        self._destructive_declined_in_run = False
        self.messages.append({"role": "user", "content": prompt})

        for iteration in range(self.settings.agent.max_iterations):
            schemas = self.registry.get_all_schemas()

            # --- Context management: trim messages to fit token budget ---
            if self.context_manager is not None:
                prepared_messages = self.context_manager.prepare(self.messages)
            else:
                prepared_messages = self.messages

            # Determine display label (router-aware or plain model name)
            model_label = self._model_label()

            llm_started_at = time.perf_counter()
            with self.console.status(f"[bold blue]Calling {model_label}...[/bold blue]"):
                try:
                    response = self.provider.complete(
                        prompt="",  # Ignored because messages is passed
                        messages=prepared_messages,
                        tools=schemas if schemas else None,
                    )
                except Exception as exc:
                    llm_duration_ms = int((time.perf_counter() - llm_started_at) * 1000)
                    self._log_llm_event(
                        event_type="llm_call",
                        status="error",
                        model=model_label,
                        duration_ms=llm_duration_ms,
                        error=str(exc),
                    )
                    raise

            llm_duration_ms = int((time.perf_counter() - llm_started_at) * 1000)
            usage = response.raw.get("usage") if response.raw else None

            # --- Cost tracking ---
            if self.tracker is not None:
                self.tracker.record(
                    model=getattr(self.provider, "current_model", getattr(self.provider, "model", "unknown")),
                    usage=usage,
                )
                if self.tracker.near_budget() and not self.tracker.over_budget():
                    pct = self.tracker.budget_percent()
                    self.console.print(
                        f"[bold yellow]⚠  Budget warning:[/bold yellow] "
                        f"[yellow]{pct:.1f}% of ${self.tracker.session_budget:.2f} used[/yellow]"
                    )
                if self.tracker.over_budget():
                    self.console.print(
                        f"[bold red]🛑 Session budget of ${self.tracker.session_budget:.2f} exceeded "
                        f"(${self.tracker.total_cost():.4f} spent). Stopping.[/bold red]"
                    )
                    return "Error: Session budget exceeded."
                self.console.print(self._cost_footer())

            self._log_llm_event(
                event_type="llm_call",
                status="success",
                model=model_label,
                duration_ms=llm_duration_ms,
                usage=usage,
                tool_calls=len(response.tool_calls),
            )

            # Record assistant generation
            assistant_msg: dict[str, Any] = {}
            if response.raw and "message" in response.raw:
                assistant_msg = response.raw["message"]
            else:
                assistant_msg = {"role": "assistant", "content": response.text}

            self.messages.append(assistant_msg)

            if not response.tool_calls:
                if self._must_ack_non_execution and not self._acknowledges_non_execution(response.text):
                    self._inject_non_execution_correction(response.text)
                    continue
                if response.text:
                    self.console.print(self._badge())
                    self.console.print(Markdown(response.text))
                self._must_ack_non_execution = False
                self._last_non_execution_note = ""
                return response.text

            if response.text:
                self.console.print(self._badge())
                self.console.print(Markdown(response.text))

            # Execute tools sequentially
            for tool_call in response.tool_calls:
                args_str = str(tool_call.arguments)
                if len(args_str) > 100:
                    args_str = args_str[:97] + "..."

                tool_label = f"{tool_call.name}({args_str})"
                self.console.print(f"[dim]→ Tool call:[/dim] {tool_label}")

                # --- Snapshot before file modifications ---
                if (
                    not self.dry_run
                    and self.snapshot_manager is not None
                    and tool_call.name in ("file_edit", "file_write")
                ):
                    file_path = tool_call.arguments.get("path", "")
                    if file_path:
                        step = self.snapshot_manager.next_step()
                        self.snapshot_manager.checkpoint(step, file_path)

                tool_started_at = time.perf_counter()
                with self.console.status(f"[bold cyan]Running {tool_call.name}...[/bold cyan]"):
                    result = self._execute_tool_call(tool_call.name, tool_call.arguments)

                tool_duration_ms = int((time.perf_counter() - tool_started_at) * 1000)
                tool_status = self._tool_status(result)
                self._log_tool_event(
                    tool_name=tool_call.name,
                    status=tool_status,
                    duration_ms=tool_duration_ms,
                    arguments=tool_call.arguments,
                )

                is_error = tool_status == "error"
                if is_error:
                    self.tool_failures[tool_call.name] = self.tool_failures.get(tool_call.name, 0) + 1
                    failures = self.tool_failures[tool_call.name]
                    if failures >= self.settings.agent.max_retries_per_tool:
                        result += f"\n\n[SYSTEM INTERVENTION] Tool '{tool_call.name}' has failed {failures} consecutive times. Stop retrying the same approach. Reflect on the error and try a completely different strategy, or ask the user for help."
                else:
                    self.tool_failures[tool_call.name] = 0

                # Append tool result to messages
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                if tool_status in {"blocked", "skipped", "dry_run"}:
                    self._must_ack_non_execution = True
                    self._last_non_execution_note = result.splitlines()[0]
                self._print_tool_result(result)

        self.console.print(f"[bold red]Stopped:[/bold red] Hit max iterations ({self.settings.agent.max_iterations}).")
        return "Error: Agent reached maximum iterations without completing the task."

    def clear_conversation(self) -> None:
        """Reset conversation history but keep the system prompt."""
        system_msg = self.messages[0] if self.messages else None
        self.messages.clear()
        if system_msg:
            self.messages.append(system_msg)
        self.tool_failures.clear()
        self._must_ack_non_execution = False
        self._last_non_execution_note = ""
        self._destructive_declined_in_run = False

    def explain(self, task: str) -> str:
        """Return a planning preview and cost estimate without tool execution."""
        task = task.strip()
        if not task:
            return "Usage: /explain <task>"

        explain_prompt = (
            "Produce an execution plan before running anything.\n"
            "Output a numbered list of concrete steps the agent would take.\n"
            "Keep it concise and practical."
            f"\n\nTask: {task}"
        )

        model_label = self._model_label()
        started_at = time.perf_counter()
        with self.console.status(f"[bold blue]Planning with {model_label}...[/bold blue]"):
            response = self.provider.complete(
                prompt=explain_prompt,
                system_prompt="You are a careful CLI planning assistant.",
            )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        usage = response.raw.get("usage") if response.raw else None

        if self.tracker is not None:
            self.tracker.record(
                model=getattr(self.provider, "current_model", getattr(self.provider, "model", "unknown")),
                usage=usage,
            )
            self.console.print(self._cost_footer())

        self._log_llm_event(
            event_type="llm_explain",
            status="success",
            model=model_label,
            duration_ms=duration_ms,
            usage=usage,
            tool_calls=0,
        )

        plan_text = response.text.strip() or f"1. Analyze the task: {task}\n2. Execute the plan."
        if not re.search(r"(?m)^\s*\d+\.\s+", plan_text):
            plan_text = f"1. {plan_text}"

        estimate = self._estimate_execution_cost(task=task, plan_text=plan_text, usage=usage)
        return f"{plan_text}\n\nEstimated execution cost (heuristic):\n{estimate}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _model_label(self) -> str:
        """Return a plain-text model name for status spinners."""
        return getattr(self.provider, "current_model", getattr(self.provider, "model", "model"))

    def _badge(self) -> str:
        """Return a Rich markup string badge for the active model/tier."""
        badge_fn = getattr(self.provider, "badge_text", None)
        if badge_fn is not None:
            return badge_fn()  # returns Rich Text
        model = getattr(self.provider, "model", "model")
        return f"[bold blue][{model}][/bold blue]"

    def _execute_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "shell":
            return self._handle_shell(arguments)
        if self.dry_run and name == "file_write":
            return self._dry_run_file_write(arguments)
        if self.dry_run and name == "file_edit":
            return self._dry_run_file_edit(arguments)
        try:
            return self.registry.dispatch(name, arguments)
        except Exception as exc:
            return f"Error: {exc}"

    def _handle_shell(self, arguments: dict[str, Any]) -> str:
        command = str(arguments.get("command", "")).strip()
        if not command:
            return "Error: Missing required argument 'command'."

        decision = self.command_safety.analyze(command) if self.command_safety else SafetyDecision()
        if decision.blocked:
            return CommandSafety.blocked_message(command, decision)
        if self._destructive_declined_in_run and decision.needs_confirmation:
            return (
                "[BLOCKED] Destructive command blocked because a previous destructive "
                "command was declined by the user in this task."
            )

        if self.dry_run:
            if decision.needs_confirmation:
                return (
                    f"[DRY-RUN] Command requires confirmation and was not executed:\n"
                    f"{command}"
                )
            return f"[DRY-RUN] Would run shell command:\n{command}"

        if decision.needs_confirmation:
            approved = self._confirm_shell_command(command)
            if approved is None:
                self._destructive_declined_in_run = True
                return f"[SKIPPED] User declined shell command: {command}"
            command = approved
            arguments["command"] = command

            post_edit = self.command_safety.analyze(command) if self.command_safety else SafetyDecision()
            if post_edit.blocked:
                return CommandSafety.blocked_message(command, post_edit)

        return self.registry.dispatch("shell", arguments)

    def _confirm_shell_command(self, command: str) -> str | None:
        current = command
        while True:
            self.console.print(
                f"[bold yellow]Confirmation required[/bold yellow] for command:\n[dim]{current}[/dim]"
            )
            try:
                choice = self.console.input("[bold cyan][y/n/edit][/bold cyan] ").strip().lower()
            except (EOFError, KeyboardInterrupt, OSError):
                return None

            if choice in {"y", "yes"}:
                return current
            if choice in {"n", "no"}:
                return None
            if choice in {"edit", "e"}:
                try:
                    edited = self.console.input("[bold cyan]Edit command:[/bold cyan] ").strip()
                except (EOFError, KeyboardInterrupt, OSError):
                    return None
                if not edited:
                    self.console.print("[yellow]Edited command cannot be empty.[/yellow]")
                    continue
                if self.command_safety is not None:
                    decision = self.command_safety.analyze(edited)
                    if decision.blocked:
                        self.console.print(CommandSafety.blocked_message(edited, decision))
                        continue
                    if not decision.needs_confirmation:
                        return edited
                current = edited
                continue

            self.console.print("[yellow]Please enter y, n, or edit.[/yellow]")

    def _dry_run_file_write(self, arguments: dict[str, Any]) -> str:
        path = str(arguments.get("path", "")).strip()
        content = arguments.get("content")
        if not path:
            return "Error: File path must be provided."
        if not isinstance(content, str):
            return "Error: File content must be a string."
        return (
            f"[DRY-RUN] Would write {len(content)} characters to {path}. "
            "No file was modified."
        )

    def _dry_run_file_edit(self, arguments: dict[str, Any]) -> str:
        path = str(arguments.get("path", "")).strip()
        target = arguments.get("target")
        replacement = arguments.get("replacement")
        if not path or target is None or replacement is None:
            return "Error: Missing required arguments."

        file_path = Path(path)
        if not file_path.exists():
            return f"Error: File '{path}' does not exist."
        if not file_path.is_file():
            return f"Error: '{path}' is not a file."

        try:
            original = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"Error: File '{path}' appears to be binary or has non-UTF-8 encoding."
        except OSError as exc:
            return f"Error reading file '{path}': {exc}"

        count = original.count(str(target))
        if count == 0:
            return (
                "Error: The target string was not found in the file. "
                "Ensure you have the exact match, including all whitespace."
            )
        if count > 1:
            return (
                f"Error: The target string was found {count} times. "
                "The target must be unique to ensure safe editing."
            )

        updated = original.replace(str(target), str(replacement))
        diff_text = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        return (
            f"[DRY-RUN] Would edit {path}. No file was modified.\n\n"
            f"Diff:\n```diff\n{diff_text}```"
        )

    def _print_tool_result(self, result: str) -> None:
        diff_match = re.search(r"```diff\n(.*?)```", result, flags=re.DOTALL)
        self.console.print("[dim]← Tool result:[/dim]")
        if diff_match:
            prefix = result[:diff_match.start()].strip()
            suffix = result[diff_match.end():].strip()
            if prefix:
                self.console.print(prefix)
            diff_block = diff_match.group(1).rstrip()
            syntax = Syntax(diff_block, "diff", theme="ansi_dark", line_numbers=False)
            self.console.print(Panel(syntax, title="Diff", border_style="green"))
            if suffix:
                self.console.print(suffix)
            return

        display_result = result
        lines = display_result.splitlines()
        if len(lines) > 15:
            display_result = "\n".join(lines[:15]) + "\n...[truncated output for display]"
        self.console.print(display_result)

    def _tool_status(self, result: str) -> str:
        if result.startswith("[BLOCKED]"):
            return "blocked"
        if result.startswith("[SKIPPED]"):
            return "skipped"
        if result.startswith("[DRY-RUN]"):
            return "dry_run"
        if self._is_error_result(result):
            return "error"
        return "success"

    @staticmethod
    def _is_error_result(result: str) -> bool:
        return (
            result.startswith("Error:")
            or "[ERROR]" in result
            or "Command exited with code" in result
        )

    def _cost_footer(self) -> str:
        if self.tracker is None:
            return ""
        used = self.tracker.total_cost()
        budget = self.tracker.session_budget
        pct = min(max(self.tracker.budget_percent(), 0.0), 100.0)
        bar_width = 20
        filled = int((pct / 100.0) * bar_width)
        bar = ("#" * filled) + ("-" * (bar_width - filled))
        style = "green"
        if self.tracker.over_budget():
            style = "bold red"
        elif self.tracker.near_budget():
            style = "yellow"
        return f"[{style}]Cost [{bar}] ${used:.4f}/${budget:.2f} ({pct:.1f}%)[/{style}]"

    def _estimate_execution_cost(
        self,
        *,
        task: str,
        plan_text: str,
        usage: dict[str, Any] | None,
    ) -> str:
        model = getattr(self.provider, "current_model", getattr(self.provider, "model", "gpt-4.1-mini"))
        step_count = len(re.findall(r"(?m)^\s*\d+\.\s+", plan_text)) or 3
        expected_calls = max(3, min(12, step_count * 2))

        prompt_tokens = int(usage.get("prompt_tokens", 0)) if usage else 0
        completion_tokens = int(usage.get("completion_tokens", 0)) if usage else 0
        if prompt_tokens == 0:
            prompt_tokens = max(250, count_tokens(task) * 3)
        if completion_tokens == 0:
            completion_tokens = max(150, count_tokens(plan_text))

        baseline = cost_for(model, prompt_tokens, completion_tokens)
        low = baseline * max(2, expected_calls // 2)
        expected = baseline * expected_calls
        high = baseline * (expected_calls + 4)
        return (
            f"1. Low: ${low:.4f}\n"
            f"2. Expected: ${expected:.4f}\n"
            f"3. High: ${high:.4f}\n"
            f"4. Basis: {expected_calls} model calls on {model}"
        )

    def _acknowledges_non_execution(self, text: str) -> bool:
        lowered = text.lower()
        required_signals = (
            "did not execute",
            "didn't execute",
            "not executed",
            "was not executed",
            "not run",
            "wasn't run",
            "skipped",
            "blocked",
            "declined",
            "dry-run",
            "dry run",
            "refused",
        )
        return any(signal in lowered for signal in required_signals)

    def _inject_non_execution_correction(self, bad_response: str) -> None:
        note = self._last_non_execution_note or "A prior command was not executed."
        self.messages.append({
            "role": "system",
            "content": (
                "SAFETY CORRECTION: The previous tool result shows non-execution. "
                "Do not claim success for actions that were blocked/skipped/dry-run. "
                "Your next response must explicitly state non-execution and reflect the exact tool result.\n"
                f"Tool note: {note}\n"
                f"Incorrect response to correct: {bad_response}"
            ),
        })

    def _log_llm_event(
        self,
        *,
        event_type: str,
        status: str,
        model: str,
        duration_ms: int,
        usage: dict[str, Any] | None = None,
        tool_calls: int = 0,
        error: str = "",
    ) -> None:
        if self.session_logger is None:
            return
        usage = usage or {}
        self.session_logger.log_event(
            event_type=event_type,
            status=status,
            model=model,
            duration_ms=duration_ms,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            tool_calls=tool_calls,
            error=error,
            dry_run=self.dry_run,
        )

    def _log_tool_event(
        self,
        *,
        tool_name: str,
        status: str,
        duration_ms: int,
        arguments: dict[str, Any],
    ) -> None:
        if self.session_logger is None:
            return
        safe_args = dict(arguments)
        if "content" in safe_args and isinstance(safe_args["content"], str):
            safe_args["content"] = f"<{len(safe_args['content'])} chars>"
        self.session_logger.log_event(
            event_type="tool_call",
            status=status,
            tool=tool_name,
            duration_ms=duration_ms,
            arguments=safe_args,
            dry_run=self.dry_run,
        )
