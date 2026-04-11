"""Core agent loop logic."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from rich.console import Console
from rich.markdown import Markdown

from config.settings import Settings
from llm.provider import LLMProvider
from tools.registry import ToolRegistry

if TYPE_CHECKING:
    from llm.router import ModelRouter
    from cost.tracker import CostTracker


class Agent:
    """Autonomous agent that executes tools in a loop until finished."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        settings: Settings,
        console: Console | None = None,
        tracker: CostTracker | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.settings = settings
        self.console = console or Console()
        self.tracker = tracker
        self.tool_failures: dict[str, int] = {}
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": (
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
            )}
        ]

    def run(self, prompt: str) -> str:
        """Run the agent core loop for a user prompt."""
        self.messages.append({"role": "user", "content": prompt})

        for iteration in range(self.settings.agent.max_iterations):
            schemas = self.registry.get_all_schemas()

            # Determine display label (router-aware or plain model name)
            model_label = self._model_label()

            with self.console.status(f"[bold blue]Calling {model_label}...[/bold blue]"):
                response = self.provider.complete(
                    prompt="",  # Ignored because messages is passed
                    messages=self.messages,
                    tools=schemas if schemas else None,
                )

            # --- Cost tracking ---
            if self.tracker is not None:
                usage = response.raw.get("usage") if response.raw else None
                rec = self.tracker.record(
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

            # Record assistant generation
            assistant_msg: dict[str, Any] = {}
            if response.raw and "message" in response.raw:
                # Retrieve raw assistant message with tool calls preserved accurately
                assistant_msg = response.raw["message"]
            else:
                assistant_msg = {"role": "assistant", "content": response.text}

            self.messages.append(assistant_msg)

            if response.text:
                self.console.print(self._badge())
                self.console.print(Markdown(response.text))

            if not response.tool_calls:
                return response.text

            # Execute tools sequentially
            for tool_call in response.tool_calls:
                # Use str() on args to prevent huge objects from breaking the UI
                args_str = str(tool_call.arguments)
                if len(args_str) > 100:
                    args_str = args_str[:97] + "..."

                tool_label = f"{tool_call.name}({args_str})"
                self.console.print(f"[dim]→ Tool call:[/dim] {tool_label}")

                with self.console.status(f"[bold cyan]Running {tool_call.name}...[/bold cyan]"):
                    try:
                        result = self.registry.dispatch(tool_call.name, tool_call.arguments)
                    except Exception as e:
                        result = f"Error: {e}"

                is_error = result.startswith("Error:") or "[ERROR]" in result or "Command exited with code" in result
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

                # Truncate output display for terminal, so it doesn't flood the UI
                display_result = result
                lines = display_result.splitlines()
                if len(lines) > 15:
                    display_result = "\\n".join(lines[:15]) + "\\n...[truncated output for display]"

                self.console.print(f"[dim]← Tool result:[/dim]\\n{display_result}")

        self.console.print(f"[bold red]Stopped:[/bold red] Hit max iterations ({self.settings.agent.max_iterations}).")
        return "Error: Agent reached maximum iterations without completing the task."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _model_label(self) -> str:
        """Return a plain-text model name for status spinners."""
        return getattr(self.provider, "current_model", getattr(self.provider, "model", "model"))

    def _badge(self) -> str:
        """Return a Rich markup string badge for the active model/tier."""
        # If the provider is a ModelRouter, use its formatted badge
        badge_fn = getattr(self.provider, "badge_text", None)
        if badge_fn is not None:
            return badge_fn()  # returns Rich Text
        model = getattr(self.provider, "model", "model")
        return f"[bold blue][{model}][/bold blue]"
